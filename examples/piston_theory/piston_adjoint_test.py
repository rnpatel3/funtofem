from __future__ import print_function
from turtle import width

import numpy as np
import os
#from fun3d.solvers      import Flow, Adjoint
#from fun3d              import interface
#from funtofem           import TransferScheme
#from .solver_interface   import SolverInterface

class PistonInterface():
    """
    FUNtoFEM interface class for FUN3D. Works for both steady and unsteady analysis.
    Requires the FUN3D directory structure.
    During the forward analysis, the FUN3D interface will operate in the scenario.name/Flow directory and scenario.name/Adjoint directory for the adjoint.

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """
    def __init__(self, qinf, M, U_inf, x0, length_dir, width_dir, L, w, nL, nw,
                 flow_dt=1.0,
                 forward_options=None, adjoint_options=None):
        """
        The instantiation of the FUN3D interface class will populate the model with the aerodynamic surface mesh, body.aero_X and body.aero_nnodes.
        The surface mesh on each processor only holds it's owned nodes. Transfer of that data to other processors is handled inside the FORTRAN side of FUN3D's FUNtoFEM interface.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model. This instantiatio
        flow_dt: float
            flow solver time step size. Used to scale the adjoint term coming into and out of FUN3D since FUN3D currently uses a different adjoint formulation than FUNtoFEM.
        """

        #  Instantiate FUN3D
        #self.fun3d_flow = Flow()
        #self.fun3d_adjoint = Adjoint()

        # command line options
        self.forward_options = forward_options
        self.adjoint_options = adjoint_options
                
        self.qinf = qinf # dynamic pressure
        self.M = M
        self.U_inf = U_inf
        self.gamma = 1.4
        self.x0 = x0
        self.length_dir = length_dir
        self.width_dir = width_dir
        self.L = L
        self.width = w
        self.nL = nL #num elems in xi direction
        self.nw = nw #num elems in eta direction

        self.n = np.cross(self.width_dir, self.length_dir) #Setup vector normal to plane

        self.w = []

        self.p = [] #Derivative verification perturbation
        self.cs = [] #CS approx of gradient

        # Get the initial aero surface meshes
        self.aero_nnodes = (self.nL+1) * (self.nw+1)
        self.aero_disps = np.zeros(self.aero_nnodes*3)
        self.aero_X = np.zeros(3*self.aero_nnodes)

        self.aero_loads = np.zeros(3*self.aero_nnodes)

        self.nmat = np.zeros((3*self.aero_nnodes, self.aero_nnodes))
        for i in range(self.aero_nnodes):
            self.nmat[3*i:3*i+3, i] = self.n
        
        #Setup central difference matrix
        self.CD_mat = np.zeros((self.aero_nnodes,self.aero_nnodes)) #Matrix for central difference
        diag_ones = np.ones(self.aero_nnodes-1)
        diag_neg = -np.ones(self.aero_nnodes-1)
        self.CD_mat += np.diag(diag_ones, 1)
        self.CD_mat += np.diag(diag_neg,-1)
        self.CD_mat[0][0] = -2
        self.CD_mat[0][1] = 2
        self.CD_mat[-1][-2] = -2
        self.CD_mat[-1][-1] = 2
        self.CD_mat *= 1/(2*self.L/self.nL)

        if self.aero_nnodes > 0:
            #Extracting node locations
            for i in range(self.nL+1):
                for j in range(self.nw+1):
                    coord = self.x0 + i*self.L/self.nL* self.length_dir + j*self.width/self.nw * self.width_dir
                    self.aero_X[3*(self.nw+1)*i + j*3] = coord[0]
                    self.aero_X[3*(self.nw+1)*i + j*3 + 1] = coord[1]
                    self.aero_X[3*(self.nw+1)*i + j*3 + 2] = coord[2]
    
    def iterate(self):
        """
        Forward iteration of Piston Theory.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the piston theory flow solver.
        #. Set the aerodynamic forces into the body data types


        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """
        
        # Deform aerodynamic mesh
        
        #Compute aero displacements, divide into dx,dy,dz
        '''
        dx = body.aero_disps[0::3]
        dy = body.aero_disps[1::3]
        dz = body.aero_disps[2::3]

        disps = np.stack((dx,dy,dz), axis=1)
        self.w = disps@np.transpose(self.n)
        '''

        #Setup perturbation for CS verification
        dh = 1e-30
        self.p = np.random.uniform(size=self.aero_nnodes*3)
        new_aero_disps = self.aero_disps + 1j*dh*self.p
        #new_aero_disps = self.aero_disps


        
        '''
        aero_forces = np.multiply(press_i, areas)

        body.aero_loads[0::3] = aero_forces*self.n[0]
        body.aero_loads[1::3] = aero_forces*self.n[1]
        body.aero_loads[2::3] = aero_forces*self.n[2]
        '''

        psi_P = 0.05*np.ones(self.aero_nnodes*3) #Dummy adjoint vars
        self.cs = psi_P.T@(((self.nmat@np.diag(areas)@press_i).imag)/dh) 
        print("complex step: ", self.cs)
        #psi_P = 0.05*np.ones(self.aero_nnodes*3)
        #forward_mat = self.nmat@np.diag(areas)@press_i
        #forward_mat = self.nmat@np.diag(areas)@np.diag(press_i)@self.CD_mat@self.nmat.T
        #self.cs = psi_P.T@self.cs 
        

        return

    def compute_forces(self, aero_disps, aero_loads):
        #Compute w for piston theory: [dx,dy,dz] DOT planarNormal

        w = self.nmat.T@aero_disps

        # Compute body.aero_loads using Piston Theory:

        #First compute dw/dxi
        dw_dxi = self.CD_mat@w

        #Set dw/dt = 0  for now (steady)
        dw_dt = np.zeros(self.aero_nnodes)

        #Call function to compute pressure
        press_i = self.compute_Pressure(dw_dxi, dw_dt)
        
        #Compute forces from pressure
        areas = self.compute_Areas()
        aero_loads[:] = self.nmat@np.diag(areas)@press_i
        return

    def compute_forces_adjoint(self, aero_disps, adjoint_loads, adjoint_disps):
        '''
        w = self.nmat.T@aero_disps
        dw_dxi = self.CD_mat@w
        #Set dw/dt = 0  for now (steady)
        dw_dt = np.zeros(self.aero_nnodes)
        areas = self.compute_Areas()
        press_adj = np.diag(areas).T@self.nmat.T@adjoint_loads #pressure adjoint
        #print("press_adj: \n", press_adj)
        dwdxi_adj, dwdt_adj = self.compute_Pressure_adjoint(dw_dxi, dw_dt, press_adj)
        #print("dwdxi_adj: \n", dwdxi_adj)
        w_adj = self.CD_mat.T@dwdxi_adj
        #print("w_adj: \n", w_adj)
        adjoint_disps[:] = self.nmat@w_adj
        #print("adjoint_disps: \n", adjoint_disps)
        '''
        w = self.nmat.T@aero_disps
        dw_dxi = self.CD_mat@w
        dw_dt = np.zeros(self.aero_nnodes)
        areas = self.compute_Areas()

        dwdxi_adj = self.compute_Pressure_deriv(dw_dxi, dw_dt)
        adjoint_disps[:] = self.nmat@np.diag(areas)@np.diag(dwdxi_adj)@self.CD_mat@self.nmat.T
        
        return
    
    def compute_Pressure_adjoint(self, dw_dxi, dw_dt, press_adj):
        d_press_dxi = self.compute_Pressure_deriv(dw_dxi, dw_dt)
        dw_dxi_adjoint = np.diag(d_press_dxi)@press_adj #Verify this is a component wise product
        return dw_dxi_adjoint, None
    
    def compute_Pressure(self, dw_dxi, dw_dt):
        '''
        Returns 'pressure' values at each node location
        '''
        press = 2*self.qinf/self.M * ((1/self.U_inf * dw_dt + dw_dxi) +
         (self.gamma+1)/4*self.M*(1/self.U_inf * dw_dt + dw_dxi)**2 + 
         (self.gamma+1)/12*self.M**2 * (1/self.U_inf* dw_dt + dw_dxi)**3)
        
        return press

    def compute_Pressure_deriv(self, dw_dxi, dw_dt):
        '''
        Returns partial derivatives 'pressure' values at each node location
        with respect to dw_dxi
        '''
        d_press_dwdxi = 2*self.qinf/self.M * ((1) +
         (self.gamma+1)/4*self.M*2*(1/self.U_inf * dw_dt + dw_dxi)*(1) + 
         (self.gamma+1)/12*self.M**2 * 3 * (1/self.U_inf* dw_dt + dw_dxi)**2 * (1))
        
        return d_press_dwdxi

    def compute_Areas(self):
        '''
        Computes area corresponding to each node (calculations based on rectangular
        evenly spaced mesh grid)
        '''
        area_array = (self.L/self.nL)*(self.width/self.nw) * np.ones(self.aero_nnodes) #Array of area corresponding to each node
        area_array[0:self.nw+1] *= 0.5
        area_array[-1:-self.nw-2:-1] *= 0.5
        area_array[0::self.nw+1] *= 0.5
        area_array[self.nw::self.nw+1] *= 0.5

        return area_array

    def iterate_adjoint(self):
        """
        Adjoint iteration of Piston Theory.
        For the aeroelastic cases, these steps are:

        #. Get the force adjoint from the body data structures
        #. Step in the piston theory adjoint solvers
        #. Set the piston theory adjoint into the body data structures

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the forward time step number
        """

        # fail = 0
        # rstep = scenario.steps - step + 1
        # if scenario.steady:
        #     rstep = step

        #nfunctions = scenario.count_adjoint_functions()

       
        if self.aero_nnodes > 0:
            # Solve the force adjoint equation
            psi_P = 0.05*np.ones(self.aero_nnodes*3) #Dummy adjoint vars
        
        
            # Extract the equivalent of dG/du_a^T psi_G from Piston Theory (dP/du_a^T psi_P)
        
        self.w = self.nmat.T@self.aero_disps
        dw_dxi = self.CD_mat@self.w
        dw_dt = np.zeros(self.aero_nnodes)
        dPress_ddw_dxi = self.compute_Pressure_deriv(dw_dxi, dw_dt)

        areas = self.compute_Areas()

        #dP/du_a solved via reverse chain rule
        #dPdua = self.nmat@np.diag(areas).T@np.diag(dPress_ddw_dxi)@self.CD_mat.T@self.nmat.T
        dPdua = self.nmat@self.CD_mat.T@np.diag(dPress_ddw_dxi)@np.diag(areas).T@self.nmat.T


        adjoint_result = self.p.T@dPdua.T@psi_P
        print("Derivative adjoint_result: ", adjoint_result)
        
        #for func in range(nfunctions):
        #    body.dGdua[:, func] = dPdua.T@psi_P.flatten()

        return adjoint_result


qinf = 101325.0 # freestream pressure Pa
M = 1.2     # Mach number
U_inf = 411 #Freestream velocity m/s
x0 = np.array([1,1,1])
alpha = 10  #Angle of attack (degrees)
length_dir = np.array([np.cos(alpha*np.pi/180), 0, np.sin(alpha*np.pi/180)]) #Unit vec in length dir
width_dir = np.array([0, 1, 0])     #Unit vec in width dir
L = 2.0 #Length
nL = 10 # Num elems in xi dir
w = 3.0  #Width
nw = 10 # Num elems in eta dir
piston = PistonInterface(qinf, M, U_inf, x0, length_dir, width_dir, L, w, nL, nw)
dh = 1e-30
aero_disps = np.random.uniform(low=0.0, high=0.1, size=piston.aero_nnodes*3)
p = 0.01*np.random.uniform(low=0.0, high=0.1, size=piston.aero_nnodes*3)
aero_loads = np.zeros(piston.aero_nnodes*3)
adjoint_disps = np.zeros((piston.aero_nnodes*3,piston.aero_nnodes*3))
piston.compute_forces(aero_disps, aero_loads)
#print("aero loads: ", aero_loads)
piston.compute_forces_adjoint(aero_disps, aero_loads, adjoint_disps)
dfdua = adjoint_disps@p
aero_loads = np.zeros(piston.aero_nnodes*3, dtype=complex)
piston.compute_forces(aero_disps+1j*dh*p, aero_loads)
cs = (aero_loads.imag/dh)

print('dfadua: ',dfdua, 'CS: ', cs, 'Rel. Error: ', 
        np.abs((dfdua - cs)/cs))



#piston.iterate()
#adjoint_result = piston.iterate_adjoint()

#print('dfadua: ',adjoint_result, 'CS: ', piston.cs, 'Rel. Error: ', 
#        np.abs((adjoint_result - piston.cs)/piston.cs))

'''
dh = 1e-30
dw_dxi = 0.05*np.ones(piston.aero_nnodes)
dw_dt = np.zeros(piston.aero_nnodes)
aero_disps = 0.5*np.ones(piston.aero_nnodes*3)
p = 0.01*np.ones(piston.aero_nnodes)
press_cs = piston.compute_Pressure(dw_dxi+1j*dh*p, dw_dt).imag/dh   #verifying dPress/ddwdxi
adjoint_result = np.dot(np.diag(piston.compute_Pressure_deriv(dw_dxi, dw_dt)), p)
#press_cs = piston.computeW(aero_disps+1j*dh*p).imag/dh
#adjoint_result = np.dot(piston.computeW_deriv(), p)

print('dfadua: ',adjoint_result, 'CS: ', press_cs, 'Rel. Error: ', 
        np.abs((adjoint_result - press_cs)/press_cs))
'''
