#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function
from turtle import width

import numpy as np
import os
#from fun3d.solvers      import Flow, Adjoint
#from fun3d              import interface
from funtofem           import TransferScheme
from .solver_interface   import SolverInterface

class PistonInterface(SolverInterface):
    """
    FUNtoFEM interface class for FUN3D. Works for both steady and unsteady analysis.
    Requires the FUN3D directory structure.
    During the forward analysis, the FUN3D interface will operate in the scenario.name/Flow directory and scenario.name/Adjoint directory for the adjoint.

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """
    def __init__(self, comm, model, qinf, M, U_inf, x0, length_dir, width_dir, L, w, nL, nw,
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

        self.comm = comm

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
        
        self.CD_mat = []
        self.nmat = []
        self.aero_nnodes = []

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        #self.post(model.scenarios[0], model.bodies, first_pass=True)

        # Temporary measure until FUN3D adjoint is reformulated
        #self.flow_dt = flow_dt

        self.dFdqinf = []

        # heat flux
        self.thermal_scale = 1.0 # = 1/2 * rho_inf * (V_inf)^3
        self.dHdq = []

        # multiple steady scenarios
        self.force_save = {}
        self.disps_save = {}
        self.heat_flux_save = {}
        self.heat_flux_mag_save = {}
        self.temps_save = {}

        # unsteady scenarios
        self.force_hist = {}
        self.heat_flux_hist = {}
        self.heat_flux_mag_hist = {}
        self.aero_temps_hist = {}
        for scenario in model.scenarios:
            self.force_hist[scenario.id] = {}
            self.heat_flux_hist[scenario.id] = {}
            self.heat_flux_mag_hist[scenario.id] = {}
            self.aero_temps_hist[scenario.id] = {}

    def initialize(self, scenario, bodies, first_pass=False):
        """
        Changes the directory to ./`scenario.name`/Flow, then
        initializes the FUN3D flow (forward) solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to set the original mesh in
        first_pass: bool
            When extracting the mesh, first_pass is set to True. Otherwise, the new mesh will be set in for bodies with shape parameterizations

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """
        #os.chdir("./" + scenario.name)
        #os.chdir("./Flow")
        # Do the steps to initialize FUN3D - NOT NEEDED, NO FUN3D
        #self.fun3d_flow.initialize_project(comm=self.comm)

        if self.forward_options is None:
            options = {}
        else:
            options = self.forward_options
        #self.fun3d_flow.setOptions(kwargs=options)

        #self.fun3d_flow.initialize_data()
        #interface.design_initialize()
        #self.fun3d_flow.initialize_grid()

        # During the first pass we don't have any meshes yet
        if not first_pass:
            for ibody,body in enumerate(bodies,1):
                self.aero_nnodes = body.aero_nnodes
                if body.shape and body.aero_nnodes > 0:
                    aero_X = np.reshape(body.aero_X, (3,-1), order='F')
                    #interface.design_push_body_mesh(ibody, aero_X, body.aero_id)
                    #interface.design_push_body_name(ibody, body.name)
                #else:
                    #interface.design_push_body_mesh(ibody, [], [])
                    #interface.design_push_body_name(ibody, body.name)

        '''
        bcont = self.fun3d_flow.initialize_solution()

        if bcont == 0:
            if self.comm.Get_rank()==0:
                print("Negative volume returning fail")
            return 1
        '''
        
        if first_pass:
            for ibody, body in enumerate(bodies,1):
                body.aero_nnodes = (self.nL+1) * (self.nw+1)
                body.aero_X = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                
                self.CD_mat = np.zeros((body.aero_nnodes,body.aero_nnodes)) #Matrix for central difference
                diag_ones = np.ones(body.aero_nnodes-1)
                diag_neg = -np.ones(body.aero_nnodes-1)
                self.CD_mat += np.diag(diag_ones, 1)
                self.CD_mat += np.diag(diag_neg,-1)
                self.CD_mat[0][0] = -2
                self.CD_mat[0][1] = 2
                self.CD_mat[-1][-2] = -2
                self.CD_mat[-1][-1] = 2
                self.CD_mat *= 1/(2*self.L/self.nL)
                
                self.nmat = np.zeros((3*body.aero_nnodes, body.aero_nnodes))
                for i in range(body.aero_nnodes):
                    self.nmat[3*i:3*i+3, i] = self.n
                
                if body.aero_nnodes > 0:
                    body.aero_id = np.arange(1,body.aero_nnodes)
                    body.aero_loads = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                    #Extracting node locations
                    for i in range(self.nL+1):
                        for j in range(self.nw+1):
                            coord = self.x0 + i*self.L/self.nL* self.length_dir + j*self.width/self.nw * self.width_dir
                            body.aero_X[3*(self.nw+1)*i + j*3] = coord[0]
                            body.aero_X[3*(self.nw+1)*i + j*3 + 1] = coord[1]
                            body.aero_X[3*(self.nw+1)*i + j*3 + 2] = coord[2]

                else:
                    body.aero_id = np.zeros(3*body.aero_nnodes, dtype=int)

                body.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)

        return 0

    def initialize_adjoint(self, scenario, bodies):
        """
        Changes the directory to ./`scenario.name`/Adjoint, then
        initializes the FUN3D adjoint solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get surface meshes from

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """
        #os.chdir("./" + scenario.name)
        #os.chdir("./Adjoint")
        if scenario.steady:
            # load the forces and displacements
            for ibody, body in enumerate(bodies):
                if body.transfer is not None:
                    body.aero_loads = self.force_save[scenario.id][ibody]
                    body.aero_disps = self.disps_save[scenario.id][ibody]

                if body.thermal_transfer is not None:
                    body.aero_heat_flux = self.heat_flux_save[scenario.id][ibody]
                    body.aero_heat_flux_mag = self.heat_flux_mag_save[scenario.id][ibody]
                    body.aero_temps = self.temps_save[scenario.id][ibody]

            # Deform the aero mesh before finishing FUN3D initialization (Probably unnecessary for PT)
            '''
            if body.aero_nnodes > 0:
                for ibody, body in enumerate(bodies,1):
                    if body.transfer is not None:
                        dx = np.asfortranarray(body.aero_disps[0::3])
                        dy = np.asfortranarray(body.aero_disps[1::3])
                        dz = np.asfortranarray(body.aero_disps[2::3])
                        self.fun3d_adjoint.input_deformation(dx, dy, dz,body=ibody)
                    if body.thermal_transfer is not None:
                        temps = np.asfortranarray(body.aero_temps[:])/body.T_ref
                        self.fun3d_adjoint.input_wall_temperature(temps, body=ibody)
            self.fun3d_adjoint.initialize_solution()
            '''
        else:
            if self.adjoint_options is None:
                options = {'timedep_adj_frozen': True}
            else:
                options = self.adjoint_options
            self.fun3d_adjoint.initialize_project(comm=self.comm)
            self.fun3d_adjoint.setOptions(kwargs=options)
            self.fun3d_adjoint.initialize_data()
            interface.design_initialize()
            for ibody, body in enumerate(bodies,1):
                if body.shape and body.aero_nnodes > 0:
                    aero_X = np.reshape(body.aero_X,(3,-1), order='F')
                    interface.design_push_body_mesh(ibody, aero_X, body.aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody,[],[])
                    interface.design_push_body_name(ibody, body.name)
            self.fun3d_adjoint.initialize_grid()
            self.fun3d_adjoint.initialize_solution()

        self.dFdqinf = np.zeros(len(scenario.functions))
        self.dHdq = np.zeros(len(scenario.functions))

        return 0

    def set_functions(self, scenario, bodies):
        """
        Set the function definitions into FUN3D using the design interface.
        Since FUNtoFEM only allows single discipline functions, the FUN3D composite function is the same as the component.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the function list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        pass 
        '''
        for function in scenario.functions:
            if function.adjoint:
                start =  1 if function.stop==-1 else function.start
                stop  =  1 if function.stop==-1 else function.stop
                ftype = -1 if function.averaging else 1
                interface.design_push_composite_func(function.id,
                                                     1,
                                                     start,
                                                     stop,
                                                     1.0,
                                                     0.0,
                                                     1.0,
                                                     function.value,
                                                     ftype,
                                                     100.0,
                                                     -100.0)

                if function.body ==-1:
                    boundary = 0
                else:
                    boundary = bodies[function.body].boundary

                # The funtofem function in FUN3D acts as any adjoint function
                # that isn't dependent on FUN3D variables
                name = function.name if function.analysis_type == 'aerodynamic' else 'funtofem'

                interface.design_push_component_func(function.id,
                                                     1,
                                                     boundary,
                                                     name,
                                                     function.value,
                                                     1.0,
                                                     0.0,
                                                     1.0)
        '''
        return

    def set_variables(self, scenario, bodies):
        """
        Set the aerodynamic variable definitions into FUN3D using the design interface.
        FUN3D expects 6 global variables (Mach number, AOA, yaw, etc.) that are stored in the scenario.
        It also expects a set of rigid motion variables for each body that are stored in the body.
        If the body has been specific as *motion_driver(i)='funtofem'*, the rigid motion variables will not affect the body's movement but must be passed regardless.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """
        pass

    '''
        interface.design_set_design(len(bodies), scenario.count_adjoint_functions())

        # push the global aerodynamic variables to fun3d
        for var in scenario.variables['aerodynamic']:
            if var.id <= 6:
                interface.design_push_global_var(var.id, var.active, var.value,
                                                 var.lower, var.upper)
            elif 'dynamic pressure' == var.name.lower():
                self.qinf = var.value
            elif 'thermal scale' == var.name.lower():
                self.thermal_scale = var.value

        # push the push the shape and rigid motion variables
        for ibody, body in enumerate(bodies,1):
            num = len(body.variables['shape']) if 'shape' in body.variables else 1
            interface.design_set_body(ibody, body.parameterization, num)
            if 'shape' in body.variables:
                for var in body.variables['shape']:
                    interface.design_push_body_shape_var(ibody, var.id, var.active,
                                                         var.value, var.lower, var.upper)

            for var in body.variables['rigid_motion']:
                interface.design_push_body_rigid_var(ibody, var.id, var.name, var.active,
                                                     var.value, var.lower, var.upper)
        '''
    def get_functions(self, scenario, bodies):
        """
        Populate the scenario with the aerodynamic function values.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        
        for function in scenario.functions:
            if function.analysis_type == 'aerodynamic':
                # the [6] index returns the value
                if self.comm.Get_rank() == 0:
                    if function.name == 'cl':
                        function.value = self.compute_cl(scenario, bodies)
                function.value = self.comm.bcast(function.value, root=0)
        
        return

    def compute_cl(self, scenario, bodies):
        for ibody, body in enumerate(bodies,1):
            lift = np.sum(body.aero_loads) 
            cl = lift/(self.L * self.width)
        
        return cl

    def get_function_gradients(self, scenario, bodies, offset):
        """
        Populates the FUNtoFEM model with derivatives w.r.t. aerodynamic variables

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        offset: int
            offset of the scenario's function index w.r.t the full list of functions in the model.
        """

        for func, function in enumerate(scenario.functions):
            # Do the scenario variables first
            for vartype in scenario.variables:
                if vartype == 'aerodynamic':
                    for i, var in enumerate(scenario.variables[vartype]):
                        if var.active:
                            if function.adjoint:
                                if var.id<=6:
                                    scenario.derivatives[vartype][offset+func][i] = interface.design_pull_global_derivative(function.id,var.id)
                                elif var.name.lower() == 'dynamic pressure':
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dFdqinf[func])
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dHdq[func])
                            else:
                                scenario.derivatives[vartype][offset+func][i] = 0.0
                            scenario.derivatives[vartype][offset+func][i] = self.comm.bcast(scenario.derivatives[vartype][offset+func][i],root=0)


            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'rigid_motion':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                # rigid motion variables are not active in funtofem path
                                body.derivatives[vartype][offset+func][i] = 0.0

        return scenario, bodies

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Adds FUN3D's contribution to the aerodynamic surface coordinate derivatives.
        This is just the grid adjoint variable, $\lambda_G$.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario.
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """
        nfunctions = scenario.count_adjoint_functions()
        for ibody, body in enumerate(bodies,1):
            if body.shape and body.aero_nnodes > 0:
                # Aero solver contribution = dGdxa0^T psi_G
                body.aero_id = self.fun3d_adjoint.extract_surface_id(body.aero_nnodes,body=ibody)

                dGdxa0_x, dGdxa0_y, dGdxa0_z = self.fun3d_adjoint.extract_grid_adjoint_product(body.aero_nnodes,
                                                                                               nfunctions,body=ibody)

                body.aero_shape_term[ ::3,:nfunctions] += dGdxa0_x[:,:] * self.flow_dt
                body.aero_shape_term[1::3,:nfunctions] += dGdxa0_y[:,:] * self.flow_dt
                body.aero_shape_term[2::3,:nfunctions] += dGdxa0_z[:,:] * self.flow_dt

    def iterate(self, scenario, bodies, step):
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
        
        # Calculate aero_loads from aero_disps
        for ibody, body in enumerate(bodies,1):
            self.compute_forces(body.aero_disps, body.aero_loads)
            #Write Loads to File at the last step
            if step == scenario.steps:
                file = open("NodalForces_redo_M1_2.txt", 'w')
                np.savetxt(file, body.aero_loads)
                file.close()
        
        return 0

    def compute_forces(self, aero_disps, aero_loads):

        #Compute w for piston theory: [dx,dy,dz] DOT planarNormal
        w = self.nmat.T@aero_disps

        ####### Compute body.aero_loads using Piston Theory ######
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
        w = self.nmat.T@aero_disps
        dw_dxi = self.CD_mat@w
        #Set dw/dt = 0  for now (steady)
        dw_dt = np.zeros(self.aero_nnodes)
        areas = self.compute_Areas()

        press_adj = np.diag(areas).T@self.nmat.T@adjoint_loads #pressure adjoint
        dwdxi_adj, dwdt_adj = self.compute_Pressure_adjoint(dw_dxi, dw_dt, press_adj)
        w_adj = self.CD_mat.T@dwdxi_adj
        adjoint_disps[:] = self.nmat@w_adj
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

    def post(self, scenario, bodies, first_pass=False):
        """
        Calls FUN3D post to save history files, deallocate memory etc.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        first_pass: bool
            Set to true during instantiation
        """
        #self.fun3d_flow.post()
        #os.chdir("../..")

        # save the forces for multiple scenarios if steady
        if scenario.steady and not first_pass:
            self.force_save[scenario.id] = {}
            self.disps_save[scenario.id] = {}
            self.heat_flux_save[scenario.id] = {}
            self.heat_flux_mag_save[scenario.id] = {}
            self.temps_save[scenario.id] = {}

            for ibody, body in enumerate(bodies):
                if body.transfer is not None:
                    self.force_save[scenario.id][ibody] = body.aero_loads
                    self.disps_save[scenario.id][ibody] = body.aero_disps
                if body.thermal_transfer is not None:
                    self.heat_flux_save[scenario.id][ibody] = body.aero_heat_flux
                    self.heat_flux_mag_save[scenario.id][ibody] = body.aero_heat_flux_mag
                    self.temps_save[scenario.id][ibody] = body.aero_temps

    def set_states(self, scenario, bodies, step):
        """
        Loads the saved aerodynamic forces for the time dependent adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """
        for ibody, body in enumerate(bodies,1):
            if body.transfer is not None:
                body.aero_loads = self.force_hist[scenario.id][step][ibody]
            if body.thermal_transfer is not None:
                body.aero_heat_flux = self.heat_flux_hist[scenario.id][step][ibody]
                body.aero_heat_flux_mag = self.heat_flux_hist_mag[scenario.id][step][ibody]
                body.aero_temps = self.aero_temps_hist[scenario.id][step][ibody]

    def iterate_adjoint(self, scenario, bodies, step):
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

        fail = 0
        rstep = scenario.steps - step + 1
        if scenario.steady:
            rstep = step

        nfunctions = scenario.count_adjoint_functions()

        for ibody, body in enumerate(bodies,1):
            if body.aero_nnodes > 0:
                # Solve the force adjoint equation
                if body.transfer is not None:
                    psi_P = - body.dLdfa
        
        for ibody, body in enumerate(bodies, 1):
            # Extract the equivalent of dG/du_a^T psi_G from Piston Theory (dP/du_a^T psi_P)
            if body.transfer is not None:
                dPdua = np.zeros((body.aero_nnodes*3, body.aero_nnodes*3), dtype=TransferScheme.dtype)
                self.compute_forces_adjoint(body.aero_disps, body.aero_loads, dPdua)

                for func in range(nfunctions):
                    body.dGdua[:, func] = dPdua.T@psi_P.flatten()
                
        return fail

    def post_adjoint(self, scenario, bodies):
        """
        Calls post fo the adjoint solver in FUN3D.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        """

        # solve the initial condition adjoint
        #self.fun3d_adjoint.post()
        #os.chdir("../..")

    def step_pre(self, scenario, bodies, step):
        self.fun3d_flow.step_pre(step)
        return 0

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        """
        Forward iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
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
        for ibody, body in enumerate(bodies, 1):
            if 'deform' in body.motion_type and body.aero_nnodes > 0 and body.transfer is not None:
                dx = np.asfortranarray(body.aero_disps[0::3])
                dy = np.asfortranarray(body.aero_disps[1::3])
                dz = np.asfortranarray(body.aero_disps[2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)
            if 'rigid' in body.motion_type and body.transfer is not None:
                self.fun3d_flow.input_rigid_transform(body.rigid_transform, body=ibody)
            if body.thermal_transfer is not None:
                temps = np.asfortranarray(body.aero_temps[:])/body.T_ref
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.step_solver()
        if bcont == 0:
            if self.comm.Get_rank()==0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir("../..")
            return fail

        # Pull out the forces from FUN3D
        for ibody, body in enumerate(bodies, 1):
            if body.aero_nnodes > 0:
                if body.transfer is not None:
                    fx, fy, fz = self.fun3d_flow.extract_forces(body.aero_nnodes, body=ibody)
                    body.aero_loads = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_loads[0::3] = fx[:]
                    body.aero_loads[1::3] = fy[:]
                    body.aero_loads[2::3] = fz[:]

                if body.thermal_transfer is not None:
                    cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_heat_flux(body.aero_nnodes, body=ibody)
                    body.aero_heat_flux = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_heat_flux_mag = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_heat_flux[0::3] = self.thermal_scale * cqx[:]
                    body.aero_heat_flux[1::3] = self.thermal_scale * cqy[:]
                    body.aero_heat_flux[2::3] = self.thermal_scale * cqz[:]
                    body.aero_heat_flux_mag[:] = self.thermal_scale * cq_mag[:]
        return 0

    def step_post(self,scenario,bodies,step):
        self.fun3d_flow.step_post(step)

        # save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        for ibody, body in enumerate(bodies,1):
            if body.transfer is not None:
                self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()

            if body.thermal_transfer is not None:
                self.heat_flux_hist[scenario.id][step][ibody] = body.aero_heat_flux.copy()
                self.heat_flux_mag_hist[scenario.id][step][ibody] = body.aero_heat_flux_mag.copy()
                self.aero_temps_hist[scenario.id][step][ibody] = body.aero_temps.copy()
        return 0
