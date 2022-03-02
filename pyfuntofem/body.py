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

import numpy as np
from .base import Base

class Body(Base):
    """
    Defines a body base class for FUNtoFEM. Can be used as is or as a
    parent class for bodies with shape parameterization.
    """
    def __init__(self, name, analysis_type, id=0, group=None,
                 boundary=0, fun3d=True, motion_type='deform'):
        """

        Parameters
        ----------
        name: str
            name of the body
        id: int
            ID number of the body in the list of bodies in the model
        group: int
            group number for the body. Coupled variables defined in the body will be coupled with
            bodies in the same group
        boundary: int
            FUN3D boundary number associated with the body
        fun3d: bool
            whether or not you are using FUN3D. If true, the body class will auto-populate 'rigid_motion' required by FUN3D
        motion_type: str
            the type of motion the body is undergoing. Possible options: 'deform','rigid','deform+rigid','rigid+deform'

        See Also
        --------
        :mod:`base` : Body inherits from Base
        :mod:`massoud_body` : example of class inheriting Body and adding shape parameterization

        """

        from .variable import Variable as dv

        super(Body, self).__init__(name, id, group)

        self.name = name
        self.analysis_type = analysis_type
        self.id = id
        self.group = group
        self.group_root = False
        self.boundary = boundary
        self.motion_type = motion_type

        self.variables = {}
        self.derivatives = {}

        self.parent = None
        self.children = []

        if fun3d:
            self.add_variable('rigid_motion', dv('RotRate', active=False, id=1))
            self.add_variable('rigid_motion', dv('RotFreq', active=False, id=2))
            self.add_variable('rigid_motion', dv('RotAmpl', active=False, id=3))
            self.add_variable('rigid_motion', dv('RotOrgx', active=False, id=4))
            self.add_variable('rigid_motion', dv('RotOrgy', active=False, id=5))
            self.add_variable('rigid_motion', dv('RotOrgz', active=False, id=6))
            self.add_variable('rigid_motion', dv('RotVecx', active=False, id=7))
            self.add_variable('rigid_motion', dv('RotVecy', active=False, id=8))
            self.add_variable('rigid_motion', dv('RotVecz', active=False, id=9))
            self.add_variable('rigid_motion', dv('TrnRate', active=False, id=10))
            self.add_variable('rigid_motion', dv('TrnFreq', active=False, id=11))
            self.add_variable('rigid_motion', dv('TrnAmpl', active=False, id=12))
            self.add_variable('rigid_motion', dv('TrnVecx', active=False, id=13))
            self.add_variable('rigid_motion', dv('TrnVecy', active=False, id=14))
            self.add_variable('rigid_motion', dv('TrnVecz', active=False, id=15))

        # shape parameterization
        self.shape = None
        self.parameterization = 1

        # load and displacement transfer
        self.transfer = None

        # heat flux and temperature transfer
        self.thermal_transfer = None

        # Number of nodes
        self.struct_nnodes = None
        self.aero_nnodes = None

        # Number of degrees of freedom on the structures side of the transfer
        self.xfer_ndof = 3

        # Number of degrees of freedom on the thermal structural side of the transfer
        self.therm_xfer_ndof = 1
        self.T_ref = 300 # reference temperature in Kelvin

        # Node locations u
        self.struct_X = None
        self.aero_X = None

        # ID number of nodes
        self.struct_id = None
        self.aero_id = None

        # forward variables
        self.struct_disps = None
        self.struct_forces = None
        self.struct_loads = None
        self.struct_shape_term = None

        self.rigid_transform = None
        self.aero_disps  = None
        self.aero_forces = None
        self.aero_loads = None
        self.aero_shape_term = None

        self.struct_temps  = None
        self.struct_heat_flux = None

        self.aero_temps  = None
        self.aero_heat_flux = None

    def update_id(self, id):
        """
        **[model call]**
        Update the id number of the body or scenario

        Parameters
        ----------
        id: int
           id number of the scenario
        """
        self.id = id

        for vartype in self.variables:
            for var in self.variables[vartype]:
                var.body = self.id

    def add_variable(self, vartype, var):
        """
        Add a new variable to the body's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        var.body = self.id

        super(Body, self).add_variable(vartype, var)

    def collect_coordinate_derivatives(self, comm, discipline, root=0):
        """
        Write the sensitivity files for the aerodynamic and structural meshes on
        the root processor.

        This code collects the sensitivities from each processor and collects the
        result to the root node.
        """

        if discipline == 'aero':
            all_aero_ids = comm.gather(self.aero_id, root=root)
            all_aero_shape = comm.gather(self.aero_shape_term, root=root)

            aero_ids = []
            aero_shape = []

            if comm.rank == root:
                # Discard any entries that are None
                aero_ids = []
                for d in all_aero_ids:
                    if d is not None:
                        aero_ids.append(d)

                aero_shape = []
                for d in all_aero_shape:
                    if d is not None:
                        aero_shape.append(d)

                if len(aero_shape) > 0:
                    aero_shape = np.concatenate(aero_shape)
                else:
                    aero_shape = np.zeros((3, 1))

                if len(aero_ids) == 0:
                    aero_ids = np.arange(aero_shape.shape[0]//3, dtype=int)
                else:
                    aero_ids = np.concatenate(aero_ids)

            return aero_ids, aero_shape

        elif discipline == 'struct':
            all_struct_ids = comm.gather(self.struct_id, root=root)
            all_struct_shape = comm.gather(self.struct_shape_term, root=root)

            struct_ids = []
            struct_shape = []

            if comm.rank == root:
                # Discard any entries that are None
                struct_ids = []
                for d in all_struct_ids:
                    if d is not None:
                        struct_ids.append(d)

                struct_shape = []
                for d in all_struct_shape:
                    if d is not None:
                        struct_shape.append(d)

                if len(struct_shape) > 0:
                    struct_shape = np.concatenate(struct_shape)
                else:
                    struct_shape = np.zeros((3, 1))

                if len(struct_ids) == 0:
                    struct_ids = np.arange(struct_shape.shape[0]//3, dtype=int)
                else:
                    struct_ids = np.concatenate(struct_ids)

            return struct_ids, struct_shape

        return

    def initialize_shape_parameterization(self):
        """
        **[driver call]**
        Create the map of structural node locations since TACS doesn't give us global id numbers

        """

        return

    def update_shape(self, complex_run=False):
        """
        **[driver call]**
        perturbs the shape of the body based on the body's shape variables and update the meshes

        Parameters
        ----------
        complex_run: bool
            whether or not the run is complex mode

        """

        return

    def shape_derivative(self, scenario, offset):
        """
        **[driver call]**
        Calculates the shape derivative given the coordinate derivatives

        Parameters
        ----------
        scenario: scenario object
            The current scenario
        offset: int
            function offset number
        """

        return
