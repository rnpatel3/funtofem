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
from funtofem import TransferScheme
from pyfuntofem.solver_interface import SolverInterface


class TestAerodynamicSolver(SolverInterface):
    def __init__(self, comm, model):
        """
        A test solver that provides the functionality that FUNtoFEM expects from
        an aerodynamic solver.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        self.comm = comm
        self.npts = 10
        np.random.seed(0)

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Count the number of aerodynamic design variables (if any)
        self.aero_variables = []  # List of the variable objects

        # List of the variable values - converted into an numpy array
        self.aero_dvs = []

        for var in self.variables:
            if var.analysis_type == "aerodynamic":
                self.aero_variables.append(var)
                self.aero_dvs.append(var.value)

        # Allocate space for the aero dvs
        self.aero_dvs = np.array(self.aero_dvs, dtype=TransferScheme.dtype)

        # Aerodynaimic forces = Jac1 * aero_disps + b1 * aero_X + c1 * aero_dvs
        self.Jac1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.b1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.c1 = 0.01 * (np.random.rand(3 * self.npts, len(self.aero_dvs)) - 0.5)

        # Aero heat flux = Jac2 * aero_temps + b2 * aero_X + c2 * aero_dvs
        self.Jac2 = 0.05 * (np.random.rand(self.npts, self.npts) - 0.5)
        self.b2 = 0.1 * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
        self.c2 = 0.01 * (np.random.rand(self.npts, len(self.aero_dvs)) - 0.5)

        # Set random initial node locations
        self.aero_X = np.random.rand(3 * self.npts)

        # Data for generating functional output values
        self.func_coefs1 = np.random.rand(3 * self.npts)
        self.func_coefs2 = np.random.rand(self.npts)

        # Initialize the coordinates of the aerodynamic or structural mesh
        for body in model.bodies:
            body.initialize_aero_mesh(self.aero_X)

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def set_variables(self, scenario, bodies):
        """Set the design variables for the solver"""

        # Set the aerodynamic design variables
        index = 0
        for var in self.variables:
            if var.analysis_type == "aerodynamic":
                self.aero_dvs[index] = var.value
                index += 1

        return

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        if scenario.steady:
            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "aerodynamic":
                    value = 0.0
                    for body in bodies:
                        aero_disps = body.get_aero_disps(scenario)
                        if aero_disps is not None:
                            value += np.dot(self.func_coefs1, aero_disps)
                        aero_temps = body.get_aero_temps(scenario)
                        if aero_temps is not None:
                            value += np.dot(self.func_coefs2, aero_temps)
                    func.value = self.comm.allreduce(value)
        else:
            # Set the time index to the final time step
            time_index = scenario.steps

            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "aerodynamic":
                    value = 0.0
                    for body in bodies:
                        aero_disps = body.get_aero_disps(scenario, time_index)
                        if aero_disps is not None:
                            value += np.dot(self.func_coefs1, aero_disps)
                        aero_temps = body.get_aero_temps(scenario, time_index)
                        if aero_temps is not None:
                            value += np.dot(self.func_coefs2, aero_temps)
                    func.value = self.comm.allreduce(value)

        return

    def eval_function_gradients(self, scenario, bodies):
        """
        Return the function gradients
        """

        # Set the derivatives of the functions for the given scenario
        for func in scenario.functions:
            for var in self.aero_variables:
                value = 0.0
                func.add_derivative(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):

        return

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            # Perform the "analysis"
            # Note that the body class expects that you will write the new
            # aerodynamic loads into the aero_loads array.
            aero_disps = body.get_aero_disps(scenario)
            aero_loads = body.get_aero_loads(scenario)
            if aero_disps is not None:
                aero_loads[:] = np.dot(self.Jac1, aero_disps)
                aero_loads[:] += np.dot(self.b1, self.aero_X)
                aero_loads[:] += np.dot(self.c1, self.aero_dvs)

            # Perform the heat transfer "analysis"
            aero_temps = body.get_aero_temps(scenario)
            aero_flux = body.get_aero_heat_flux(scenario)
            if aero_temps is not None:
                aero_flux[:] = np.dot(self.Jac2, aero_temps)
                aero_flux[:] = np.dot(self.b2, self.aero_X)
                aero_flux[:] = np.dot(self.c2, self.aero_dvs)

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def set_states(self, scenario, bodies, step):
        pass

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        # We solve for all the adjoints simultaneously
        nfunctions = scenario.count_adjoint_functions()

        # if self.solver == "aerodynamic":
        #     # Solve for the flow adjoint
        #     for body in bodies:
        #         psi_F = body.get_aero_loads_adjoint(scenario)
        #         adjD_rhs = body.get
        #         adjD_rhs = - np.dot(self.Jac1.T, psi_F)

        # if self.solver == "structural":
        #     for body in bodies:

        # if self.solver == "aerodynamic":
        #     for body in bodies:
        #         # Perform the "analysis"
        #         # Note that the body class expects that you will write the new
        #         # aerodynamic loads into the aero_loads array.
        #         aero_disps = body.get_aero_disps(scenario)
        #         aero_loads = body.get_aero_loads(scenario)
        #         if aero_disps is not None:
        #             aero_loads[:] = np.dot(self.Jac1, aero_disps) + self.b1

        #         # Perform the heat transfer "analysis"
        #         aero_temps = body.get_aero_temps(scenario)
        #         aero_flux = body.get_aero_heat_flux(scenario)
        #         if aero_temps is not None:
        #             aero_flux[:] = np.dot(self.Jac2, aero_temps) + self.b2
        # else:  # The solver is a structural solver
        #     for body in bodies:
        #         # Perform the "analysis"
        #         struct_loads = body.get_struct_loads(scenario)
        #         struct_disps = body.get_struct_disps(scenario)
        #         if struct_loads is not None:
        #             struct_disps[:] = np.dot(self.Jac1, struct_loads) + self.b1

        #         # Perform the heat transfer "analysis"
        #         struct_flux = body.get_struct_heat_flux(scenario)
        #         struct_temps = body.get_struct_temps(scenario)
        #         if struct_flux is not None:
        #             struct_temps[:] = np.dot(self.Jac2, struct_flux) + self.b2

        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        pass

    def step_pre(self, scenario, bodies, step):
        return 0

    def step_post(self, scenario, bodies, step):
        return 0


class TestStructuralSolver(SolverInterface):
    def __init__(self, comm, model, solver="aerodynamic"):
        """
        A test solver that provides the functionality that FUNtoFEM expects from
        a structural solver.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        self.comm = comm
        self.npts = 25
        np.random.seed(54321)

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Count the number of structural design variables (if any)
        self.struct_variables = []  # List of the variable objects

        # List of the variable values - converted into an numpy array
        self.struct_dvs = []

        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)
                self.struct_dvs.append(var.value)

        # Allocate space for the aero dvs
        self.struct_dvs = np.array(self.struct_dvs, dtype=TransferScheme.dtype)

        # Struct disps = Jac1 * struct_forces + b1 * struct_X + c1 * struct_dvs
        self.Jac1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.b1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.c1 = 0.01 * (np.random.rand(3 * self.npts, len(self.struct_dvs)) - 0.5)

        # Struct temps = Jac2 * struct_flux + b2 * struct_X + c2 * struct_dvs
        self.Jac2 = 0.05 * (np.random.rand(self.npts, self.npts) - 0.5)
        self.b2 = 0.1 * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
        self.c2 = 0.01 * (np.random.rand(self.npts, len(self.struct_dvs)) - 0.5)

        # Set random initial node locations
        self.struct_X = np.random.rand(3 * self.npts)

        # Data for output functional values
        self.func_coefs1 = np.random.rand(3 * self.npts)
        self.func_coefs2 = np.random.rand(self.npts)

        # Initialize the coordinates of the structural mesh
        for body in model.bodies:
            body.initialize_struct_mesh(self.struct_X)

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def set_variables(self, scenario, bodies):
        """Set the design variables for the solver"""

        # Set the aerodynamic design variables
        index = 0
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_dvs[index] = var.value
                index += 1

        return

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        if scenario.steady:
            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "structural":
                    value = 0.0
                    for body in bodies:
                        struct_loads = body.get_struct_loads(scenario)
                        if struct_loads is not None:
                            value += np.dot(self.func_coefs1, struct_loads)
                        struct_flux = body.get_struct_heat_flux(scenario)
                        if struct_flux is not None:
                            value += np.dot(self.func_coefs2, struct_flux)
                    func.value = self.comm.allreduce(value)
        else:
            # Set the time index to the final time step
            time_index = scenario.steps

            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "structural":
                    value = 0.0
                    for body in bodies:
                        struct_loads = body.get_struct_loads(scenario, time_index)
                        if struct_loads is not None:
                            value += np.dot(self.func_coefs1, struct_loads)
                        struct_flux = body.get_struct_heat_flux(scenario, time_index)
                        if struct_flux is not None:
                            value += np.dot(self.func_coefs2, struct_flux)
                    func.value = self.comm.allreduce(value)

        return

    def eval_function_gradients(self, scenario, bodies):
        """
        Return the function gradients
        """

        # Set the derivatives of the functions for the given scenario
        for func in scenario.functions:
            for var in self.struct_variables:
                value = 0.0
                func.add_derivative(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):

        return

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            # Perform the "analysis"
            struct_loads = body.get_struct_loads(scenario)
            struct_disps = body.get_struct_disps(scenario)
            if struct_loads is not None:
                struct_disps[:] = np.dot(self.Jac1, struct_loads)
                struct_disps[:] += np.dot(self.b1, self.struct_X)
                struct_disps[:] += np.dot(self.c1, self.struct_dvs)

            # Perform the heat transfer "analysis"
            struct_flux = body.get_struct_heat_flux(scenario)
            struct_temps = body.get_struct_temps(scenario)
            if struct_flux is not None:
                struct_temps[:] = np.dot(self.Jac2, struct_flux)
                struct_temps[:] += np.dot(self.b2, self.struct_X)
                struct_temps[:] += np.dot(self.c2, self.struct_dvs)

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def set_states(self, scenario, bodies, step):
        pass

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        # We solve for all the adjoints simultaneously
        nfunctions = scenario.count_adjoint_functions()

        # if self.solver == "aerodynamic":
        #     # Solve for the flow adjoint
        #     for body in bodies:
        #         psi_F = body.get_aero_loads_adjoint(scenario)
        #         adjD_rhs = body.get
        #         adjD_rhs = - np.dot(self.Jac1.T, psi_F)

        # if self.solver == "structural":
        #     for body in bodies:

        # if self.solver == "aerodynamic":
        #     for body in bodies:
        #         # Perform the "analysis"
        #         # Note that the body class expects that you will write the new
        #         # aerodynamic loads into the aero_loads array.
        #         aero_disps = body.get_aero_disps(scenario)
        #         aero_loads = body.get_aero_loads(scenario)
        #         if aero_disps is not None:
        #             aero_loads[:] = np.dot(self.Jac1, aero_disps) + self.b1

        #         # Perform the heat transfer "analysis"
        #         aero_temps = body.get_aero_temps(scenario)
        #         aero_flux = body.get_aero_heat_flux(scenario)
        #         if aero_temps is not None:
        #             aero_flux[:] = np.dot(self.Jac2, aero_temps) + self.b2
        # else:  # The solver is a structural solver
        #     for body in bodies:
        #         # Perform the "analysis"
        #         struct_loads = body.get_struct_loads(scenario)
        #         struct_disps = body.get_struct_disps(scenario)
        #         if struct_loads is not None:
        #             struct_disps[:] = np.dot(self.Jac1, struct_loads) + self.b1

        #         # Perform the heat transfer "analysis"
        #         struct_flux = body.get_struct_heat_flux(scenario)
        #         struct_temps = body.get_struct_temps(scenario)
        #         if struct_flux is not None:
        #             struct_temps[:] = np.dot(self.Jac2, struct_flux) + self.b2

        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        pass

    def step_pre(self, scenario, bodies, step):
        return 0

    def step_post(self, scenario, bodies, step):
        return 0
