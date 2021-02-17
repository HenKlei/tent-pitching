import numpy as np

from tent_pitching.functions import DGFunction
from .quadrature import gauss_quadrature
from .numerical_fluxes import LaxFriedrichsFlux


class DiscontinuousGalerkin:
    def __init__(self, flux, flux_derivative, inverse_transformation, local_space_grid_size, local_time_grid_size, NumericalFlux=LaxFriedrichsFlux, eta_D=1.):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.inverse_transformation = inverse_transformation

        self.eta_D = eta_D

        self.LocalSpaceFunctionType = DGFunction

        self.local_space_grid_size = local_space_grid_size
        self.local_time_grid_size = local_time_grid_size

        lambda_ = self.local_time_grid_size / self.local_space_grid_size
        self.NumericalFlux = NumericalFlux(flux, lambda_)

    def _get_mass_matrix(self, tent):
        num_dofs = len(tent.get_space_patch().get_elements()) * int(1. / self.local_space_grid_size)
        mass_matrix = self.local_space_grid_size * np.eye(num_dofs)
        return mass_matrix

    def _R(self, tent, local_solution, u_D_left, u_D_right, t_ref):
        vector = np.zeros(len(tent.get_space_patch().get_elements()) * int(1. / self.local_space_grid_size))
        l = 0

        for j, function in enumerate(local_solution):
            function_value = function.get_values()

            for i in range(len(function_value)):
                x_ref_left = tent.get_space_patch().to_local(function.element.to_global(self.local_space_grid_size * i))
                x_ref_right = tent.get_space_patch().to_local(function.element.to_global(self.local_space_grid_size * (1. + i)))

                val = 0.

                if l == 0:
                    # Boundary values
                    val -= self.eta_D * 1. * u_D_left

                    function_value_left = function_value[i]
                    function_value_central = function_value[i]
                    function_value_right = function_value[i+1]
                elif l == len(vector) - 1:
                    # Boundary values
                    val -= self.eta_D * 1. * u_D_right

                    function_value_left = function_value[i-1]
                    function_value_central = function_value[i]
                    function_value_right = function_value[i]
                else:
                    if i == 0:
                        function_value_left = local_solution[j-1].get_values()[-1] # Does this work in every case???? -> Order of local_solutions????
                        function_value_central = function_value[i]
                        function_value_right = function_value[i+1]
                    elif i == int(1. / self.local_space_grid_size) - 1:
                        function_value_right = local_solution[j+1].get_values()[0] # Does this work in every case???? -> Order of local_solutions????
                        function_value_left = function_value[i-1]
                        function_value_central = function_value[i]
                    else:
                        function_value_left = function_value[i-1]
                        function_value_central = function_value[i]
                        function_value_right = function_value[i+1]

                lambda_ = .5
                phi_1 = tent.get_space_transformation(x_ref_left)
                phi_2 = tent.get_space_transformation(x_ref_right)

                def LaxFriedrichsFlux(u_1, u_2, n):
                    return (self.flux(u_1) + self.flux(u_2)) / 2. * n + lambda_ * (u_1 - u_2)

                val += (LaxFriedrichsFlux(function_value_central, function_value_left, -1.) + LaxFriedrichsFlux(function_value_central, function_value_right, 1.)) * (tent.get_top_front_value(phi_2) - tent.get_bottom_front_value(phi_2))

                vector[l] = val
                l += 1

        return vector

    def _get_rhs_vector(self, tent, local_solution, t_ref):
        transformed_solution = [self.LocalSpaceFunctionType(function.element, self.local_space_grid_size) for function in local_solution]
        for i, function in enumerate(transformed_solution):
            function.set_values(local_solution[i].get_values())
            function_values = function.get_values()
            val = np.zeros(len(function_values))
            for j in range(len(function_values)):
                x_ref = tent.get_space_patch().to_local(function.element.to_global((j + 0.5) * self.local_space_grid_size))
                phi_1 = tent.get_space_transformation(x_ref)
                phi_1_prime = tent.get_space_transformation_dx(x_ref)
                phi_2 = tent.get_time_transformation(x_ref, t_ref)
                phi_2_dt = tent.get_time_transformation_dt(x_ref, t_ref)
                phi_2_dx = tent.get_time_transformation_dx(x_ref, t_ref)
                u = function_values[j]
                val[j] = self.inverse_transformation(u, phi_1, phi_1_prime, phi_2, phi_2_dt, phi_2_dx)
            function.set_values(val)

        u_D_left = local_solution[0].get_values()[0]
        u_D_right = local_solution[-1].get_values()[-1]

        return self._R(tent, transformed_solution, u_D_left, u_D_right, t_ref)

    def right_hand_side(self, tent, local_solution, t_ref):
        assert all(isinstance(sol, self.LocalSpaceFunctionType) for sol in local_solution)
        assert self.local_space_grid_size == local_solution[0].local_space_grid_size

        mass_matrix = self._get_mass_matrix(tent)

        rhs_vector = self._get_rhs_vector(tent, local_solution, t_ref)

        # Solve for numpy array containing all the dofs and distribute them to the respective local element-wise functions afterwards!
        rhs = np.linalg.solve(mass_matrix, rhs_vector)

        # Fill functions in local_functions with the respective values from rhs!
        local_functions = [self.LocalSpaceFunctionType(element, self.local_space_grid_size) for element in tent.get_space_patch().get_elements()]

        for i, function in enumerate(local_functions):
            function.set_values(rhs[i * int(1. / self.local_space_grid_size):(i + 1) * int(1. / self.local_space_grid_size)])

        return local_functions
