import numpy as np

from tent_pitching.functions import DGFunction
from .numerical_fluxes import LaxFriedrichsFlux


class DiscontinuousGalerkin:
    def __init__(self, flux, flux_derivative, inverse_transformation, local_space_grid_size,
                 local_time_grid_size, NumericalFlux=LaxFriedrichsFlux, eta_dirichlet=1.):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.inverse_transformation = inverse_transformation

        self.eta_dirichlet = eta_dirichlet

        self.LocalSpaceFunctionType = DGFunction

        self.local_space_grid_size = local_space_grid_size
        self.local_time_grid_size = local_time_grid_size

        lambda_ = self.local_time_grid_size / self.local_space_grid_size
        self.numerical_flux = NumericalFlux(flux, lambda_)

    def _get_mass_matrix(self, tent):
        num_dofs = len(tent.get_space_patch().get_elements()) * int(1. / self.local_space_grid_size)
        mass_matrix = self.local_space_grid_size * np.eye(num_dofs)
        return mass_matrix

    def _R(self, tent, local_solution, dirichlet_value_left, dirichlet_value_right, t_ref):
        vector = np.zeros(len(tent.get_space_patch().get_elements())
                          * int(1. / self.local_space_grid_size))
        pos = 0

        for j, function in enumerate(local_solution):
            function_value = function.get_values()

            for i, function_value_central in enumerate(function_value):
                x_ref = tent.get_space_patch().to_local(function.element.to_global(
                    self.local_space_grid_size * (0.5 + i)))

                val = 0.

                if pos == 0:
                    # Boundary values
                    val -= self.eta_dirichlet * 1. * dirichlet_value_left

                    function_value_left = function_value[i]
                    function_value_right = function_value[i+1]
                elif pos == len(vector) - 1:
                    # Boundary values
                    val -= self.eta_dirichlet * 1. * dirichlet_value_right

                    function_value_left = function_value[i-1]
                    function_value_right = function_value[i]
                else:
                    if i == 0:
                        # Does this work in every case???? -> Order of local_solutions????
                        function_value_left = local_solution[j-1].get_values()[-1]
                        function_value_right = function_value[i+1]
                    elif i == int(1. / self.local_space_grid_size) - 1:
                        # Does this work in every case???? -> Order of local_solutions????
                        function_value_right = local_solution[j+1].get_values()[0]
                        function_value_left = function_value[i-1]
                    else:
                        function_value_left = function_value[i-1]
                        function_value_right = function_value[i+1]

                # Using the values at the boundary here does not seem to work somehow,
                # although that should be the proper way to do this!
                phi = tent.get_space_transformation(x_ref)
                delta_phi = tent.get_top_front_value(phi) - tent.get_bottom_front_value(phi)
                val += ((self.numerical_flux(function_value_central, function_value_left, -1.)
                         + self.numerical_flux(function_value_central, function_value_right, 1.)
                         ) * delta_phi)

                vector[pos] = val
                pos += 1

        return vector

    def _get_rhs_vector(self, tent, local_solution, t_ref):
        transformed_solution = [self.LocalSpaceFunctionType(function.element,
                                                            self.local_space_grid_size)
                                for function in local_solution]
        for i, function in enumerate(transformed_solution):
            function.set_values(local_solution[i].get_values())
            function_values = function.get_values()
            val = np.zeros(len(function_values))
            for j, u_hat in enumerate(function_values):
                x_ref = tent.get_space_patch().to_local(
                    function.element.to_global((j + 0.5) * self.local_space_grid_size))
                phi_1 = tent.get_space_transformation(x_ref)
                phi_1_prime = tent.get_space_transformation_dx(x_ref)
                phi_2 = tent.get_time_transformation(x_ref, t_ref)
                phi_2_dt = tent.get_time_transformation_dt(x_ref, t_ref)
                phi_2_dx = tent.get_time_transformation_dx(x_ref, t_ref)
                val[j] = self.inverse_transformation(u_hat, phi_1, phi_1_prime,
                                                     phi_2, phi_2_dt, phi_2_dx)
            function.set_values(val)

        dirichlet_value_left = local_solution[0].get_values()[0]
        dirichlet_value_right = local_solution[-1].get_values()[-1]

        return self._R(tent, transformed_solution, dirichlet_value_left,
                       dirichlet_value_right, t_ref)

    def right_hand_side(self, tent, local_solution, t_ref):
        assert all(isinstance(sol, self.LocalSpaceFunctionType) for sol in local_solution)
        assert self.local_space_grid_size == local_solution[0].local_space_grid_size

        mass_matrix = self._get_mass_matrix(tent)

        rhs_vector = self._get_rhs_vector(tent, local_solution, t_ref)

        # Solve for numpy array containing all the dofs and distribute
        # them to the respective local element-wise functions afterwards!
        rhs = np.linalg.solve(mass_matrix, rhs_vector)

        # Fill functions in local_functions with the respective values from rhs!
        local_functions = [self.LocalSpaceFunctionType(element, self.local_space_grid_size)
                           for element in tent.get_space_patch().get_elements()]

        for i, function in enumerate(local_functions):
            function.set_values(rhs[i * int(1. / self.local_space_grid_size):
                                    (i + 1) * int(1. / self.local_space_grid_size)])

        return local_functions
