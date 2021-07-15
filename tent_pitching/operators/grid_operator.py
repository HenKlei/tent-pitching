import numpy as np

from tent_pitching.functions.global_functions import SpaceTimeFunction
from tent_pitching.functions.local_functions import P1DGLocalFunction
from tent_pitching.utils.logger import getLogger


class GridOperator:
    def __init__(self, space_time_grid, flux, flux_derivative, u_0, inflow_boundary_values,
                 LocalFunctionType=P1DGLocalFunction):
        self.space_time_grid = space_time_grid
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.u_0 = u_0
        self.inflow_boundary_values = inflow_boundary_values
        self.LocalFunctionType = LocalFunctionType

    def space_time_flux(self, u):
        return np.concatenate((self.flux(u), u), axis=None)

    def space_time_flux_derivative(self, u):
        return np.concatenate((self.flux_derivative(u), 1.), axis=None)

    def solve(self):
        function = SpaceTimeFunction(self.space_time_grid)

        logger = getLogger('tent_pitching.GridOperator')

        with logger.block("Iteration over the tents of the space time grid ..."):
            for tent in self.space_time_grid.tents:
                with logger.block(f"Solving on {tent} ..."):
                    inflow_tents = tent.inflow_tents()
                    solution_on_inflow_tents = [function.get_function_on_tent(inflow_tent)
                                                for inflow_tent in inflow_tents]
                    local_solution = self.solve_local_problem(tent, solution_on_inflow_tents)
                    function.set_function_on_tent(tent, local_solution)

        return function

    def solve_local_problem(self, tent, solution_on_inflow_tents,
                            tol=1e-6, max_iter=None):
        assert tent in self.space_time_grid.tents
        # local_initial_value has to be list of LocalSpaceFunctions

        local_solution = self.LocalFunctionType(tent.element)
        num_dofs = local_solution.NUM_DOFS

        inflow_tents = tent.inflow_tents()

        integral_inflow = 0.
        for inflow_face in tent.inflow_faces():
            points, weights = inflow_face.quadrature()

            if inflow_face.outside:  # real inflow face
                inflow_tent = inflow_face.outside
                solution_on_inflow_tent = solution_on_inflow_tents[inflow_tents.index(inflow_tent)]
                for x_hat, w in zip(points, weights):
                    x = inflow_face.to_global(x_hat)
                    integral_inflow += (w * inflow_face.volume()
                                        * self.space_time_flux(solution_on_inflow_tent(x)).dot(
                                              inflow_face.outer_unit_normal()))
            else:  # boundary face - either space boundary or time boundary
                for x_hat, w in zip(points, weights):
                    x = inflow_face.to_global(x_hat)
                    if np.isclose(x[1], 0.):
                        f = self.space_time_flux(self.u_0(x[0]))
                    elif np.isclose(x[0], 0.):
                        f = self.space_time_flux(self.inflow_boundary_values(x[1]))
                    else:
                        raise ValueError
                    integral_inflow += (w * inflow_face.volume()
                                        * f.dot(inflow_face.outer_unit_normal()))

        S_t = 0.
        for outflow_face in tent.outflow_faces():
            points, weights = outflow_face.quadrature()
            for x_hat, w in zip(points, weights):
                x = outflow_face.to_global(x_hat)
                S_t += (w * outflow_face.volume() * outflow_face.outer_unit_normal()[-1])

        initial_value = - np.ones(num_dofs) * integral_inflow / S_t
        local_solution.set_values(initial_value)

        inflow_vector = np.zeros(num_dofs)
        for i in range(num_dofs):
            for inflow_face in tent.inflow_faces():
                i_th_unit_vector = np.zeros(num_dofs)
                i_th_unit_vector[i] = 1.
                phi_i = self.LocalFunctionType(tent.element, i_th_unit_vector)
                points, weights = inflow_face.quadrature()

                if inflow_face.outside:  # real inflow face
                    inflow_tent = inflow_face.outside
                    solution_on_inflow_tent = solution_on_inflow_tents[
                        inflow_tents.index(inflow_tent)]
                    for x_hat, w in zip(points, weights):
                        x = inflow_face.to_global(x_hat)
                        inflow_vector[i] += (phi_i(x) * w * inflow_face.volume()
                                             * self.space_time_flux(solution_on_inflow_tent(x)).dot(
                                                   inflow_face.outer_unit_normal()))
                else:  # boundary face - either space boundary or time boundary
                    for x_hat, w in zip(points, weights):
                        x = inflow_face.to_global(x_hat)
                        if np.isclose(x[1], 0.):
                            f = self.space_time_flux(self.u_0(x[0]))
                        elif np.isclose(x[0], 0.):
                            f = self.space_time_flux(self.inflow_boundary_values(x[1]))
                        else:
                            raise ValueError
                        inflow_vector[i] += (phi_i(x) * w * inflow_face.volume()
                                             * f.dot(inflow_face.outer_unit_normal()))

        res = self.compute_residuum(tent, local_solution, inflow_vector)

        logger = getLogger('tent_pitching.GridOperator')

        logger.info("Performing residual minimization ...")

        iteration = 0
        while np.linalg.norm(res) > tol and (max_iter is None or iteration < max_iter):
            mat = self.compute_matrix(tent, local_solution)
            update = np.linalg.solve(mat, -res)
            local_solution = local_solution + update
            res = self.compute_residuum(tent, local_solution, inflow_vector)
            iteration += 1

        logger.info("Finished Newton iterations with residual norm of "
                    f"{np.linalg.norm(res):.3e} ...")

        return local_solution

    def compute_residuum(self, tent, local_solution, inflow_vector):
        num_dofs = local_solution.NUM_DOFS
        residuum = np.zeros(num_dofs)

        for i in range(num_dofs):
            i_th_unit_vector = np.zeros(num_dofs)
            i_th_unit_vector[i] = 1.
            phi_i = self.LocalFunctionType(tent.element, i_th_unit_vector)
            for outflow_face in tent.outflow_faces():
                points, weights = outflow_face.quadrature()
                for x_hat, w in zip(points, weights):
                    x = outflow_face.to_global(x_hat)
                    residuum[i] += (phi_i(x) * w * outflow_face.volume()
                                    * self.space_time_flux(local_solution(x)).dot(
                                          outflow_face.outer_unit_normal()))

            points, weights = tent.element.quadrature()
            for x_hat, w in zip(points, weights):
                x = tent.element.to_global(x_hat)
                residuum[i] -= (phi_i.gradient(x).dot(self.space_time_flux(local_solution(x))) * w
                                * tent.element.volume())

        return (residuum + inflow_vector) / tent.element.volume()

    def compute_matrix(self, tent, local_solution):
        num_dofs = local_solution.NUM_DOFS
        mat = np.zeros((num_dofs, num_dofs))

        for i in range(num_dofs):
            i_th_unit_vector = np.zeros(num_dofs)
            i_th_unit_vector[i] = 1.
            phi_i = self.LocalFunctionType(tent.element, i_th_unit_vector)
            for j in range(num_dofs):
                j_th_unit_vector = np.zeros(num_dofs)
                j_th_unit_vector[j] = 1.
                phi_j = self.LocalFunctionType(tent.element, j_th_unit_vector)
                for outflow_face in tent.outflow_faces():
                    points, weights = outflow_face.quadrature()
                    for x_hat, w in zip(points, weights):
                        x = outflow_face.to_global(x_hat)
                        mat[i][j] += (phi_j(x) * phi_i(x) * w * outflow_face.volume()
                                      * self.space_time_flux_derivative(local_solution(x)).dot(
                                            outflow_face.outer_unit_normal()))

                points, weights = tent.element.quadrature()
                for x_hat, w in zip(points, weights):
                    x = tent.element.to_global(x_hat)
                    mat[i][j] -= (phi_j(x) * w * tent.element.volume()
                                  * phi_i.gradient(x).dot(
                                        self.space_time_flux_derivative(local_solution(x))))

        return mat / tent.element.volume()
