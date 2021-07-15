import os
import numpy as np

from typer import Option, run

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.operators import GridOperator
from tent_pitching.utils.error_computations import compute_error
from tent_pitching.utils.visualization import (plot_space_time_function, write_space_time_grid,
                                               plot_space_time_function_difference)


T_MAX = 1.
EPS = 1.
FILEPATH_RESULTS = 'results_linear_transport_st_dg/'


def main(mu: float = Option(1., help='Parameter mu that determines the velocity'),
         n: int = Option(10, help='Number of elements in the space grid')):
    GLOBAL_SPACE_GRID_SIZE = 1. / n
    grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    def characteristic_speed(x):
        return mu + EPS

    def linear_transport_flux(u):
        return np.array([mu * u, ])

    def linear_transport_flux_derivative(u):
        return np.array([mu, ])

    def u_0_function(x, jumps=True):
        if jumps:
            return 1. * (x <= 0.4)
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 1. * (x < 0.0)

    def inflow_boundary_values(x):
        return 1.

    space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=None)

    write_space_time_grid(space_time_grid, FILEPATH_RESULTS +
                          f'grid_linear_transport_mu_{str(mu).replace(".", "_")}_n_{n}')

    grid_operator = GridOperator(space_time_grid, linear_transport_flux,
                                 linear_transport_flux_derivative,
                                 u_0_function, inflow_boundary_values)

    u = grid_operator.solve()

    plot_solution = plot_space_time_function(u)
    plot_solution.savefig(FILEPATH_RESULTS + f'u_mu_{str(mu).replace(".", "_")}_n_{n}_global.pdf')

    def exact_solution(x):
        return u_0_function(x[0] - mu * x[1])

    plot_solution = plot_space_time_function(exact_solution)
    plot_solution.savefig(FILEPATH_RESULTS
                          + f'exact_solution_mu_{str(mu).replace(".", "_")}_n_{n}_global.pdf')

    plot_error = plot_space_time_function_difference(u, exact_solution)
    plot_error.savefig(FILEPATH_RESULTS +
                       f'u_mu_{str(mu).replace(".", "_")}_n_{n}_global_error.pdf')

    relative_error, absolute_error = compute_error(u, exact_solution)

    with open(FILEPATH_RESULTS + 'errors_linear_transport_st_dg_p1.txt', 'a') as file_obj:
        file_obj.write(f"{mu}\t{n}\t{len(space_time_grid.tents)}\t"
                       f"{absolute_error}\t{relative_error}\n")


if __name__ == '__main__':
    run(main)
