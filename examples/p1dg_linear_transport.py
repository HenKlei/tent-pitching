import os
import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.functions import SpaceTimeFunction
from tent_pitching.grids import create_uniform_grid
from tent_pitching.operators import GridOperator
from tent_pitching.utils.visualization import plot_space_time_function, write_space_time_grid
from tent_pitching.utils.error_computation import compute_l2_errors


GLOBAL_SPACE_GRID_SIZE = 1./10.
grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)
T_MAX = 1.
MU = 1.
EPS = 1.
FILEPATH_RESULTS = 'results_linear_transport_st_dg/'

if not os.path.exists(FILEPATH_RESULTS):
    os.makedirs(FILEPATH_RESULTS)


def characteristic_speed(x):
    return MU + EPS


space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=None)

write_space_time_grid(space_time_grid, FILEPATH_RESULTS + 'grid_linear_transport')


def linear_transport_flux(u):
    return np.array([MU * u, ])


def linear_transport_flux_derivative(u):
    return np.array([MU, ])


def u_0_function(x, jumps=True):
    if jumps:
        return 1. * (x <= 0.4)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 1. * (x < 0.0)


def inflow_boundary_values(x):
    return 1.


grid_operator = GridOperator(space_time_grid, linear_transport_flux,
                             linear_transport_flux_derivative,
                             u_0_function, inflow_boundary_values)

u = grid_operator.solve()

plot_solution = plot_space_time_function(u)
plot_solution.savefig(FILEPATH_RESULTS + f'u_mu_{str(MU).replace(".", "_")}_global.pdf')


def exact_solution(x):
    return u_0_function(x[0] - MU * x[1])


exact_solution_space_time_function = SpaceTimeFunction(space_time_grid)
exact_solution_space_time_function.interpolate(exact_solution)
plot_error = plot_space_time_function(u - exact_solution_space_time_function)
plot_error.savefig(FILEPATH_RESULTS + f'u_mu_{str(MU).replace(".", "_")}_global_error.pdf')

relative_error, absolute_error = compute_l2_errors(u, exact_solution)

with open('errors_linear_transport_st_dg_p1.txt', 'a') as file_obj:
    file_obj.write(f"{MU}\t{GLOBAL_SPACE_GRID_SIZE}\t{len(space_time_grid.tents)}\t"
                   f"{absolute_error}\t{relative_error}\n")
