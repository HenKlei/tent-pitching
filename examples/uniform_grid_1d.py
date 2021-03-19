import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from typer import Option, run

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.visualization import (plot_1d_space_time_grid, plot_space_function,
                                         plot_space_time_function, plot_on_reference_tent)
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, RungeKutta4


GLOBAL_SPACE_GRID_SIZE = 1./3.
T_MAX = 1.
MAX_SPEED = 5.

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2

TENT_NUMBER = 5

FILEPATH_RESULTS = 'results_linear_transport/'


def main(MU: float = Option(1., help='Parameter mu that determines the velocity.')):
    assert 0. < MU <= MAX_SPEED / 2.

    grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)

    def characteristic_speed(x):
        return MAX_SPEED

    space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)

    plot_grid = plot_1d_space_time_grid(space_time_grid, title='Spacetime mesh obtained via tent pitching')
    plot_grid.savefig(FILEPATH_RESULTS + 'grid.pdf')

    def linear_transport_flux(u):
        return MU * u

    def linear_transport_flux_derivative(u):
        return MU

    def inverse_transformation(u, phi_2, phi_2_dt, phi_2_dx):
        return u / (1. - phi_2_dx * MU)

    discretization = DiscontinuousGalerkin(linear_transport_flux, linear_transport_flux_derivative,
                                           inverse_transformation, LOCAL_SPACE_GRID_SIZE,
                                           LOCAL_TIME_GRID_SIZE)

    grid_operator = GridOperator(space_time_grid, discretization, DGFunction,
                                 TimeStepperType=RungeKutta4,
                                 local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                                 local_time_grid_size=LOCAL_TIME_GRID_SIZE)

    def u_0_function(x, jump=True):
        if jump:
            return 1. * (x <= 0.25)
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)

    u_0 = grid_operator.interpolate(u_0_function)

    plot_space_function(u_0, title='Initial condition interpolated to DG space')

    u = grid_operator.solve(u_0)

    plot_space_time_function(u, inverse_transformation, title=r"Spacetime solution for $\mu=$" + str(MU),
                             three_d=True, space_time_grid=space_time_grid)

    u_plot = plot_space_time_function(u, inverse_transformation, title=r"Spacetime solution for $\mu=$" + str(MU),
                                      space_time_grid=space_time_grid)
    u_plot.savefig(FILEPATH_RESULTS + f'u_mu_{str(MU).replace(".", "_")}_global.pdf')

    u_local = u.get_function_on_tent(space_time_grid.tents[TENT_NUMBER])
    plot_on_reference_tent(u_local, inverse_transformation,
                           title=r"Local solution on reference tent for $\mu=$" + str(MU), three_d=True)

    plt.show()

    # Compute L2-error
    def exact_solution(x, t):
        return u_0_function(x - MU * t)

    u_values = u.get_function_values(inverse_transformation)
    u_exact = []
    error = 0.
    c = 0
    for x_val, t_val, z_val in zip(*u_values):
        tmp = []
        for xs, ts, zs in zip(x_val, t_val, z_val):
            tmp_2 = []
            for x, t, z in zip(xs, ts, zs):
                tmp_2.append(exact_solution(x, t))
                error += (exact_solution(x, t) - z)**2
                c += 1
            tmp.append(tmp_2)
        u_exact.append(tmp)

    print(f"L2-error: {np.sqrt(error / c)}")

    with open('errors.txt', 'a') as file_obj:
        file_obj.write(f"{MU}\t{LOCAL_SPACE_GRID_SIZE}\t{np.sqrt(error / c)}\n")

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    # Save computed solution on disk
    with open(FILEPATH_RESULTS + f'u_linear_transport_mu_{str(MU).replace(".", "_")}', 'wb') as file_obj:
        pickle.dump(u_local.get_function_values_as_matrix(inverse_transformation), file_obj)


if __name__ == '__main__':
    run(main)
