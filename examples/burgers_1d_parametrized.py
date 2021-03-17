import numpy as np
import matplotlib.pyplot as plt
import pickle

from typer import Option, run

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid, create_uniform_grid
from tent_pitching.visualization import (plot_1d_space_time_grid, plot_space_function,
                                         plot_space_time_function, plot_on_reference_tent)
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, RungeKutta4


GLOBAL_SPACE_GRID_SIZE = 1./3.
T_MAX = 1.
MAX_SPEED = 6.

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2

TENT_NUMBER = 5


def main(MU: float = Option(1., help='Parameter mu that determines the velocity')):
    assert 0. < MU <= MAX_SPEED / 2.

    grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)


    def characteristic_speed(x):
        return MAX_SPEED


    space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000, log=True)

    plot_1d_space_time_grid(space_time_grid, title='Space time grid obtained via tent pitching')



    def burgers_flux(u):
        return 0.5 * MU * u**2


    def burgers_flux_derivative(u):
        return u * MU


    def inverse_transformation(u, phi_2, phi_2_dt, phi_2_dx):
        return 2 * u / (1 + np.sqrt(1 - 2 * u * phi_2_dx * MU))


    discretization = DiscontinuousGalerkin(burgers_flux, burgers_flux_derivative,
                                           inverse_transformation, LOCAL_SPACE_GRID_SIZE,
                                           LOCAL_TIME_GRID_SIZE)

    grid_operator = GridOperator(space_time_grid, discretization, DGFunction,
                                 TimeStepperType=RungeKutta4,
                                 local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                                 local_time_grid_size=LOCAL_TIME_GRID_SIZE)


    def u_0_function(x, jumps=False):
        if jumps:
            return 1. * (x <= 0.25) + 0.25 * (0.25 < x <= 0.5)
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)


    u_0 = grid_operator.interpolate(u_0_function)

    plot_space_function(u_0, title='Initial condition interpolated to DG space')

    u = grid_operator.solve(u_0)

    plot_space_time_function(u, inverse_transformation, title='Space time solution',
                             three_d=True, space_time_grid=space_time_grid)

    u_local = u.get_function_on_tent(space_time_grid.tents[TENT_NUMBER])
    plot_on_reference_tent(u_local, inverse_transformation,
                           title='Local solution on reference tent', three_d=True)

    plt.show()

    # Save computed solution on disk
    with open(f'u_Burgers_mu_{str(MU).replace(".", "_")}', 'wb') as file_obj:
        pickle.dump(u_local.get_function_values_as_matrix(inverse_transformation), file_obj)


if __name__ == '__main__':
    run(main)
