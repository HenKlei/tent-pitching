import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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

TENT_NUMBERS = [5, 8, 9, 10]

FILEPATH_RESULTS = 'results_Burgers/'


def main(MU: float = Option(1., help='Parameter mu that determines the velocity')):
    assert 0. < MU <= MAX_SPEED / 2.

    if not os.path.exists(FILEPATH_RESULTS):
        os.makedirs(FILEPATH_RESULTS)

    if not os.path.exists(FILEPATH_RESULTS + 'images/'):
          os.makedirs(FILEPATH_RESULTS + 'images/')

    grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)

    def characteristic_speed(x):
        return MAX_SPEED

    space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)
    TENT_NUMBERS = range(0, len(space_time_grid.tents))

    plot_spacetime_grid = plot_1d_space_time_grid(space_time_grid, title='Spacetime mesh obtained via tent pitching')
    plot_spacetime_grid.savefig(FILEPATH_RESULTS + 'images/spacetime_mesh_Burgers.pdf')
    plt.close(plot_spacetime_grid)

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

    def u_0_function(x, jumps=True):
        if jumps:
            return 1. * (x <= 0.25) + 0.25 * (0.25 < x <= 0.5)
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)

    u_0 = grid_operator.interpolate(u_0_function)

    plot_space_function(u_0, title='Initial condition interpolated to DG space')

    u = grid_operator.solve(u_0)

    u_plot_3d = plot_space_time_function(u, inverse_transformation, title=r'Spacetime solution for $\mu=$' + str(MU), interval=2,
                             three_d=True, space_time_grid=space_time_grid)
    u_plot_3d.savefig(FILEPATH_RESULTS + f'images/u_mu_{str(MU).replace(".", "_")}_global_3d.pdf')
    plt.close(u_plot_3d)

    u_plot = plot_space_time_function(u, inverse_transformation, title=r'Spacetime solution for $\mu=$' + str(MU), interval=5,
                                      three_d=False, space_time_grid=space_time_grid)
    u_plot.savefig(FILEPATH_RESULTS + f'images/u_mu_{str(MU).replace(".", "_")}_global.pdf')
    plt.close(u_plot)

    for number in TENT_NUMBERS:
        u_local = u.get_function_on_tent(space_time_grid.tents[number])
        u_plot_local_3d = plot_on_reference_tent(u_local, inverse_transformation,
                               title='Local solution on tent ' + str(number) + r' (mapped to reference tent) for $\mu=$' + str(MU) + ' ', interval=2, three_d=True)
        u_plot_local_3d.savefig(FILEPATH_RESULTS + f'images/u_mu_{str(MU).replace(".", "_")}_local_tent_{number}_3d.pdf')
        plt.close(u_plot_local_3d)

        u_plot_local = plot_on_reference_tent(u_local, inverse_transformation,
                               title='Local solution on tent ' + str(number) + r' (mapped to reference tent) for $\mu=$' + str(MU) + ' ', interval=2, three_d=False)
        u_plot_local.savefig(FILEPATH_RESULTS + f'images/u_mu_{str(MU).replace(".", "_")}_local_tent_{number}.pdf')
        plt.close(u_plot_local)

        # Save computed solution on disk
        with open(FILEPATH_RESULTS + f'u_Burgers_mu_{str(MU).replace(".", "_")}_tent_{number}', 'wb') as file_obj:
            pickle.dump(u_local.get_function_values_as_matrix(inverse_transformation), file_obj)

#    plt.show()


if __name__ == '__main__':
    run(main)
