import numpy as np
import matplotlib.pyplot as plt

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.utils.visualization import (plot_space_time_grid, plot_space_function,
                                               plot_space_time_function)
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin, EngquistOsherFlux


grid = create_uniform_grid(1.)
T_MAX = 1.
MU = 1.


def characteristic_speed(x):
    return MU


space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)

plot_space_time_grid(space_time_grid, title='Spacetime mesh obtained via tent pitching')

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2


def linear_transport_flux(u):
    return MU * u


def linear_transport_flux_derivative(u):
    return MU


def linear_transport_positive_flux(u):
    return max(u, 0.)


def linear_transport_negative_flux(u):
    return min(u, 0.)


def inverse_transformation(u, phi_2, phi_2_dt, phi_2_dx):
    return u / (1. - phi_2_dx * MU)


lambda_ = LOCAL_TIME_GRID_SIZE / LOCAL_SPACE_GRID_SIZE
numerical_flux = EngquistOsherFlux(linear_transport_flux,
                                   flux_positive=linear_transport_positive_flux,
                                   flux_negative=linear_transport_negative_flux)

discretization = DiscontinuousGalerkin(numerical_flux, inverse_transformation,
                                       LOCAL_SPACE_GRID_SIZE)


def u_0_function(x, jumps=True):
    if jumps:
        return 1. * (x <= 0.25)
        return 1. * (x <= 0.2) + 0.5 * (0.2 < x <= 0.5)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)


grid_operator = GridOperator(space_time_grid, discretization, DGFunction, u_0_function,
                             local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                             local_time_grid_size=LOCAL_TIME_GRID_SIZE)


plot_space_function(grid_operator.u_0, title='Initial condition interpolated to DG space')

u = grid_operator.solve()

plot_space_time_function(u, inverse_transformation, title='Spacetime solution')

plt.show()
