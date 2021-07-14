import numpy as np
import matplotlib.pyplot as plt

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import create_uniform_grid
from tent_pitching.operators import GridOperator
from tent_pitching.utils.visualization import plot_space_time_function, write_space_time_grid


GLOBAL_SPACE_GRID_SIZE = 1./100.
grid = create_uniform_grid(GLOBAL_SPACE_GRID_SIZE)
T_MAX = 1.
MU = 1.
EPS = 1.


def characteristic_speed(x):
    return MU + EPS


space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=20000)

write_space_time_grid(space_time_grid, 'grid_burgers')


def burgers_flux(u):
    return np.array([0.5 * MU * u * u, ])


def burgers_flux_derivative(u):
    return np.array([MU * u, ])


def u_0_function(x, jumps=True):
    if jumps:
        return 1. * (x >= 0.4)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5)


def inflow_boundary_values(x):
    return 0.


grid_operator = GridOperator(space_time_grid, burgers_flux, burgers_flux_derivative,
                             u_0_function, inflow_boundary_values)

u = grid_operator.solve()

plot_space_time_function(u)
plt.show()
