import numpy as np
import matplotlib.pyplot as plt

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid
from tent_pitching.utils.visualization import (plot_space_time_grid, plot_space_function,
                                               plot_space_time_function)
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin


vertex0 = Vertex(0., label="Vertex 0")
vertex1 = Vertex(0.25, label="Vertex 1")
vertex2 = Vertex(0.75, label="Vertex 2")
vertex3 = Vertex(1., label="Vertex 3")
element0 = Element(vertex0, vertex1, label="Element 0")
element1 = Element(vertex1, vertex2, label="Element 1")
element2 = Element(vertex2, vertex3, label="Element 2")
elements = [element0, element1, element2, ]
grid = Grid(elements)
T_MAX = 1.
MU = 1.
EPS = 2e-0


def characteristic_speed(x):
    return MU + EPS


space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000)

plot_space_time_grid(space_time_grid, title='Spacetime mesh obtained via tent pitching')

LOCAL_SPACE_GRID_SIZE = 1e-2
LOCAL_TIME_GRID_SIZE = 1e-2


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
                             local_space_grid_size=LOCAL_SPACE_GRID_SIZE,
                             local_time_grid_size=LOCAL_TIME_GRID_SIZE)


def u_0_function(x, jumps=True):
    if jumps:
        return 1. * (x <= 0.2) + 0.5 * (0.2 < x <= 0.5)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)


u_0 = grid_operator.interpolate(u_0_function)

plot_space_function(u_0, title='Initial condition interpolated to DG space')

u = grid_operator.solve(u_0)

plot_space_time_function(u, inverse_transformation, title='Spacetime solution')

plt.show()
