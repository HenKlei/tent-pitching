import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid
from tent_pitching.visualization import plot_1d_space_time_grid, plot_space_function, plot_space_time_function
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin


vertex1 = Vertex(0., label="Vertex 0")
vertex2 = Vertex(1., label="Vertex 1")
element1 = Element(vertex1, vertex2, label="Element 0")
elements = [element1,]
grid = Grid(elements)
T_MAX = 1.
mu = 1.
characteristic_speed = lambda x: mu

space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000, log=True)

plot_1d_space_time_grid(space_time_grid)

local_space_grid_size = 1e-1
local_time_grid_size = 1e-1

grid_operator = GridOperator(space_time_grid, DGFunction, local_space_grid_size=local_space_grid_size, local_time_grid_size=local_time_grid_size)

def u_0_function(x):
    return 1. * (x<=0.25) + 0.
u_0 = grid_operator.interpolate(u_0_function)

plot_space_function(u_0)

def linear_transport_flux(u):
    return mu * u

def linear_transport_flux_derivative(u):
    return 0.

def inverse_transformation(u, phi_1, phi_1_prime, phi_2, phi_2_dt, phi_2_dx):
    return u / (phi_1_prime - phi_2_dx * mu)

eta_D = 1e-4

discretization = DiscontinuousGalerkin(linear_transport_flux, linear_transport_flux_derivative, inverse_transformation, local_space_grid_size, local_time_grid_size, eta_D=eta_D)

u = grid_operator.solve(u_0, discretization)

plot_space_time_function(u)

import matplotlib.pyplot as plt
plt.show()
