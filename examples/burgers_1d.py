import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid
from tent_pitching.visualization import plot_1d_space_time_grid, plot_space_function, plot_space_time_function
from tent_pitching.operators import GridOperator
from tent_pitching.functions import DGFunction
from tent_pitching.discretizations import DiscontinuousGalerkin


vertex1 = Vertex(0., label="Vertex 1")
vertex2 = Vertex(0.25, label="Vertex 2")
vertex3 = Vertex(0.75, label="Vertex 3")
vertex4 = Vertex(1., label="Vertex 4")
element1 = Element(vertex1, vertex2, label="Element 1")
element2 = Element(vertex2, vertex3, label="Element 2")
element3 = Element(vertex3, vertex4, label="Element 2")
elements = [element1, element2, element3]
grid = Grid(elements)
T_MAX = 1.
characteristic_speed = lambda x: 1.0

space_time_grid = perform_tent_pitching(grid, T_MAX, characteristic_speed, n_max=1000, log=True)

plot_1d_space_time_grid(space_time_grid)

grid_operator = GridOperator(space_time_grid, DGFunction, local_space_grid_size=1e-1, local_time_grid_size=1e-1)
def u_0_function(x):
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * x)) * (0.0 <= x <= 0.5) + 0. * (x > 0.5)
u_0 = grid_operator.interpolate(u_0_function)

plot_space_function(u_0)

def burgers_flux(u):
    return 0.5 * u**2

def burgers_flux_derivative(u):
    return u

def inverse_transformation(u, phi_1, phi_1_prime, phi_2, phi_2_dt, phi_2_dx):
    return 2 * u / (phi_1_prime + np.sqrt(phi_1_prime**2 - 2 * u * phi_2_dx))

alpha = 1e-1
eta = 1e-1

discretization = DiscontinuousGalerkin(burgers_flux, burgers_flux_derivative, inverse_transformation, alpha=alpha, eta=eta)
u = grid_operator.solve(u_0, discretization)

plot_space_time_function(u)

import matplotlib.pyplot as plt
plt.show()
