import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid
from tent_pitching.visualization import plot_1d_space_time_mesh


vertex1 = Vertex(np.array([0.]), label="Vertex 1")
vertex2 = Vertex(np.array([0.25]), label="Vertex 2")
vertex3 = Vertex(np.array([0.75]), label="Vertex 3")
vertex4 = Vertex(np.array([1.]), label="Vertex 4")
element1 = Element(vertex1, vertex2, label="Element 1")
element2 = Element(vertex2, vertex3, label="Element 2")
element3 = Element(vertex3, vertex4, label="Element 2")
elements = [element1, element2, element3]
grid = Grid(elements)
T_MAX = 1.
characteristic_speed = lambda x: 1.0

space_time_mesh = perform_tent_pitching(grid, T_MAX, characteristic_speed)

plot_1d_space_time_mesh(space_time_mesh)
