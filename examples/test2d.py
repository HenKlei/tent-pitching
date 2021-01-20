import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Edge, Element, Grid
from tent_pitching.visualization import plot_2d_space_time_mesh


vertex1 = Vertex(np.array([0., 0.]), label="Vertex 1")
vertex2 = Vertex(np.array([1., 0.]), label="Vertex 2")
vertex3 = Vertex(np.array([1., 1.]), label="Vertex 3")
vertex4 = Vertex(np.array([0., 1.]), label="Vertex 4")
edge1 = Edge(vertex1, vertex2, label="Edge 1")
edge2 = Edge(vertex2, vertex3, label="Edge 2")
edge3 = Edge(vertex3, vertex4, label="Edge 3")
edge4 = Edge(vertex4, vertex1, label="Edge 4")
edge5 = Edge(vertex1, vertex3, label="Edge 5")
element1 = Element([edge3, edge4, edge5], label="Element 1")
element2 = Element([edge1, edge2, edge5], label="Element 2")
elements = [element1, element2]
grid = Grid(elements)
T_MAX = 1.
characteristic_speed = lambda x: 1.0

space_time_mesh = perform_tent_pitching(grid, T_MAX, characteristic_speed)

plot_2d_space_time_mesh(space_time_mesh)
