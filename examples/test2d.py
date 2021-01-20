import numpy as np

from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Edge, Element, Grid
from tent_pitching.visualization import plot_2d_space_time_mesh


vertex1 = Vertex(np.array([0., 0.]), label="Vertex 1")
vertex2 = Vertex(np.array([.5, 0.]), label="Vertex 2")
vertex3 = Vertex(np.array([1., 0.]), label="Vertex 3")
vertex4 = Vertex(np.array([0., .5]), label="Vertex 4")
vertex5 = Vertex(np.array([.5, .5]), label="Vertex 5")
vertex6 = Vertex(np.array([1., .5]), label="Vertex 6")
vertex7 = Vertex(np.array([0., 1.]), label="Vertex 7")
vertex8 = Vertex(np.array([.5, 1.]), label="Vertex 8")
vertex9 = Vertex(np.array([1., 1.]), label="Vertex 9")
edge1 = Edge(vertex1, vertex2, label="Edge 1")
edge2 = Edge(vertex2, vertex3, label="Edge 2")
edge3 = Edge(vertex1, vertex4, label="Edge 3")
edge4 = Edge(vertex1, vertex5, label="Edge 4")
edge5 = Edge(vertex2, vertex5, label="Edge 5")
edge6 = Edge(vertex3, vertex5, label="Edge 6")
edge7 = Edge(vertex3, vertex6, label="Edge 7")
edge8 = Edge(vertex4, vertex5, label="Edge 8")
edge9 = Edge(vertex5, vertex6, label="Edge 9")
edge10 = Edge(vertex4, vertex7, label="Edge 10")
edge11 = Edge(vertex7, vertex5, label="Edge 11")
edge12 = Edge(vertex8, vertex5, label="Edge 12")
edge13 = Edge(vertex9, vertex5, label="Edge 13")
edge14 = Edge(vertex6, vertex9, label="Edge 14")
edge15 = Edge(vertex7, vertex8, label="Edge 15")
edge16 = Edge(vertex8, vertex9, label="Edge 16")
element1 = Element([edge1, edge4, edge5], label="Element 1")
element2 = Element([edge2, edge5, edge6], label="Element 2")
element3 = Element([edge3, edge4, edge8], label="Element 3")
element4 = Element([edge6, edge7, edge9], label="Element 4")
element5 = Element([edge8, edge10, edge11], label="Element 5")
element6 = Element([edge9, edge13, edge14], label="Element 6")
element7 = Element([edge11, edge12, edge15], label="Element 7")
element8 = Element([edge12, edge13, edge16], label="Element 8")
elements = [element1, element2, element3, element4, element5, element6, element7, element8]
grid = Grid(elements)
T_MAX = 1.
characteristic_speed = lambda x: 1.0

space_time_mesh = perform_tent_pitching(grid, T_MAX, characteristic_speed)

plot_2d_space_time_mesh(space_time_mesh, tents_to_mark=[space_time_mesh.tents[0].get_space_patch_elements(),])
