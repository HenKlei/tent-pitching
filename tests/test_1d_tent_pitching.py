from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid


def test_1d_tent_pitching():
    vertex0 = Vertex(0., label="Vertex 0")
    vertex1 = Vertex(0.25, label="Vertex 1")
    vertex2 = Vertex(0.75, label="Vertex 2")
    vertex3 = Vertex(1., label="Vertex 3")
    element0 = Element(vertex0, vertex1, label="Element 0")
    element1 = Element(vertex1, vertex2, label="Element 1")
    element2 = Element(vertex2, vertex3, label="Element 2")
    elements = [element0, element1, element2]
    grid = Grid(elements)
    t_max = 1.

    def characteristic_speed(_):
        return 1.

    space_time_grid = perform_tent_pitching(grid, t_max, characteristic_speed, n_max=1000)

    for vertex in grid.get_vertices():
        print(f'{vertex} is boundary: {vertex.is_boundary_vertex()}')

    i = 1
    pos = .75
    print(space_time_grid.tents[i].get_bottom_front_value(pos))
    print(space_time_grid.tents[i].get_bottom_front_derivative(pos))
    print(space_time_grid.tents[i].get_top_front_value(pos))
    print(space_time_grid.tents[i].get_top_front_derivative(pos))
