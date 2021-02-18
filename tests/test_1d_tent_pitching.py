from tent_pitching import perform_tent_pitching
from tent_pitching.grids import Vertex, Element, Grid
from tent_pitching.visualization import plot_1d_space_time_grid


def test_1d_tent_pitching():
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

    for vertex in grid.get_vertices():
        print(f'{vertex} is boundary: {vertex.is_boundary_vertex()}')

    i = 1
    X_REF = .75
    print(space_time_grid.tents[i]
          .get_bottom_front_value(space_time_grid.tents[i].get_space_patch().to_global(X_REF)))
    print(space_time_grid.tents[i]
          .get_bottom_front_derivative(space_time_grid.tents[i].get_space_patch().to_global(X_REF)))
    print(space_time_grid.tents[i]
          .get_top_front_value(space_time_grid.tents[i].get_space_patch().to_global(X_REF)))
    print(space_time_grid.tents[i]
          .get_top_front_derivative(space_time_grid.tents[i].get_space_patch().to_global(X_REF)))
