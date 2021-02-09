import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tent_pitching.grids import SpaceTimeGrid
from tent_pitching.functions import SpaceFunction


def plot_1d_space_time_grid(space_time_grid):
    assert isinstance(space_time_grid, SpaceTimeGrid)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for tent in space_time_grid.tents:
        for v1 in tent.space_time_vertices:
            for v2 in tent.space_time_vertices:
                if v1 != v2 and space_time_grid.check_space_time_vertices_adjacent(v1, v2):
                    if v1.space_vertex == v2.space_vertex:
                        line = Line2D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], linewidth=0.5, linestyle='--', color='green')
                    else:
                        line = Line2D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], linewidth=1, linestyle='-', color='blue')
                    ax.add_line(line)

    for vertex in space_time_grid.space_time_vertices:
        ax.plot(*vertex.coordinates, marker='o', color='lightblue')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_aspect('equal')

    plt.show()


def plot_space_function(u):
    assert isinstance(u, SpaceFunction)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(*u.get_function_global())
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    #ax.set_aspect('equal')

    plt.show()
