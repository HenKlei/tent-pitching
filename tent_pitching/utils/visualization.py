import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tent_pitching.grids import SpaceTimeGrid
from tent_pitching.functions.global_functions import SpaceTimeFunction


def plot_space_time_grid(space_time_grid, title=''):
    assert isinstance(space_time_grid, SpaceTimeGrid)
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

    for tent in space_time_grid.tents:
        for vertex1 in tent.space_time_vertices:
            for vertex2 in tent.space_time_vertices:
                if (vertex1 != vertex2 and
                        space_time_grid.check_space_time_vertices_adjacent(vertex1, vertex2)):
                    if vertex1.space_vertex == vertex2.space_vertex:
                        line = Line2D([vertex1.coordinates[0], vertex2.coordinates[0]],
                                      [vertex1.coordinates[1], vertex2.coordinates[1]],
                                      linewidth=0.5, linestyle='--', color='green')
                    else:
                        line = Line2D([vertex1.coordinates[0], vertex2.coordinates[0]],
                                      [vertex1.coordinates[1], vertex2.coordinates[1]],
                                      linewidth=1, linestyle='-', color='blue')
                    axes.add_line(line)

    for vertex in space_time_grid.space_time_vertices:
        axes.plot(*vertex.coordinates, marker='o', color='lightblue')

    axes.set_xlabel('x')
    axes.set_ylabel('t')
    axes.set_aspect('equal')
    axes.set_title(title)

    return fig


def plot_space_function(u, title='', n=100):
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

    xs = []
    ys = []
    for x in np.linspace(0, 1, n):
        xs.append(x)
        ys.append(u(x))

    axes.scatter(xs, ys)

    axes.set_xlabel('x')
    axes.set_ylabel('u(x)')
    axes.set_aspect('equal')
    axes.set_title(title)

    return fig


def plot_space_time_function(u, title='', interval=1, three_d=False, space_time_grid=None,
                             nx=50, ny=50):
    assert isinstance(u, SpaceTimeFunction)
    assert space_time_grid is None or isinstance(space_time_grid, SpaceTimeGrid)

    fig = plt.figure()
    if three_d:
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        axes = fig.add_subplot(1, 1, 1)

    xs = np.linspace(0., 1., nx)
    ys = np.linspace(0., 1., ny)
    xv, yv = np.meshgrid(xs, ys)
    uv = np.zeros(xv.flatten().shape)
    for i, (x, y) in enumerate(zip(xv.flatten(), yv.flatten())):
        uv[i] = u(np.array([x, y]))

    min_val = np.min(uv)
    max_val = np.max(uv)

    if three_d:
        scatter = axes.scatter(xv, yv, uv, c=uv, vmin=min_val, vmax=max_val)
    else:
        scatter = axes.scatter(xv, yv, c=uv, vmin=min_val, vmax=max_val)

    fig.colorbar(scatter)

    if space_time_grid and three_d:
        import mpl_toolkits.mplot3d.art3d as art3d
        for tent in space_time_grid.tents:
            for vertex1 in tent.space_time_vertices:
                for vertex2 in tent.space_time_vertices:
                    if (vertex1 != vertex2 and
                            space_time_grid.check_space_time_vertices_adjacent(vertex1, vertex2)):
                        if vertex1.space_vertex == vertex2.space_vertex:
                            line = art3d.Line3D([vertex1.coordinates[0], vertex2.coordinates[0]],
                                                [vertex1.coordinates[1], vertex2.coordinates[1]],
                                                [0., 0.],
                                                linewidth=0.5, linestyle='--', color='green')
                        else:
                            line = art3d.Line3D([vertex1.coordinates[0], vertex2.coordinates[0]],
                                                [vertex1.coordinates[1], vertex2.coordinates[1]],
                                                [0., 0.],
                                                linewidth=1, linestyle='-', color='blue')
                        axes.add_line(line)

        for vertex in space_time_grid.space_time_vertices:
            axes.plot(*vertex.coordinates, marker='o', color='lightblue')

    axes.set_xlabel('x')
    axes.set_ylabel('t')
    if not three_d:
        axes.set_aspect('equal')
    axes.set_title(title)

    return fig


def write_space_time_grid(space_time_grid, filename):
    import meshio

    points = []
    triangles = []
    quads = []
    for tent in space_time_grid.tents:
        for vertex in tent.space_time_vertices:
            points.append(vertex.coordinates)
        len_points = len(points)
        if len(tent.space_time_vertices) == 3:
            triangles.append([len_points-3, len_points-2, len_points-1])
        elif len(tent.space_time_vertices) == 4:
            quads.append([len_points-4, len_points-3, len_points-2, len_points-1])

    cells = [('triangle', triangles), ('quad', quads)]

    meshio.write_points_cells(f'{filename}.vtu', points, cells)
