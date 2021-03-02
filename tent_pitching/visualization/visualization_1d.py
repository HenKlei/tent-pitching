import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tent_pitching.grids import SpaceTimeGrid
from tent_pitching.functions import SpaceFunction, SpaceTimeFunction, LocalSpaceTimeFunction


def plot_1d_space_time_grid(space_time_grid, title=''):
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


def plot_space_function(u, title=''):
    assert isinstance(u, SpaceFunction)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)

    for values in u.get_function_values():
        axes.plot(*values)

    axes.set_xlabel('x')
    axes.set_ylabel('u(x)')
    axes.set_aspect('equal')
    axes.set_title(title)

    return fig


def plot_space_time_function(u, transformation, title='', three_d=False, space_time_grid=None):
    assert isinstance(u, SpaceTimeFunction)
    assert space_time_grid is None or isinstance(space_time_grid, SpaceTimeGrid)

    fig = plt.figure()
    if three_d:
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        axes = fig.add_subplot(1, 1, 1)

    function_values = u.get_function_values(transformation)

    max_val = np.max(np.max(np.array(function_values, dtype=list)))
    min_val = np.min(np.min(np.array(function_values, dtype=list)))

    for x_val, y_val, z_val in zip(*function_values):
        if three_d:
            scatter = axes.scatter(x_val, y_val, z_val, c=z_val, vmin=min_val, vmax=max_val)
        else:
            scatter = axes.scatter(x_val, y_val, c=z_val, vmin=min_val, vmax=max_val)

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


def plot_on_reference_tent(u_local, transformation, title='', three_d=False):
    assert isinstance(u_local, LocalSpaceTimeFunction)

    fig = plt.figure()
    if three_d:
        axes = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        axes = fig.add_subplot(1, 1, 1)

    num_elements = len(u_local.tent.get_space_patch().get_elements())

    function_values = u_local.get_function_values(transformation)

    max_val = np.max(np.max(np.array(function_values, dtype=list)))
    min_val = np.min(np.min(np.array(function_values, dtype=list)))

    for i, element_function in enumerate(u_local.function):
        z_val = [element_time_function.get_values() for element_time_function in element_function]
        x_val = np.linspace(0.5 * element_function[0].local_space_grid_size, 1 - 0.5 * element_function[0].local_space_grid_size, len(z_val[0]))
        y_val = [[y,] * len(z_val[0]) for y in np.linspace(0, 1, len(z_val))]
        for y, z in zip(y_val, z_val):
            z_transformed = []
            x_transformed = []
            for x_ref, t_ref, val in zip(x_val, y, z):
                x = element_function[0].element.vertex_left.coordinate + x_ref * (element_function[0].element.vertex_right.coordinate - element_function[0].element.vertex_left.coordinate)
                phi_2 = u_local.tent.get_time_transformation(x, t_ref)
                phi_2_dt = u_local.tent.get_time_transformation_dt(x, t_ref)
                phi_2_dx = u_local.tent.get_time_transformation_dx(x, t_ref)
                z_transformed.append(transformation(val, phi_2, phi_2_dt, phi_2_dx))
                x_transformed.append((x_ref + i) / num_elements)
            if three_d:
                scatter = axes.scatter(x_transformed, y, z_transformed, c=z_transformed,
                                       vmin=min_val, vmax=max_val)
            else:
                scatter = axes.scatter(x_transformed, y, c=z_transformed,
                                       vmin=min_val, vmax=max_val)

    fig.colorbar(scatter)

    axes.set_xlabel('x')
    axes.set_ylabel('t')
    if not three_d:
        axes.set_aspect('equal')
    axes.set_title(title)

    return fig
