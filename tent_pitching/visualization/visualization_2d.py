import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import mpl_toolkits.mplot3d.art3d as art3d


def plot_2d_space_time_mesh(space_time_mesh):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for tent in space_time_mesh.tents:
        for v1 in tent.space_time_vertices:
            for v2 in tent.space_time_vertices:
                if v1 != v2 and space_time_mesh.check_space_time_vertices_adjacent(v1, v2):
                    if v1.space_vertex == v2.space_vertex:
                        line = art3d.Line3D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], [v1.coordinates[2], v2.coordinates[2]], linewidth=0.5, linestyle="--", color="green")
                    else:
                        line = art3d.Line3D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], [v1.coordinates[2], v2.coordinates[2]], linewidth=1, linestyle="-", color="blue")
                    ax.add_line(line)

    for vertex in space_time_mesh.space_time_vertices:
        ax.plot(*vertex.coordinates, marker='o', color='lightblue')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")

    plt.show()
