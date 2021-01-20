import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_1d_space_time_mesh(space_time_mesh):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    for tent in space_time_mesh.tents:
        for v1 in tent.space_time_vertices:
            for v2 in tent.space_time_vertices:
                if v1 != v2 and space_time_mesh.check_space_time_vertices_adjacent(v1, v2):
                    if v1.space_vertex == v2.space_vertex:
                        line = Line2D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], linewidth=0.5, linestyle="--", color="green")
                    else:
                        line = Line2D([v1.coordinates[0], v2.coordinates[0]], [v1.coordinates[1], v2.coordinates[1]], linewidth=1, linestyle="-", color="blue")
                    ax.add_line(line)

    for vertex in space_time_mesh.space_time_vertices:
        ax.plot(*vertex.coordinates, marker='o', color='lightblue')

    plt.show()
