from tent_pitching.grid import SpaceTimeMesh


def perform_tent_pitching(space_grid, t_max, characteristic_speed, n_max=1000, log=True):
    space_time_mesh = SpaceTimeMesh(space_grid, t_max, characteristic_speed)

    n = 0

    while True and n < n_max:
        space_time_vertex = space_time_mesh.advancing_front.get_feasible_vertex()
        if space_time_vertex is None:
            break

        if log:
            print(space_time_vertex)

        space_time_mesh.pitch_tent(space_time_vertex)

        n += 1

    return space_time_mesh
