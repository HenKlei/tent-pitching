from tent_pitching.grids import SpaceTimeMesh


def perform_tent_pitching(space_grid, t_max, characteristic_speed, n_max=1000, log=True):
    space_time_mesh = SpaceTimeMesh(space_grid, t_max, characteristic_speed)

    n = 0

    if log:
        print("Order of tent pitching locations:")

    while True and n < n_max:
        space_time_vertex = space_time_mesh.advancing_front.get_feasible_vertex()
        if space_time_vertex is None:
            break

        if log:
            print(space_time_vertex)

        space_time_mesh.pitch_tent(space_time_vertex)

        n += 1

    if log:
        print("Finished tent pitching...")

    if space_time_mesh.advancing_front.get_feasible_vertex() is not None:
        assert n == n_max
        raise Exception(f"The maximum number of {n_max} tents is reached without finishing the spacetime meshing process!")

    return space_time_mesh
