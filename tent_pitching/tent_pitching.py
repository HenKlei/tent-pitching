from tent_pitching.grids import SpaceTimeGrid


def perform_tent_pitching(space_grid, t_max, characteristic_speed, n_max=1000, log=True):
    space_time_grid = SpaceTimeGrid(space_grid, t_max, characteristic_speed)

    iteration = 0

    if log:
        print("\033[1mCreating spacetime grid via tent pitching...\033[0m")

    while True and iteration < n_max:
        space_time_vertex = space_time_grid.advancing_front.get_feasible_vertex()
        if space_time_vertex is None:
            break

        if log:
            print(f"|   Pitching tent on {space_time_vertex}")

        space_time_grid.pitch_tent(space_time_vertex)

        iteration += 1

    if log:
        print("\033[1mFinished tent pitching...\033[0m")

    if space_time_grid.advancing_front.get_feasible_vertex() is not None:
        assert iteration == n_max
        raise Exception(f"""The maximum number of {n_max} tents is reached without\
finishing the spacetime meshing process!""")

    return space_time_grid
