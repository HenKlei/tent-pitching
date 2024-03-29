from tent_pitching.grids import SpaceTimeGrid
from tent_pitching.utils.logger import getLogger


def perform_tent_pitching(space_grid, t_max, characteristic_speed, n_max=1000):
    logger = getLogger('tent_pitching')

    space_time_grid = SpaceTimeGrid(space_grid, t_max, characteristic_speed)

    iteration = 0

    with logger.block("Creating spacetime grid via tent pitching ..."):
        while iteration < n_max:
            space_time_vertex = space_time_grid.advancing_front.get_feasible_vertex()
            if space_time_vertex is None:
                break

            logger.info(f"Pitching tent on {space_time_vertex} ...")

            space_time_grid.pitch_tent(space_time_vertex)

            iteration += 1

    logger.info(f"Finished tent pitching with {iteration} tents ...")
    logger.info("")

    if space_time_grid.advancing_front.get_feasible_vertex() is not None:
        assert iteration == n_max
        raise Exception(f"The maximum number of {n_max} tents is reached without"
                        "finishing the spacetime meshing process!")

    return space_time_grid
