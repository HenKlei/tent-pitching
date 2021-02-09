import numpy as np

from tent_pitching.functions import SpaceTimeFunction, SpaceFunction


class GridOperator:
    def __init__(self, space_time_grid, local_space_grid_size=1e-1, local_time_grid_size=1e-2):
        self.space_time_grid = space_time_grid

        assert 0.0 < local_space_grid_size <= 1.0
        assert 0.0 < local_time_grid_size <= 1.0

        self.local_space_grid_size = local_space_grid_size
        self.local_time_grid_size = local_time_grid_size

    def interpolate(self, u):
        u_interpolated = SpaceFunction(self.space_time_grid.space_grid, u, local_space_grid_size=self.local_space_grid_size)
        return u_interpolated

    def solve(self, u_0, problem):
        assert isinstance(u_0, SpaceFunction)

        u = SpaceTimeFunction(self.space_time_grid, local_space_grid_size=self.local_space_grid_size, local_time_grid_size=self.local_time_grid_size)

        u.set_global_initial_value(u_0)

        for tent in self.space_time_grid.tents:
            local_initial_value = u.get_initial_value_on_tent(tent)
            local_solution = self.solve_local_problem(tent, local_initial_value, problem)
            u.set_values_in_tent(tent, local_solution) # !!!!! We have to set the initial values in the next tents above !!!!!

        return u

    def solve_local_problem(self, tent, local_initial_value, problem):
        assert tent in self.space_time_grid.tents

        local_solution = np.zeros((int(1. / self.local_time_grid_size) + 1, (int(1. / self.local_space_grid_size) + 1) * len(tent.get_space_patch().get_elements())))

        local_solution[0] = local_initial_value
        for n in range(1, local_solution.shape[0] + 1):
            local_solution[n] = local_solution[n-1] + self.local_time_grid_size * problem.right_hand_side(tent, local_solution[n-1])

        return local_solution
