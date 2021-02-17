import numpy as np

from tent_pitching.functions import SpaceFunction, SpaceTimeFunction, LocalSpaceTimeFunction


class GridOperator:
    def __init__(self, space_time_grid, LocalSpaceFunctionType, local_space_grid_size=1e-1, local_time_grid_size=1e-1):
        self.space_time_grid = space_time_grid

        assert 0.0 < local_space_grid_size <= 1.0
        assert 0.0 < local_time_grid_size <= 1.0

        self.local_space_grid_size = local_space_grid_size
        self.local_time_grid_size = local_time_grid_size

        self.LocalSpaceFunctionType = LocalSpaceFunctionType

    def interpolate(self, u):
        u_interpolated = SpaceFunction(self.space_time_grid.space_grid, self.LocalSpaceFunctionType, u=u, local_space_grid_size=self.local_space_grid_size)
        return u_interpolated

    def solve(self, u_0, discretization):
        assert isinstance(u_0, SpaceFunction)
        assert discretization.LocalSpaceFunctionType == self.LocalSpaceFunctionType

        u = SpaceTimeFunction(self.space_time_grid, self.LocalSpaceFunctionType, local_space_grid_size=self.local_space_grid_size, local_time_grid_size=self.local_time_grid_size)

        u.set_global_initial_value(u_0)

        print("\033[1mIteration over the tents of the space time grid...\033[0m")
        for tent in self.space_time_grid.tents: # Only the first tent for testing purposes!!!!
            print(f"|   Solving on {tent}")
            local_initial_value = u.get_initial_value_on_tent(tent)
            local_solution = self.solve_local_problem(tent, local_initial_value, discretization)
            u.set_function_on_tent(tent, local_solution)

        return u

    def solve_local_problem(self, tent, local_initial_value, discretization):
        assert tent in self.space_time_grid.tents
        # local_initial_value has to be list of LocalSpaceFunctions

        local_solution = LocalSpaceTimeFunction(tent, self.LocalSpaceFunctionType, local_space_grid_size=self.local_space_grid_size, local_time_grid_size=self.local_time_grid_size)

        local_solution.set_initial_value(local_initial_value)
        for n in range(1, int(1. / self.local_time_grid_size) + 1):
            old_solution = local_solution.get_value(n - 1) # List of LocalSpaceFunctionType members!
            update = discretization.right_hand_side(tent, old_solution, n * self.local_time_grid_size) # List of LocalSpaceFunctionType members!
            local_solution.set_value(n, [f1 - self.local_time_grid_size * f2 for f1, f2 in zip(old_solution, update)])

        return local_solution
