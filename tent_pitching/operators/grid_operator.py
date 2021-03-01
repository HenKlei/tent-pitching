from tent_pitching.functions import SpaceFunction, SpaceTimeFunction, LocalSpaceTimeFunction
from tent_pitching.discretizations import ExplicitEuler


class GridOperator:
    def __init__(self, space_time_grid, discretization, LocalSpaceFunctionType,
                 TimeStepperType=ExplicitEuler, local_space_grid_size=1e-1,
                 local_time_grid_size=1e-1):
        self.space_time_grid = space_time_grid

        assert 0.0 < local_space_grid_size <= 1.0
        assert 0.0 < local_time_grid_size <= 1.0

        self.local_space_grid_size = local_space_grid_size
        self.local_time_grid_size = local_time_grid_size

        self.LocalSpaceFunctionType = LocalSpaceFunctionType
        self.discretization = discretization
        assert self.discretization.LocalSpaceFunctionType == self.LocalSpaceFunctionType

        self.time_stepper = TimeStepperType(self.discretization, self.local_time_grid_size)

    def interpolate(self, u):
        u_interpolated = SpaceFunction(self.space_time_grid.space_grid,
                                       self.LocalSpaceFunctionType,
                                       u=u, local_space_grid_size=self.local_space_grid_size)
        return u_interpolated

    def solve(self, u_0):
        assert isinstance(u_0, SpaceFunction)

        function = SpaceTimeFunction(self.space_time_grid, self.LocalSpaceFunctionType,
                                     local_space_grid_size=self.local_space_grid_size,
                                     local_time_grid_size=self.local_time_grid_size)

        def transformation(u_hat, phi_1_prime, phi_2_dx):
            return phi_1_prime * (u_hat - self.discretization.flux(u_hat) * phi_2_dx)

        function.set_global_initial_value(u_0, transformation)

        print("\033[1mIterating over the tents of the space time grid...\033[0m")
        for tent in self.space_time_grid.tents:
            print(f"|   Solving on {tent}")
            local_initial_value = function.get_initial_value_on_tent(tent)
            local_solution = self.solve_local_problem(tent, local_initial_value)
            function.set_function_on_tent(tent, local_solution)

        return function

    def solve_local_problem(self, tent, local_initial_value):
        assert tent in self.space_time_grid.tents
        assert isinstance(local_initial_value, list)
        assert all(isinstance(function, self.LocalSpaceFunctionType)
                   for function in local_initial_value)

        local_solution = LocalSpaceTimeFunction(tent, self.LocalSpaceFunctionType,
                                                local_space_grid_size=self.local_space_grid_size,
                                                local_time_grid_size=self.local_time_grid_size)

        local_solution.set_initial_value(local_initial_value)
        for time in range(1, int(1. / self.local_time_grid_size) + 1):
            # List of LocalSpaceFunctionType members!
            old_solution = local_solution.get_value(time - 1)
            # List of LocalSpaceFunctionType members!
            update = self.time_stepper(tent, old_solution, time)
            local_solution.set_value(time, [f1 + self.local_time_grid_size * f2
                                            for f1, f2 in zip(old_solution, update)])

        return local_solution
