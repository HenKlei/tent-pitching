from tent_pitching.functions.local_functions import P1DGLocalFunction


class SpaceTimeFunction:
    """
    Function that is defined for each tent in the space time grid on its reference element.
    """
    def __init__(self, space_time_grid):
        self.space_time_grid = space_time_grid

        self.functions = [P1DGLocalFunction(tent.element) for tent in self.space_time_grid.tents]

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        assert x.shape == (self.space_time_grid.dim,)
        for i, tent in enumerate(self.space_time_grid.tents):
            if x in tent:
                return self.functions[i](x)
        raise ValueError

    def get_function_on_tent(self, tent):
        assert tent in self.space_time_grid.tents
        return self.functions[self.space_time_grid.tents.index(tent)]

    def set_function_on_tent(self, tent, u):
        assert tent in self.space_time_grid.tents
        self.functions[self.space_time_grid.tents.index(tent)] = u
