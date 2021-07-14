import numpy as np

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

    def __add__(self, u):
        assert isinstance(u, SpaceTimeFunction)
        assert self.space_time_grid == u.space_time_grid
        res = SpaceTimeFunction(self.space_time_grid)
        for i, (f1, f2) in enumerate(zip(self.functions, u.functions)):
            res.functions[i] = f1 + f2
        return res

    def __mul__(self, x):
        assert isinstance(x, int) or isinstance(x, float)
        res = SpaceTimeFunction(self.space_time_grid)
        for i, f in enumerate(self.functions):
            res.functions[i] = x * f
        return res

    __rmul__ = __mul__

    def __sub__(self, u):
        return self + (-1.) * u

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
        assert isinstance(u, P1DGLocalFunction)
        self.functions[self.space_time_grid.tents.index(tent)] = u

    def interpolate(self, u):
        for function in self.functions:
            function.interpolate(u)

    def two_norm(self):
        norm = 0.
        for function in self.functions:
            points, weights = function.element.quadrature()
            for x_hat, w in zip(points, weights):
                x = function.element.to_global(x_hat)
                norm += w * function.element.volume() * function(x)**2

        return np.sqrt(norm)
