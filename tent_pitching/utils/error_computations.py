import numpy as np

from tent_pitching.functions import SpaceTimeFunction


def compute_error(u, exact_solution):
    assert isinstance(u, SpaceTimeFunction)
    absolute_error = 0.
    norm_exact_solution = 0.
    for function in u.functions:
        points, weights = function.element.quadrature()
        for x_hat, w in zip(points, weights):
            x = function.element.to_global(x_hat)
            absolute_error += w * function.element.volume() * (function(x) - exact_solution(x))**2
            norm_exact_solution += w * function.element.volume() * exact_solution(x)**2

    norm_exact_solution = np.sqrt(norm_exact_solution)
    absolute_error = np.sqrt(absolute_error)
    relative_error = absolute_error / norm_exact_solution

    return relative_error, absolute_error
