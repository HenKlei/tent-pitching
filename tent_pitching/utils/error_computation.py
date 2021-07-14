import numpy as np


def compute_l2_errors(u, exact_solution):
    absolute_error = 0.
    norm_exact_solution = 0.

    for function in u.functions:
        points, weights = function.element.quadrature()
        for x_hat, w in zip(points, weights):
            x = function.element.to_global(x_hat)
            absolute_error += w * function.element.volume() * (function(x) - exact_solution(x))**2
            norm_exact_solution += w * function.element.volume() * exact_solution(x)**2

    absolute_error = np.sqrt(absolute_error)
    norm_exact_solution = np.sqrt(norm_exact_solution)

    return absolute_error / norm_exact_solution, absolute_error
