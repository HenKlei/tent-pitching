import numpy as np

from tent_pitching.geometry.quadrature import gauss_quadrature


def test_gauss_quadrature():
    # Quadrature of order 0
    points, weights = gauss_quadrature(0)

    def function_to_integrate(x):
        return 3.

    integral = sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
    assert np.abs(integral - 3.) < 1e-8

    # Quadrature of order 1
    points, weights = gauss_quadrature(1)

    def function_to_integrate(x):
        return 3. * x

    integral = sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
    assert np.abs(integral - 1.5) < 1e-8

    # Quadrature of order 2
    points, weights = gauss_quadrature(2)

    def function_to_integrate(x):
        return x**2 + 2.*x + 4.

    integral = sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
    assert np.abs(integral - 16./3.) < 1e-8

    # Quadrature of order 3
    points, weights = gauss_quadrature(3)

    def function_to_integrate(x):
        return x**3

    integral = sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
    assert np.abs(integral - 0.25) < 1e-8


test_gauss_quadrature()
