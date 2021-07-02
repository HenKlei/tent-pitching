import numpy as np

from tent_pitching.geometry.entities import Triangle


def test_triangle_mappings():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]))
    t = Triangle(vertices)
    assert np.linalg.norm(t.to_global(np.array([0.25, 0.25])) - np.array([1., 1.5])) < 1e-8
    assert np.linalg.norm(t.to_local(np.array([1., 1.5])) - np.array([0.25, 0.25])) < 1e-8
    assert np.abs(t.volume - 1.) < 1e-8


def test_triangle_quadrature():
    vertices = (np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]))
    t = Triangle(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = t.quadrature(0)

    for b in np.random.rand(N):
        def func_to_integrate(x):
            return b

        exact_result = b / 2.
        integral = t.volume * sum([func_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.linalg.norm(exact_result - integral) < 1e-8

    # Quadrature of order 1
    points, weights = t.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def func_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = b / 2. + a_1 / 6. + a_2 / 6.
        integral = t.volume * sum([func_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.linalg.norm(exact_result - integral) < 1e-8


test_triangle_mappings()
test_triangle_quadrature()
