import numpy as np

from tent_pitching.geometry.entities import Triangle, Quadrilateral


EPS = 1e-8


def test_triangle_mappings():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]))
    t = Triangle(vertices)
    assert np.linalg.norm(t.to_global(np.array([0., 0.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(t.to_global(np.array([1., 0.])) - np.array([2., 2.])) < EPS
    assert np.linalg.norm(t.to_global(np.array([0., 1.])) - np.array([0., 2.])) < EPS
    assert np.linalg.norm(t.to_global(np.array([0.25, 0.25])) - np.array([1., 1.5])) < EPS
    assert np.linalg.norm(t.to_local(np.array([1., 1.5])) - np.array([0.25, 0.25])) < EPS
    assert np.abs(t.volume - 1.) < EPS


def test_triangle_quadrature():
    vertices = (np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]))
    t = Triangle(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = t.quadrature(0)

    for b in np.random.rand(N):
        def function_to_integrate(x):
            return b

        exact_result = b / 2.
        integral = t.volume * sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.abs(exact_result - integral) < EPS

    # Quadrature of order 1
    points, weights = t.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def function_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = b / 2. + a_1 / 6. + a_2 / 6.
        integral = t.volume * sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.abs(exact_result - integral) < EPS


def test_quadrilateral_mappings():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]), np.array([-1., 1.]))
    q = Quadrilateral(vertices)
    assert np.abs(q.volume - 2.) < EPS
    assert np.linalg.norm(q.to_global(np.array([0., 0.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(q.to_global(np.array([1., 0.])) - np.array([2., 2.])) < EPS
    assert np.linalg.norm(q.to_global(np.array([0., 1.])) - np.array([-1., 1.])) < EPS
    assert np.linalg.norm(q.to_global(np.array([1., 1.])) - np.array([0., 2.])) < EPS
    assert np.linalg.norm(q.to_global(np.array([0.25, 0.25])) - np.array([.75, 1.25])) < EPS
    assert np.linalg.norm(q.to_local(np.array([1., 1.])) - np.array([0., 0.])) < EPS
    assert np.linalg.norm(q.to_local(np.array([2., 2.])) - np.array([1., 0.])) < EPS
    assert np.linalg.norm(q.to_local(np.array([0., 2.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(q.to_local(np.array([-1., 1.])) - np.array([0., 1.])) < EPS
    assert np.linalg.norm(q.to_local(np.array([.75, 1.25])) - np.array([0.25, 0.25])) < EPS


def test_quadrilateral_quadrature():
    vertices = (np.array([0., 0.]), np.array([1., 0.]), np.array([1., 1.]), np.array([0., 1.]))
    q = Quadrilateral(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = q.quadrature(0)

    for b in np.random.rand(N):
        def function_to_integrate(x):
            return b

        exact_result = b
        integral = q.volume * sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS

    # Quadrature of order 1
    points, weights = q.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def function_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = b + 0.5 * (a_1 + a_2)
        integral = q.volume * sum([function_to_integrate(x) * w for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS


test_triangle_mappings()
test_triangle_quadrature()

test_quadrilateral_mappings()
test_quadrilateral_quadrature()
