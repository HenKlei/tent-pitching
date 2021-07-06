import numpy as np

from tent_pitching.geometry.entities import Line, Triangle, Quadrilateral


EPS = 1e-8


def test_line_mappings():
    vertices = (np.array([1., 1.]), np.array([3., 2.]))
    line = Line(vertices)
    assert np.linalg.norm(line.to_global(np.array([0., 0.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(line.to_global(np.array([1., 0.])) - np.array([3., 2.])) < EPS
    assert np.linalg.norm(line.to_global(np.array([0.25, 0.])) - np.array([1.5, 1.25])) < EPS
    assert np.linalg.norm(line.to_global(np.array([0.5, 0.])) - np.array([2., 1.5])) < EPS
    assert np.linalg.norm(line.to_global(np.array([0.75, 0.])) - np.array([2.5, 1.75])) < EPS
    assert np.linalg.norm(line.to_local(np.array([2.5, 1.75])) - np.array([0.75, 0.])) < EPS
    assert np.abs(line.volume - np.sqrt(5.)) < EPS
    assert np.linalg.norm(line.outer_unit_normal() - np.array([1, -2]) / np.sqrt(5.)) < EPS


def test_line_quadrature():
    vertices = (np.array([1., 1.]), np.array([3., 2.]))
    line = Line(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = line.quadrature(0)

    for b in np.random.rand(N):
        def function_to_integrate(x):
            return b

        exact_result = b * np.sqrt(5.)
        integral = line.volume * sum([function_to_integrate(line.to_global(x)) * w
                                      for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS

    # Quadrature of order 1
    points, weights = line.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def function_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = (b + 2. * a_1 + 1.5 * a_2) * np.sqrt(5.)
        integral = line.volume * sum([function_to_integrate(line.to_global(x)) * w
                                      for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS


def test_triangle_mappings():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]))
    triangle = Triangle(vertices)
    assert np.linalg.norm(triangle.to_global(np.array([0., 0.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(triangle.to_global(np.array([1., 0.])) - np.array([2., 2.])) < EPS
    assert np.linalg.norm(triangle.to_global(np.array([0., 1.])) - np.array([0., 2.])) < EPS
    assert np.linalg.norm(triangle.to_global(np.array([0.25, 0.25])) - np.array([1., 1.5])) < EPS
    assert np.linalg.norm(triangle.to_local(np.array([1., 1.5])) - np.array([0.25, 0.25])) < EPS
    assert np.abs(triangle.volume - 1.) < EPS
    assert np.linalg.norm(triangle.outer_unit_normal(np.array([1., 2.])) - np.array([0., 1.])) < EPS


def test_triangle_quadrature():
    vertices = (np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.]))
    triangle = Triangle(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = triangle.quadrature(0)

    for b in np.random.rand(N):
        def function_to_integrate(x):
            return b

        exact_result = b / 2.
        integral = triangle.volume * sum([function_to_integrate(triangle.to_global(x)) * w
                                          for x, w in zip(points, weights)])
        assert np.abs(exact_result - integral) < EPS

    # Quadrature of order 1
    points, weights = triangle.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def function_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = b / 2. + a_1 / 6. + a_2 / 6.
        integral = triangle.volume * sum([function_to_integrate(triangle.to_global(x)) * w
                                          for x, w in zip(points, weights)])
        assert np.abs(exact_result - integral) < EPS


def test_quadrilateral_mappings():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]), np.array([-1., 1.]))
    quadrilateral = Quadrilateral(vertices)
    assert np.linalg.norm(quadrilateral.to_global(np.array([0., 0.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(quadrilateral.to_global(np.array([1., 0.])) - np.array([2., 2.])) < EPS
    assert np.linalg.norm(quadrilateral.to_global(np.array([0., 1.])) - np.array([-1., 1.])) < EPS
    assert np.linalg.norm(quadrilateral.to_global(np.array([1., 1.])) - np.array([0., 2.])) < EPS
    assert np.linalg.norm(quadrilateral.to_global(np.array([0.25, 0.25]))
                          - np.array([.75, 1.25])) < EPS
    assert np.linalg.norm(quadrilateral.to_local(np.array([1., 1.])) - np.array([0., 0.])) < EPS
    assert np.linalg.norm(quadrilateral.to_local(np.array([2., 2.])) - np.array([1., 0.])) < EPS
    assert np.linalg.norm(quadrilateral.to_local(np.array([0., 2.])) - np.array([1., 1.])) < EPS
    assert np.linalg.norm(quadrilateral.to_local(np.array([-1., 1.])) - np.array([0., 1.])) < EPS
    assert np.linalg.norm(quadrilateral.to_local(np.array([.75, 1.25]))
                          - np.array([0.25, 0.25])) < EPS
    assert np.abs(quadrilateral.volume - 2.) < EPS


def test_quadrilateral_quadrature():
    vertices = (np.array([0., 0.]), np.array([1., 0.]), np.array([1., 1.]), np.array([0., 1.]))
    quadrilateral = Quadrilateral(vertices)
    N = 100

    # Quadrature of order 0
    points, weights = quadrilateral.quadrature(0)

    for b in np.random.rand(N):
        def function_to_integrate(x):
            return b

        exact_result = b
        integral = quadrilateral.volume * sum([function_to_integrate(quadrilateral.to_global(x)) * w
                                               for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS

    # Quadrature of order 1
    points, weights = quadrilateral.quadrature(1)

    for b, a_1, a_2 in zip(np.random.rand(N), np.random.rand(N), np.random.rand(N)):
        def function_to_integrate(x):
            return b + a_1 * x[0] + a_2 * x[1]

        exact_result = b + 0.5 * (a_1 + a_2)
        integral = quadrilateral.volume * sum([function_to_integrate(quadrilateral.to_global(x)) * w
                                               for x, w in zip(points, weights)])
        assert np.abs(integral - exact_result) < EPS


test_line_mappings()
test_line_quadrature()

test_triangle_mappings()
test_triangle_quadrature()

test_quadrilateral_mappings()
test_quadrilateral_quadrature()
