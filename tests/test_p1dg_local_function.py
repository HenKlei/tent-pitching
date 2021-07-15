import numpy as np

from tent_pitching.functions.local_functions import P1DGLocalFunction
from tent_pitching.geometry.reference_elements import Vertex, Line, Triangle, Quadrilateral


EPS = 1e-8


def test_triangle():
    vertices = [Vertex(np.array([1., 1.])), Vertex(np.array([2., 2.])), Vertex(np.array([0., 2.]))]
    lines = [Line([vertices[0], vertices[1]]), Line([vertices[1], vertices[2]]),
             Line([vertices[2], vertices[0]])]
    t = Triangle(lines)
    local_values = np.array([1., 2., 2.])
    func = P1DGLocalFunction(t, local_values=local_values)
    assert np.abs(func(np.array([1., 1.5])) - 1.) < EPS
    assert np.linalg.norm(func.gradient(np.array([1., 1.5])) - np.array([0., 4.])) < EPS


def test_quadrilateral():
    vertices = [Vertex(np.array([1., 1.])), Vertex(np.array([2., 2.])),
                Vertex(np.array([0., 2.])), Vertex(np.array([-1., 1.]))]
    lines = [Line([vertices[0], vertices[1]]), Line([vertices[1], vertices[2]]),
             Line([vertices[2], vertices[3]]), Line([vertices[3], vertices[0]])]
    q = Quadrilateral(lines)
    local_values = np.array([2., 2., 1.])
    func = P1DGLocalFunction(q, local_values=local_values)
    assert np.abs(func(np.array([.75, 1.25])) - 2.) < EPS
    assert np.linalg.norm(func.gradient(np.array([0., 1.25])) - np.array([1., -1.])) < EPS


def test_constant_function():
    vertices = [Vertex(np.array([1., 1.])), Vertex(np.array([2., 2.])),
                Vertex(np.array([0., 2.])), Vertex(np.array([-1., 1.]))]
    lines = [Line([vertices[0], vertices[1]]), Line([vertices[1], vertices[2]]),
             Line([vertices[2], vertices[3]]), Line([vertices[3], vertices[0]])]
    q = Quadrilateral(lines)
    local_values = np.array([2., 2., 2.])
    func = P1DGLocalFunction(q, local_values=local_values)
    for x in np.linspace(0., 1., 100):
        for y in np.linspace(0., 1., 100):
            assert np.abs(func(q.to_global(np.array([x, y]))) - 2.) < EPS


test_triangle()

test_quadrilateral()

test_constant_function()
