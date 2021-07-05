import numpy as np

from tent_pitching.functions.local_functions import P1DGLocalFunction
from tent_pitching.geometry.entities import Triangle, Quadrilateral


EPS = 1e-8


def test_triangle():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]))
    t = Triangle(vertices)
    local_values = [1., 2., 2.]
    func = P1DGLocalFunction(t, local_values=local_values)
    assert np.abs(func(np.array([1., 1.5])) - 1.) < EPS
    assert np.linalg.norm(func.gradient(np.array([1., 1.5])) - np.array([0., 4.])) < EPS


def test_quadrilateral():
    vertices = (np.array([1., 1.]), np.array([2., 2.]), np.array([0., 2.]), np.array([-1., 1.]))
    q = Quadrilateral(vertices)
    local_values = [2., 2., 1.]
    func = P1DGLocalFunction(q, local_values=local_values)
    assert np.abs(func(np.array([.75, 1.25])) - 2.) < EPS
    assert np.linalg.norm(func.gradient(np.array([0., 1.25])) - np.array([1., -1.])) < EPS


test_triangle()

test_quadrilateral()
