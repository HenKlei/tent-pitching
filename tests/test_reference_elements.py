import numpy as np

from tent_pitching.geometry.entities import Triangle


def test_triangle_mappings():
    vertices = (np.array([1, 1]), np.array([2, 2]), np.array([0, 2]))
    t = Triangle(vertices)
    assert np.linalg.norm(t.to_global(np.array([0.25, 0.25])) - np.array([1., 1.5])) < 1e-8
    assert np.linalg.norm(t.to_local(np.array([1., 1.5])) - np.array([0.25, 0.25])) < 1e-8
    assert np.abs(t.volume - 1.) < 1e-8


test_triangle_mappings()
