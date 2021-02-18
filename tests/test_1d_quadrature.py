import numpy as np

from tent_pitching.discretizations import gauss_quadrature


def test_gauss_quadrature():
    assert np.abs(gauss_quadrature(lambda x: x**2 + 2.*x + 4., 0., 1.) - 16./3.) < 1e-8
    assert np.abs(gauss_quadrature(lambda x: x**3, 0., 1.) - 0.25) < 1e-8
  
  
test_gauss_quadrature()
