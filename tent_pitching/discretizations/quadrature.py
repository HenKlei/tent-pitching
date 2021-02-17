import numpy as np


points = [[0.], [-1./np.sqrt(3.), 1./np.sqrt(3.)], [0., -np.sqrt(3./5.), np.sqrt(3./5.)], [-np.sqrt(3./7.-2./7.*np.sqrt(6./5.)),np.sqrt(3./7.-2./7.*np.sqrt(6./5.)), -np.sqrt(3./7.+2./7.*np.sqrt(6./5.)), np.sqrt(3./7.+2./7.*np.sqrt(6./5.))]]
weights = [[2.], [1., 1.], [8./9., 5./9., 5./9.], [(18.+np.sqrt(30.))/36., (18.+np.sqrt(30.))/36., (18.-np.sqrt(30.))/36., (18.-np.sqrt(30.))/36.]]

def gauss_quadrature(f, a, b, n=2):
    assert 1 <= n <= 4
    return (b - a) / 2. * sum(w*f_val for w, f_val in zip(weights[n-1], [f((b-a)/2.*p+(a+b)/2.) for p in points[n-1]]))


def test_gauss_quadrature():
    assert np.abs(gauss_quadrature(lambda x: x**2 + 2.*x + 4., 0., 1.) - 16./3.) < 1e-8
    assert np.abs(gauss_quadrature(lambda x: x**3, 0., 1.) - 0.25) < 1e-8


test_gauss_quadrature()
