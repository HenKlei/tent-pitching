import numpy as np


points = [[0.], [-1./np.sqrt(3.), 1./np.sqrt(3.)], [0., -np.sqrt(3./5.), np.sqrt(3./5.)],
          [-np.sqrt(3./7.-2./7.*np.sqrt(6./5.)), np.sqrt(3./7.-2./7.*np.sqrt(6./5.)),
           -np.sqrt(3./7.+2./7.*np.sqrt(6./5.)), np.sqrt(3./7.+2./7.*np.sqrt(6./5.))]]
weights = [[2.], [1., 1.], [8./9., 5./9., 5./9.],
           [(18.+np.sqrt(30.))/36., (18.+np.sqrt(30.))/36.,
            (18.-np.sqrt(30.))/36., (18.-np.sqrt(30.))/36.]]


def gauss_quadrature(order=2):
    assert 0 <= order <= 3
    return [np.array([0.5 + 0.5 * p]) for p in points[order]], [0.5 * w for w in weights[order]]
