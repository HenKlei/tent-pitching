import numpy as np


class Triangle:
    def __init__(self, vertices):
        assert len(vertices) == 3
        self.vertices = vertices
        self.A = np.array([self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]]).T
        self.A_inv = np.linalg.inv(self.A)
        self.b = self.vertices[0]
        self.volume = 0.5 * np.abs(np.linalg.det(self.A))

    def to_global(self, x_hat):
        assert x_hat.shape == (2,)
        assert 0. <= x_hat[0] <= 1. and 0. <= x_hat[1] <= 1.
        assert x_hat[0] + x_hat[1] <= 1.
        return self.A.dot(x_hat) + self.b

    def to_local(self, x):
        assert x.shape == (2,)
        return self.A_inv.dot(x) - self.A_inv.dot(self.b)

    def quadrature(self, order):
        assert order in (0, 1)
        if order == 0:
            return [np.array([1./3., 1./3.])], [1.]
        elif order == 1:
            return ([np.array([0., 0.]), np.array([1., 0.]), np.array([0., 1.])],
                    [1./3., 1./3., 1./3.])
