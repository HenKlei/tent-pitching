import numpy as np

from tent_pitching.geometry.quadrature import gauss_quadrature


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


class Quadrilateral:
    def __init__(self, vertices):
        assert len(vertices) == 4
        self.vertices = vertices

    def quadrature(self, order):
        points = []
        weights = []

        points_1d, weights_1d = gauss_quadrature(order)

        for p_1, w_1 in zip(points_1d, weights_1d):
            for p_2, w_2 in zip(points_1d, weights_1d):
                points.append(np.concatenate((p_1, p_2), axis=None))
                weights.append(w_1 * w_2)

        return points, weights
