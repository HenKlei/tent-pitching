import numpy as np

from tent_pitching.geometry.quadrature import gauss_quadrature


class Line:
    def __init__(self, vertices):
        assert len(vertices) == 2
        self.vertices = vertices
        self.A = np.array([self.vertices[1]-self.vertices[0], [0, 0]]).T
        self.b = self.vertices[0]
        self.volume = np.linalg.norm(self.vertices[1]-self.vertices[0])

    def to_global(self, x_hat):
        assert x_hat.shape == (2,)
        assert 0. <= x_hat[0] <= 1. and x_hat[1] == 0.
        return self.A.dot(x_hat) + self.b

    def derivative_to_global(self, x_hat):
        return self.A.T

    def to_local(self, x):
        assert x.shape == (2,)
        x_transformed = x - self.b
        divided = np.divide(x_transformed, self.A.T[0])
        assert np.all(divided == divided[0])
        return np.array([divided[0], 0.])

    def quadrature(self, order):
        points_1d, weights = gauss_quadrature(order)
        points = [np.array([x, 0.]) for x in points_1d]
        return points, weights


class Triangle:
    def __init__(self, vertices):
        assert len(vertices) == 3
        self.vertices = vertices
        self.A = np.array([self.vertices[1]-self.vertices[0], self.vertices[2]-self.vertices[0]]).T
        self.A_inv = np.linalg.inv(self.A)
        self.b = self.vertices[0]
        self.volume = 0.5 * np.abs(np.linalg.det(self.A))

        self.faces = [Line([self.vertices[0], self.vertices[1]]),
                      Line([self.vertices[1], self.vertices[2]]),
                      Line([self.vertices[2], self.vertices[0]])]

    def to_global(self, x_hat):
        assert x_hat.shape == (2,)
        assert 0. <= x_hat[0] <= 1. and 0. <= x_hat[1] <= 1.
        assert x_hat[0] + x_hat[1] <= 1.
        return self.A.dot(x_hat) + self.b

    def derivative_to_global(self, x_hat):
        return self.A.T

    def to_local(self, x):
        assert x.shape == (2,)
        return self.A_inv.dot(x) - self.A_inv.dot(self.b)

    def derivative_to_local(self, x):
        return np.linalg.inv(self.derivative_to_global(self.to_local(x)))

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
        self.A = np.array([self.vertices[1]-self.vertices[0], self.vertices[3]-self.vertices[0]]).T
        self.b = self.vertices[0]
        self.volume = 0.5 * (self.vertices[0][0] * self.vertices[1][1]
                             + self.vertices[1][0] * self.vertices[2][1]
                             + self.vertices[2][0] * self.vertices[3][1]
                             + self.vertices[3][0] * self.vertices[0][1]
                             - self.vertices[1][0] * self.vertices[0][1]
                             - self.vertices[2][0] * self.vertices[1][1]
                             - self.vertices[3][0] * self.vertices[2][1]
                             - self.vertices[0][0] * self.vertices[3][1])

        self.faces = [Line([self.vertices[0], self.vertices[1]]),
                      Line([self.vertices[1], self.vertices[2]]),
                      Line([self.vertices[2], self.vertices[3]]),
                      Line([self.vertices[3], self.vertices[0]])]

    def to_global(self, x_hat):
        assert x_hat.shape == (2,)
        assert 0. <= x_hat[0] <= 1. and 0. <= x_hat[1] <= 1.
        return self.A.dot(x_hat) + self.b + x_hat[0] * x_hat[1] * (self.vertices[2]
                                                                   + self.vertices[0]
                                                                   - self.vertices[1]
                                                                   - self.vertices[3])

    def derivative_to_global(self, x_hat):
        return self.A.T + np.array([x_hat[1] * (self.vertices[2] + self.vertices[0]
                                                - self.vertices[1] - self.vertices[3]),
                                    x_hat[0] * (self.vertices[2] + self.vertices[0]
                                                - self.vertices[1] - self.vertices[3])]).T

    def to_local(self, x):
        a_1 = self.vertices[0]
        a_2 = self.vertices[1] - self.vertices[0]
        a_3 = self.vertices[3] - self.vertices[0]
        a_4 = self.vertices[2] + self.vertices[0] - self.vertices[1] - self.vertices[3]

        if a_4[0] * a_3[1] - a_3[0] * a_4[1] == 0.:  # transformation is linear!
            p = (a_2[0] * a_3[1] + a_4[1] * x[0] - a_1[0] * a_4[1] - a_2[1] * a_3[0]
                 + a_1[1] * a_4[0] - a_4[0] * x[1])
            q = (a_2[1] * x[0] - a_2[1] * a_1[0] + a_1[1] * a_2[0] - a_2[0] * x[1])
            x_hat_2 = -q / p
        else:  # transformation is non-linear!
            p = (1. / (a_4[0] * a_3[1] - a_3[0] * a_4[1])) * (a_2[0] * a_3[1] + a_4[1] * x[0]
                                                              - a_1[0] * a_4[1] - a_2[1] * a_3[0]
                                                              + a_1[1] * a_4[0] - a_4[0] * x[1])
            q = (1. / (a_4[0] * a_3[1] - a_3[0] * a_4[1])) * (a_2[1] * x[0] - a_2[1] * a_1[0]
                                                              + a_1[1] * a_2[0] - a_2[0] * x[1])
            x_hat_2 = -p / 2. - np.sqrt((p / 2.)**2 - q)
            if (a_2[0] + a_4[0] * x_hat_2) <= 0.:
                x_hat_2 = -p / 2. + np.sqrt((p / 2.)**2 - q)
        x_hat_1 = (1. / (a_2[0] + a_4[0] * x_hat_2)) * (x[0] - a_1[0] - a_3[0] * x_hat_2)
        return np.array([x_hat_1, x_hat_2])

    def derivative_to_local(self, x):
        return np.linalg.inv(self.derivative_to_global(self.to_local(x)))

    def quadrature(self, order):
        points = []
        weights = []

        points_1d, weights_1d = gauss_quadrature(order)

        for p_1, w_1 in zip(points_1d, weights_1d):
            for p_2, w_2 in zip(points_1d, weights_1d):
                points.append(np.concatenate((p_1, p_2), axis=None))
                weights.append(w_1 * w_2)

        return points, weights
