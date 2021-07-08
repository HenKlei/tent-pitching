import numpy as np

from tent_pitching.geometry.quadrature import gauss_quadrature
from tent_pitching.utils.helper_functions import flatten, is_left


class Entity:
    def get_subentities(self, codim=0):
        assert 0 <= codim <= self.dim
        if codim == 0:
            return self
        elif codim == 1:
            return self.subentities
        return flatten([s.get_subentities(codim=codim-1) for s in self.subentities])

    def to_global(self, x_hat):
        raise NotImplementedError

    def derivative_to_global(self, x_hat):
        raise NotImplementedError

    def to_local(self, x):
        raise NotImplementedError

    def derivative_to_local(self, x):
        raise NotImplementedError

    def quadrature(self, order):
        raise NotImplementedError

    def center(self):
        raise NotImplementedError

    def volume(self):
        raise NotImplementedError


class Vertex(Entity):
    def __init__(self, coordinates):
        assert coordinates.ndim == 1
        self.dim = 0
        self.world_dim = coordinates.shape[0]
        self.codim = self.world_dim - self.dim
        self.subentities = []
        self.coordinates = coordinates

    def __containes__(self, x):
        return x == self.coordinates

    def to_global(self, x_hat):
        assert x_hat == np.zeros(self.world_dim)
        return self.coordinates

    def derivative_to_global(self, x_hat):
        raise NotImplementedError

    def to_local(self, x):
        assert x == self.coordinates
        return np.zeros(self.world_dim)

    def derivative_to_local(self, x):
        raise NotImplementedError

    def quadrature(self, order):
        raise NotImplementedError

    def center(self):
        return self.coordinates

    def volume(self):
        return 1.


class Line(Entity):
    def __init__(self, vertices, inside=None, outside=None):
        assert len(vertices) == 2
        assert all(isinstance(vertex, Vertex) for vertex in vertices)
        self.dim = 1
        self.world_dim = vertices[0].world_dim
        self.codim = self.world_dim - self.dim
        self.subentities = vertices
        self.A = self.subentities[1].coordinates-self.subentities[0].coordinates
        self.b = self.subentities[0].coordinates
        self.inside = inside
        self.outside = outside

    def __contains__(self, x):
        assert x.shape == (self.world_dim,)
        x_transformed = x - self.b
        divided = np.divide(x_transformed, self.A, where=(self.A != 0))
        e = divided[(divided != 0).argmax()]
        return (np.all(divided == e, where=(self.A != 0))
                and np.all(x_transformed == 0, where=(self.A == 0)))

    def to_global(self, x_hat):
        assert x_hat.shape == (1,)
        assert 0. <= x_hat[0] <= 1.
        return x_hat * self.A + self.b

    def derivative_to_global(self, x_hat):
        return self.A.T

    def to_local(self, x):
        assert x in self
        x_transformed = x - self.b
        divided = np.divide(x_transformed, self.A, where=(self.A != 0))
        return np.array([divided[(divided != 0).argmax()]])

    def derivative_to_local(self, x):
        raise NotImplementedError

    def quadrature(self, order):
        return gauss_quadrature(order)

    def outer_unit_normal(self):
        assert self.world_dim == 2
        direction = self.subentities[1].coordinates - self.subentities[0].coordinates
        orthogonal_vector = np.array([direction[1], -direction[0]])
        return orthogonal_vector / np.linalg.norm(orthogonal_vector)

    def center(self):
        return self.to_global(np.array([0.5, ]))

    def volume(self):
        return np.linalg.norm(self.subentities[1].coordinates-self.subentities[0].coordinates)


class Triangle(Entity):
    def __init__(self, lines):
        assert len(lines) == 3
        seen = set()
        seen_add = seen.add
        vertices = [s for line in lines for s in line.subentities if not (s in seen or seen_add(s))]
        assert len(vertices) == 3
        """
        if not is_left(vertices[0], vertices[1], vertices[2]):
            tmp_2 = vertices[2]
            vertices[2] = vertices[1]
            vertices[1] = tmp_2
        """
        self.A = np.array([vertices[1].coordinates-vertices[0].coordinates,
                           vertices[2].coordinates-vertices[0].coordinates]).T
        self.A_inv = np.linalg.inv(self.A)
        self.b = vertices[0].coordinates

        self.subentities = lines
        self.dim = 2
        self.world_dim = self.subentities[0].world_dim
        self.codim = self.world_dim - self.dim

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
            return [np.array([1./3., 1./3.] + [0, ] * self.codim)], [1.]
        elif order == 1:
            return ([np.array([0., 0.] + [0, ] * self.codim),
                     np.array([1., 0.] + [0, ] * self.codim),
                     np.array([0., 1.] + [0, ] * self.codim)],
                    [1./3., 1./3., 1./3.])

    def center(self):
        return self.to_global(np.array([1./3., 1./3.] + [0., ] * self.codim))

    def volume(self):
        return 0.5 * np.abs(np.linalg.det(self.A))


class Quadrilateral(Entity):
    def __init__(self, lines):
        assert len(lines) == 4
        seen = set()
        seen_add = seen.add
        vertices = [s for line in lines for s in line.subentities if not (s in seen or seen_add(s))]
        assert len(vertices) == 4
        """
        if (is_left(vertices[0], vertices[1], vertices[2])
           and is_left(vertices[0], vertices[1], vertices[3])):
            if not is_left(vertices[1], vertices[2], vertices[3]):
                tmp_2 = vertices[2]
                vertices[2] = vertices[3]
                vertices[3] = tmp_2
        elif is_left(vertices[0], vertices[1], vertices[2]):
            tmp_1 = vertices[1]
            tmp_2 = vertices[2]
            vertices[1] = vertices[3]
            vertices[2] = tmp_1
            vertices[3] = tmp_2
        elif is_left(vertices[0], vertices[1], vertices[3]):
            tmp_1 = vertices[1]
            vertices[1] = vertices[2]
            vertices[2] = tmp_1
        elif is_left(vertices[0], vertices[3], vertices[2]):
            tmp_1 = vertices[1]
            vertices[1] = vertices[3]
            vertices[3] = tmp_1
        else:
            tmp_1 = vertices[1]
            tmp_2 = vertices[2]
            vertices[2] = vertices[3]
            vertices[1] = tmp_2
            vertices[3] = tmp_1
        """
        assert (is_left(vertices[0], vertices[1], vertices[2])
                and is_left(vertices[0], vertices[1], vertices[3])
                and is_left(vertices[1], vertices[2], vertices[3])
                and is_left(vertices[1], vertices[2], vertices[0])
                and is_left(vertices[2], vertices[3], vertices[0])
                and is_left(vertices[2], vertices[3], vertices[1])
                and is_left(vertices[3], vertices[0], vertices[1])
                and is_left(vertices[3], vertices[0], vertices[2]))

        self.vertices = vertices
        self.A = np.array([self.vertices[1].coordinates-self.vertices[0].coordinates,
                           self.vertices[3].coordinates-self.vertices[0].coordinates]).T
        self.b = self.vertices[0].coordinates

        self.subentities = lines

        self.dim = 2
        self.world_dim = self.subentities[0].world_dim
        self.codim = self.world_dim - self.dim

    def to_global(self, x_hat):
        assert x_hat.shape == (2,)
        assert 0. <= x_hat[0] <= 1. and 0. <= x_hat[1] <= 1.
        return self.A.dot(x_hat) + self.b + x_hat[0] * x_hat[1] * (self.vertices[2].coordinates
                                                                   + self.vertices[0].coordinates
                                                                   - self.vertices[1].coordinates
                                                                   - self.vertices[3].coordinates)

    def derivative_to_global(self, x_hat):
        return self.A.T + np.array([x_hat[1] * (self.vertices[2].coordinates
                                                + self.vertices[0].coordinates
                                                - self.vertices[1].coordinates
                                                - self.vertices[3].coordinates),
                                    x_hat[0] * (self.vertices[2].coordinates
                                                + self.vertices[0].coordinates
                                                - self.vertices[1].coordinates
                                                - self.vertices[3].coordinates)]).T

    def to_local(self, x):
        a_1 = self.vertices[0].coordinates
        a_2 = self.vertices[1].coordinates - self.vertices[0].coordinates
        a_3 = self.vertices[3].coordinates - self.vertices[0].coordinates
        a_4 = (self.vertices[2].coordinates + self.vertices[0].coordinates
               - self.vertices[1].coordinates - self.vertices[3].coordinates)

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
                points.append(np.concatenate((np.concatenate((p_1, p_2), axis=None),
                                             np.zeros(self.codim)), axis=None))
                weights.append(w_1 * w_2)

        return points, weights

    def center(self):
        return self.to_global(np.array([0.5, 0.5]))

    def volume(self):
        return 0.5 * (self.vertices[0].coordinates[0] * self.vertices[1].coordinates[1]
                      + self.vertices[1].coordinates[0] * self.vertices[2].coordinates[1]
                      + self.vertices[2].coordinates[0] * self.vertices[3].coordinates[1]
                      + self.vertices[3].coordinates[0] * self.vertices[0].coordinates[1]
                      - self.vertices[1].coordinates[0] * self.vertices[0].coordinates[1]
                      - self.vertices[2].coordinates[0] * self.vertices[1].coordinates[1]
                      - self.vertices[3].coordinates[0] * self.vertices[2].coordinates[1]
                      - self.vertices[0].coordinates[0] * self.vertices[3].coordinates[1])
