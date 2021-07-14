import numpy as np

from tent_pitching.geometry.reference_elements import Triangle, Quadrilateral


class P1DGLocalFunction:
    NUM_DOFS = 3

    def __init__(self, element, local_values=np.zeros(3)):
        assert type(element) in (Triangle, Quadrilateral)
        assert local_values.shape == (self.NUM_DOFS,)

        self.element = element

        if type(element) == Triangle:
            self.local_points = [np.array([0.25, 0.25]),
                                 np.array([0.5, 0.25]),
                                 np.array([0.25, 0.5])]
        elif type(element) == Quadrilateral:
            self.local_points = [np.array([0.25, 0.25]),
                                 np.array([0.75, 0.25]),
                                 np.array([0.5, 0.75])]

        self.local_values = local_values

    def __call__(self, x):
        return self.evaluate(x)

    def __add__(self, u):
        if isinstance(u, P1DGLocalFunction):
            self.set_values(self.local_values + u.local_values)
            return self
        assert u.shape == (self.NUM_DOFS,)
        self.set_values(self.local_values + u)
        return self

    def __mul__(self, x):
        assert isinstance(x, int) or isinstance(x, float)
        self.set_values(x * self.local_values)
        return self

    __rmul__ = __mul__

    def set_values(self, u):
        assert u.shape == (self.NUM_DOFS,)
        self.local_values = u

    def evaluate(self, x):
        return self.evaluate_local(self.element.to_local(x))

    def evaluate_local(self, x_hat):
        mat = np.array([self.local_points[1]-self.local_points[0],
                        self.local_points[2]-self.local_points[0]])
        rhs = np.array([self.local_values[1]-self.local_values[0],
                        self.local_values[2]-self.local_values[0]])
        A = np.linalg.solve(mat, rhs)
        b = self.local_values[0] - A.dot(self.local_points[0])
        return A.dot(x_hat) + b

    def gradient(self, x):
        return self.element.derivative_to_local(x).dot(
            self.gradient_local(self.element.to_local(x)))

    def gradient_local(self, x_hat):
        mat = np.array([self.local_points[1]-self.local_points[0],
                        self.local_points[2]-self.local_points[0]])
        rhs = np.array([self.local_values[1]-self.local_values[0],
                        self.local_values[2]-self.local_values[0]])
        return np.linalg.solve(mat, rhs)

    def interpolate(self, u):
        values = np.zeros(self.NUM_DOFS)
        for i, x_hat in enumerate(self.local_points):
            values[i] = u(self.element.to_global(x_hat))
        self.set_values(values)
