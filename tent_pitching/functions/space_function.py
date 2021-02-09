import numpy as np


class SpaceFunction:
    """
    Function that is defined for each element in the space grid on its reference element.
    """
    def __init__(self, space_grid, u, local_space_grid_size=1e-1):
        self.space_grid = space_grid
        self.function = []

        for element in self.space_grid.elements:
            tmp = np.zeros(int(1. / local_space_grid_size) + 1)
            for i, x_hat in enumerate(np.linspace(0., 1., len(tmp))):
                tmp[i] = u(element.to_global(x_hat))
            self.function.append(tmp)

    def get_function_on_element(self, element):
        assert element in self.space_grid.elements
        return self.function[self.space_grid.elements.index(element)]

    def get_function_global(self):
        return (np.vstack([element.to_global(np.linspace(0., 1., len(self.function[0]))) for element in self.space_grid.elements]).ravel(), np.vstack(self.function).ravel())
