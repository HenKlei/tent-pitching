import numpy as np


class SpaceTimeFunction:
    """
    Function that is defined for each tent in the space time grid on its reference element.
    """
    def __init__(self, space_time_grid, local_space_grid_size=1e-1, local_time_grid_size=1e-2):
        self.space_time_grid = space_time_grid
        self.function = []
        for tent in self.space_time_grid.tents:
            self.function.append(np.zeros((int(1. / local_time_grid_size) + 1, (int(1. / local_space_grid_size) + 1) * len(tent.get_space_patch().get_elements()))))

    def set_global_initial_value(self, u_0):
        for i, tent in enumerate(self.space_time_grid.tents):
            if tent.is_initial_boundary():
                self.function[i][0] = np.hstack([u_0.get_function_on_element(element) for element in tent.get_space_patch().get_elements()])

    def set_values_in_tent(self, tent, local_values):
        assert tent in self.space_time_grid.tent
        # !!!!! We have to set the initial values in the next tents above !!!!!
        self.function[self.space_time_grid.tents.index(tent)] = local_values

    def get_initial_value_on_tent(self, tent):
        assert tent in self.space_time_grid.tents
        return self.function[self.space_time_grid.tents.index(tent)][0]
