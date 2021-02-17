import numpy as np
import matplotlib.pyplot as plt


class DGFunction:
    def __init__(self, element, local_space_grid_size=1e-1):
        self.element = element
        self.local_space_grid_size = local_space_grid_size
        self.function = np.zeros(int(1. / local_space_grid_size))

    def interpolate(self, u):
        for i in range(len(self.function)):
            local_coordinate = self.local_space_grid_size / 2. + i * self.local_space_grid_size
            self.function[i] = u(self.element.to_global(local_coordinate))

    def get_values(self):
        return self.function

    def set_values(self, values):
        self.function = values

    def get_function_values(self):
        x = []
        y = []
        for i in range(len(self.function)):
            local_coordinate = self.local_space_grid_size / 2. + i * self.local_space_grid_size
            x.append(self.element.to_global(local_coordinate))
            y.append(self.function[i])
        return x, y

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        assert self.element == other.element
        assert self.local_space_grid_size == other.local_space_grid_size

        tmp = DGFunction(self.element, local_space_grid_size=self.local_space_grid_size)
        tmp.function = self.function + other.function
        return tmp

    def __rmul__(self, other):
        assert isinstance(other, float)

        tmp = DGFunction(self.element, local_space_grid_size=self.local_space_grid_size)
        tmp.function = other * self.function
        return tmp

    def __sub__(self, other):
        return self + (-1.) * other


class LocalSpaceTimeFunction:
    def __init__(self, tent, LocalSpaceFunctionType, local_space_grid_size=1e-1, local_time_grid_size=1e-1):
        self.tent = tent
        self.local_time_grid_size = local_time_grid_size

        self.function = []
        for element in tent.get_space_patch().get_elements():
            tmp = []
            for _ in range(int(1. / local_time_grid_size) + 1):
                tmp.append(LocalSpaceFunctionType(element, local_space_grid_size=local_space_grid_size))
            self.function.append(tmp)

    def set_value(self, time, function_list):
        assert isinstance(time, int)
        assert 0 <= time <= int(1. / self.local_time_grid_size) + 1
        assert len(function_list) == len(self.function)

        for function in function_list:
            self.function[self.tent.get_space_patch().get_elements().index(function.element)][time] = function

    def set_initial_value(self, function_list):
        self.set_value(0, function_list)

    def set_initial_value_per_element(self, func):
        assert func.element in self.tent.get_space_patch().get_elements()
        self.function[self.tent.get_space_patch().get_elements().index(func.element)][0] = func

    def get_value(self, time):
        assert isinstance(time, int)
        assert 0 <= time <= int(1. / self.local_time_grid_size) + 1

        return [self.function[i][time] for i in range(len(self.function))]

    def get_initial_value(self):
        return self.get_value(0)

    def get_function_values(self):
        x = []
        t = []
        y = []

        for element_functions in self.function:
            for n, func in enumerate(element_functions):
                t_ref = n * self.local_time_grid_size
                tmp = func.get_function_values()
                x.append(tmp[0])
                tmp2 = []
                for x_val in tmp[0]:
                    x_ref = self.tent.get_space_patch().to_local(x_val)
                    tmp2.append(self.tent.get_time_transformation(x_ref, t_ref))
                t.append(tmp2)
                y.append(tmp[1])

        return x, t, y
