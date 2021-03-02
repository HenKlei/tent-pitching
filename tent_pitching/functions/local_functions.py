import numpy as np


class DGFunction:
    def __init__(self, element, local_space_grid_size=1e-1):
        self.element = element
        self.local_space_grid_size = local_space_grid_size
        self.function = np.zeros(int(1. / local_space_grid_size))

    def interpolate(self, u):
        # Integration would be preferable here!
        for i in range(len(self.function)):
            local_coordinate = (0.5 + i) * self.local_space_grid_size
            self.function[i] = u(self.element.vertex_left.coordinate +
                                 local_coordinate * (self.element.vertex_right.coordinate -
                                                     self.element.vertex_left.coordinate))

    def get_values(self):
        return self.function

    def set_values(self, values):
        assert self.function.shape == values.shape
        self.function = values

    def get_function_values(self):
        x_vals = []
        y_vals = []
        for i in range(len(self.function)):
            local_coordinate = (0.5 + i) * self.local_space_grid_size
            x_vals.append(self.element.vertex_left.coordinate +
                          local_coordinate * (self.element.vertex_right.coordinate -
                                              self.element.vertex_left.coordinate))
            y_vals.append(self.function[i])
        return x_vals, y_vals

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        assert self.element == other.element
        assert self.function.shape == other.function.shape

        tmp = DGFunction(self.element, local_space_grid_size=self.local_space_grid_size)
        tmp.function = self.function + other.function
        return tmp

    def __rmul__(self, other):
        assert isinstance(other, float)

        tmp = DGFunction(self.element, local_space_grid_size=self.local_space_grid_size)
        tmp.function = other * self.function
        return tmp

    def __truediv__(self, other):
        assert isinstance(other, float)

        return (1. / other) * self

    def __sub__(self, other):
        return self + (-1.) * other


class LocalSpaceTimeFunction:
    def __init__(self, tent, LocalSpaceFunctionType, local_space_grid_size=1e-1,
                 local_time_grid_size=1e-1):
        self.tent = tent
        self.local_time_grid_size = local_time_grid_size

        self.function = []
        for element in tent.get_space_patch().get_elements():
            tmp = []
            for _ in range(int(1. / local_time_grid_size) + 1):
                tmp.append(LocalSpaceFunctionType(element,
                                                  local_space_grid_size=local_space_grid_size))
            self.function.append(tmp)

    def set_value(self, time, function_list):
        assert isinstance(time, int)
        assert 0 <= time <= int(1. / self.local_time_grid_size) + 1
        assert len(function_list) == len(self.function)

        for function in function_list:
            self.function[self.tent.get_space_patch()
                          .get_elements().index(function.element)][time] = function

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

    def get_function_values(self, transformation):
        x_vals = []
        t_vals = []
        y_vals = []

        for element_functions in self.function:
            for time, func in enumerate(element_functions):
                t_ref = time * self.local_time_grid_size
                tmp = func.get_function_values()
                x_vals.append(tmp[0])
                tmp2 = []
                for x_val in tmp[0]:
                    tmp2.append(self.tent.get_time_transformation(x_val, t_ref))
                t_vals.append(tmp2)

                y_transformed = []
                for x_val, y_val in zip(tmp[0], tmp[1]):
                    phi_2 = self.tent.get_time_transformation(x_val, t_ref)
                    phi_2_dt = self.tent.get_time_transformation_dt(x_val, t_ref)
                    phi_2_dx = self.tent.get_time_transformation_dx(x_val, t_ref)
                    y_transformed.append(transformation(y_val, phi_2, phi_2_dt, phi_2_dx))
                y_vals.append(y_transformed)

        return x_vals, t_vals, y_vals

    def get_function_values_as_matrix(self, transformation):
        _, _, values = self.get_function_values(transformation)
        return np.array(values)
