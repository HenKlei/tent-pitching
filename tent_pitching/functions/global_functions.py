from tent_pitching.functions import LocalSpaceTimeFunction


class SpaceFunction:
    """
    Function that is defined for each element in the space grid on its reference element.
    """
    def __init__(self, space_grid, LocalSpaceFunctionType, u=None, local_space_grid_size=1e-1):
        self.space_grid = space_grid

        self.function = [LocalSpaceFunctionType(element,
                                                local_space_grid_size=local_space_grid_size)
                         for element in space_grid.elements]

        if u is not None:
            self.interpolate(u)

    def interpolate(self, u):
        for local_function in self.function:
            local_function.interpolate(u)

    def get_function_on_element(self, element):
        assert element in self.space_grid.elements
        return self.function[self.space_grid.elements.index(element)]

    def get_function_on_space_patch(self, patch):
        return [self.function[self.space_grid.elements.index(element)]
                for element in patch.get_elements()]

    def get_function_values(self):
        return [self.function[i].get_function_values() for i in range(len(self.function))]


class SpaceTimeFunction:
    """
    Function that is defined for each tent in the space time grid on its reference element.
    """
    def __init__(self, space_time_grid, LocalSpaceFunctionType,
                 local_space_grid_size=1e-1, local_time_grid_size=1e-1):
        self.space_time_grid = space_time_grid

        self.function = [LocalSpaceTimeFunction(tent, LocalSpaceFunctionType,
                                                local_space_grid_size=local_space_grid_size,
                                                local_time_grid_size=local_time_grid_size)
                         for tent in space_time_grid.tents]

    def set_global_initial_value(self, u_0):
        assert isinstance(u_0, SpaceFunction)
        for i, tent in enumerate(self.space_time_grid.tents):
            for element in tent.get_initial_boundary_elements():
                self.function[i].set_initial_value_per_element(u_0.get_function_on_element(element))

    def set_function_on_tent(self, tent, local_function):
        assert tent in self.space_time_grid.tents
        assert isinstance(local_function, LocalSpaceTimeFunction)
        # Check if for each element the values from the tent below and above fit together!
        if not tent.has_initial_boundary():
            pass
            # assert local_function.get_value(0)

        self.function[self.space_time_grid.tents.index(tent)] = local_function
        # Set initial values for neighboring tents above!
        print("|   |   Setting initial values on neighboring tents...")
        for neighboring_tent, element in tent.neighboring_tents_above:
            print(f"|   |   |   On {neighboring_tent}")
            func = (local_function.get_value(len(local_function.function[0]) - 1)
                    [tent.get_space_patch().get_elements().index(element)])
            self.function[self.space_time_grid.tents
                          .index(neighboring_tent)].set_initial_value_per_element(func)

    def get_function_on_tent(self, tent):
        assert tent in self.space_time_grid.tents
        return self.function[self.space_time_grid.tents.index(tent)]

    def get_initial_value_on_tent(self, tent):
        return self.get_function_on_tent(tent).get_initial_value()

    def get_function_values(self, transformation):
        x_vals = []
        t_vals = []
        y_vals = []

        for func in self.function:
            tmp = func.get_function_values(transformation)
            x_vals.append(tmp[0])
            t_vals.append(tmp[1])
            y_vals.append(tmp[2])

        return x_vals, t_vals, y_vals
