import numpy as np


class SpaceTimeVertex:
    def __init__(self, space_vertex, time):
        self.space_vertex = space_vertex
        self.time = time
        self.potential_tent_height = 0.

        self.coordinates = [self.space_vertex.coordinate, self.time, ]

        self.tent_below = None

    def __str__(self):
        return self.space_vertex.__str__() + f"; time: {self.time:.3f}"


class SpaceTimeTent:
    def __init__(self, bottom_space_time_vertex, top_space_time_vertex,
                 space_time_vertices=None, number=None):
        self.bottom_space_time_vertex = bottom_space_time_vertex
        self.top_space_time_vertex = top_space_time_vertex
        assert self.bottom_space_time_vertex.space_vertex == self.top_space_time_vertex.space_vertex
        self.height = top_space_time_vertex.time - bottom_space_time_vertex.time

        self.neighboring_tents_above = []
        self.neighboring_tents_below = []

        self.space_time_vertices = space_time_vertices
        assert len(self.space_time_vertices) <= 4

        self.number = number

    def __str__(self):
        if self.number is not None:
            return (f"Tent number {self.number} pitched above of "
                    f"{self.bottom_space_time_vertex.space_vertex}")
        return f"Tent without number pitched above of {self.bottom_space_time_vertex.space_vertex}"

    def __contains__(self, point):
        x = point[0]
        t = point[1]
        if (x in self.get_space_patch()
           and self.get_bottom_front_value(x) <= t <= self.get_top_front_value(x)):
            return True
        return False

    def has_initial_boundary(self):
        if sum(vertex.time == 0.0 for vertex in self.space_time_vertices) >= 2:
            assert self.bottom_space_time_vertex.time == 0.0
            assert len(self.get_initial_boundary_elements()) > 0
            return True
        assert len(self.get_initial_boundary_elements()) == 0
        return False

    def get_initial_boundary_elements(self):
        elements = []
        try:
            left_vertex = self.get_left_space_time_vertex()
            if left_vertex.time == 0.0:
                elements.append(self.bottom_space_time_vertex.space_vertex.get_left_element())
        except AttributeError:
            pass

        try:
            right_vertex = self.get_right_space_time_vertex()
            if right_vertex.time == 0.0:
                elements.append(self.bottom_space_time_vertex.space_vertex.get_right_element())
        except AttributeError:
            pass

        return elements

    def get_left_space_time_vertex(self):
        for vertex in self.space_time_vertices:
            if (vertex.space_vertex.coordinate <
                    self.bottom_space_time_vertex.space_vertex.coordinate):
                return vertex
        return None

    def get_right_space_time_vertex(self):
        for vertex in self.space_time_vertices:
            if (vertex.space_vertex.coordinate >
                    self.bottom_space_time_vertex.space_vertex.coordinate):
                return vertex
        return None

    def get_space_patch(self):
        return self.bottom_space_time_vertex.space_vertex.patch

    def _get_front_value(self, x, central_space_time_vertex):
        # Input x is in global coordinates of the underlying space patch!
        patch = self.get_space_patch()
        assert x in patch
        if patch.element_left is not None and x in patch.element_left:
            assert patch.element_left.vertex_right == central_space_time_vertex.space_vertex
            assert patch.element_left.vertex_left == self.get_left_space_time_vertex().space_vertex
            x_left = patch.element_left.vertex_left.coordinate
            x_right = patch.element_left.vertex_right.coordinate
            y_left = self.get_left_space_time_vertex().time
            y_right = central_space_time_vertex.time
        else:
            assert patch.element_right is not None and x in patch.element_right
            assert patch.element_right.vertex_left == central_space_time_vertex.space_vertex
            assert (patch.element_right.vertex_right ==
                    self.get_right_space_time_vertex().space_vertex)
            x_left = patch.element_right.vertex_left.coordinate
            x_right = patch.element_right.vertex_right.coordinate
            y_left = central_space_time_vertex.time
            y_right = self.get_right_space_time_vertex().time

        return y_left + (y_right - y_left) * (x - x_left) / (x_right - x_left)

    def _get_front_derivative(self, x, central_space_time_vertex):
        # Input x is in global coordinates of the underlying space patch!
        patch = self.get_space_patch()
        assert x in patch
        if patch.element_left is not None and x in patch.element_left:
            assert patch.element_left.vertex_right == central_space_time_vertex.space_vertex
            assert patch.element_left.vertex_left == self.get_left_space_time_vertex().space_vertex
            x_left = patch.element_left.vertex_left.coordinate
            x_right = patch.element_left.vertex_right.coordinate
            y_left = self.get_left_space_time_vertex().time
            y_right = central_space_time_vertex.time
        else:
            assert patch.element_right is not None and x in patch.element_right
            assert patch.element_right.vertex_left == central_space_time_vertex.space_vertex
            assert (patch.element_right.vertex_right ==
                    self.get_right_space_time_vertex().space_vertex)
            x_left = patch.element_right.vertex_left.coordinate
            x_right = patch.element_right.vertex_right.coordinate
            y_left = central_space_time_vertex.time
            y_right = self.get_right_space_time_vertex().time

        return (y_right - y_left) / (x_right - x_left)

    def get_bottom_front_value(self, x):
        return self._get_front_value(x, self.bottom_space_time_vertex)

    def get_bottom_front_derivative(self, x):
        return self._get_front_derivative(x, self.bottom_space_time_vertex)

    def get_top_front_value(self, x):
        return self._get_front_value(x, self.top_space_time_vertex)

    def get_top_front_derivative(self, x):
        return self._get_front_derivative(x, self.top_space_time_vertex)

    def get_time_transformation(self, x, t_ref):
        # phi_2
        return ((1. - t_ref) * self.get_bottom_front_value(x) +
                t_ref * self.get_top_front_value(x))

    def get_time_transformation_dx(self, x, t_ref):
        # phi_2_dx
        return ((1. - t_ref) * self.get_bottom_front_derivative(x) +
                t_ref * self.get_top_front_derivative(x))

    def get_time_transformation_dt(self, x, t_ref):
        # phi_2_dt
        return -self.get_bottom_front_value(x) + self.get_top_front_value(x)

    def get_inverse_time_transformation(self, x, t):
        # phi_2^{-1}
        if self.get_bottom_front_value(x) == self.get_top_front_value(x):
            return 0.
        return ((t - self.get_bottom_front_value(x))
                / (self.get_top_front_value(x) - self.get_bottom_front_value(x)))


class AdvancingFront:
    def __init__(self, space_grid, t_max, characteristic_speed):
        self.space_grid = space_grid
        self.t_max = t_max
        self.characteristic_speed = characteristic_speed

        self.space_time_vertices = [SpaceTimeVertex(vertex, 0.)
                                    for vertex in space_grid.get_vertices()]
        self.potential_pitch_locations = list(self.space_time_vertices)
        # set(list(self.space_time_vertices))

        for vertex in self.space_time_vertices:
            vertex.potential_tent_height = np.min([element.length *
                                                   self.space_grid.shape_regularity_constant /
                                                   element.get_maximum_speed(characteristic_speed)
                                                   for element
                                                   in vertex.space_vertex.patch.get_elements()])

    def get_feasible_vertex(self):
        if len(self.potential_pitch_locations) > 0:
            return self.potential_pitch_locations[0]
            # return next(iter(self.potential_pitch_locations))
        return None


class SpaceTimeGrid:
    def __init__(self, space_grid, t_max, characteristic_speed, gamma=0.5):
        self.space_grid = space_grid
        self.t_max = t_max
        self.characteristic_speed = characteristic_speed
        assert 0. < gamma < 1.
        self.gamma = gamma

        self.advancing_front = AdvancingFront(self.space_grid, self.t_max,
                                              self.characteristic_speed)
        self.space_time_vertices = list(self.advancing_front.space_time_vertices)
        self.tents = []

    def pitch_tent(self, space_time_vertex):
        assert space_time_vertex in self.advancing_front.potential_pitch_locations
        # Update advancing front
        self.advancing_front.space_time_vertices.remove(space_time_vertex)
        self.advancing_front.potential_pitch_locations.remove(space_time_vertex)
        new_space_time_vertex = SpaceTimeVertex(space_time_vertex.space_vertex,
                                                (space_time_vertex.time +
                                                 space_time_vertex.potential_tent_height))
        self.advancing_front.space_time_vertices.append(new_space_time_vertex)
        self.space_time_vertices.append(new_space_time_vertex)
        # Create tent
        space_time_vertices_of_tent = [space_time_vertex, new_space_time_vertex]
        for vertex in self.advancing_front.space_time_vertices:
            if vertex.space_vertex in space_time_vertex.space_vertex.get_adjacent_vertices():
                space_time_vertices_of_tent.append(vertex)
        tent = SpaceTimeTent(space_time_vertex, new_space_time_vertex,
                             space_time_vertices=space_time_vertices_of_tent,
                             number=len(self.tents))
        # Set neighboring tents properly
        for vertex in space_time_vertices_of_tent:
            if (vertex.tent_below and
                    len(set(vertex.tent_below.space_time_vertices).intersection(
                        space_time_vertices_of_tent)) == 2):
                # Find element that corresponds to the intersection of the tents
                element = None
                for elem in tent.get_space_patch().get_elements():
                    if set(elem.get_vertices()) == set(st_vertex.space_vertex
                                                       for st_vertex
                                                       in set(vertex.tent_below.space_time_vertices)
                                                       .intersection(space_time_vertices_of_tent)):
                        element = elem
                assert element is not None
                vertex.tent_below.neighboring_tents_above.append((tent, element))
                tent.neighboring_tents_below.append((vertex.tent_below, element))
        self.tents.append(tent)
        new_space_time_vertex.tent_below = tent

        # Update potential tent heights
        for vertex in self.advancing_front.space_time_vertices:
            if (vertex.space_vertex in space_time_vertex.space_vertex.get_adjacent_vertices() or
                    vertex.space_vertex == space_time_vertex.space_vertex):
                vertex.potential_tent_height = self.t_max - vertex.time
                for element in vertex.space_vertex.patch.get_elements():
                    other_vertex = None
                    for vertex_af in self.advancing_front.space_time_vertices:
                        if (vertex_af.space_vertex in element.get_vertices() and
                                vertex_af is not vertex):
                            other_vertex = vertex_af
                            break
                    assert other_vertex is not None
                    deriv = (element.length * self.space_grid.shape_regularity_constant /
                             element.get_maximum_speed(self.characteristic_speed))
                    vertex.potential_tent_height = np.min([vertex.potential_tent_height,
                                                           other_vertex.time - vertex.time + deriv])

                tent_height_on_flat_front = np.min([(element.length *
                                                     self.space_grid.shape_regularity_constant /
                                                     element.get_maximum_speed(
                                                         self.characteristic_speed))
                                                    for element
                                                    in vertex.space_vertex.patch.get_elements()])
                if (vertex.potential_tent_height >=
                        np.min([self.gamma * tent_height_on_flat_front,
                                self.t_max - vertex.time])
                        and vertex.potential_tent_height > 0.):
                    if vertex not in self.advancing_front.potential_pitch_locations:
                        self.advancing_front.potential_pitch_locations.append(vertex)
                        # .update([vertex,])
                elif vertex in self.advancing_front.potential_pitch_locations:
                    self.advancing_front.potential_pitch_locations.remove(vertex)

    def check_space_time_vertices_adjacent(self, vertex1, vertex2):
        assert isinstance(vertex1, SpaceTimeVertex)
        assert isinstance(vertex2, SpaceTimeVertex)

        for element in self.space_grid.elements:
            if (vertex1.space_vertex in element.get_vertices() and
                    vertex2.space_vertex in element.get_vertices()):
                return True
        return False
