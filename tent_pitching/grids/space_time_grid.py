import numpy as np

from tent_pitching.geometry.reference_elements import Vertex, Line, Triangle, Quadrilateral


class SpaceTimeVertex:
    def __init__(self, space_vertex, time):
        self.space_vertex = space_vertex
        self.time = time
        self.potential_tent_height = 0.

        self.coordinates = np.concatenate((self.space_vertex.coordinate, np.array([self.time, ])),
                                          axis=None)

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
        assert 3 <= len(self.space_time_vertices) <= 4

        self.number = number

        if len(self.space_time_vertices) == 3:
            vertices = [Vertex(self.bottom_space_time_vertex.coordinates), None, None]
            lines = [None, None, None]
            orientation = None
            for v in self.space_time_vertices:
                if (v.space_vertex.coordinate >
                   self.bottom_space_time_vertex.space_vertex.coordinate):
                    vertices[1] = Vertex(v.coordinates)
                    vertices[2] = Vertex(self.top_space_time_vertex.coordinates)
                    vertex = v
                    orientation = 'right'
                elif (v.space_vertex.coordinate <
                      self.bottom_space_time_vertex.space_vertex.coordinate):
                    vertices[1] = Vertex(self.top_space_time_vertex.coordinates)
                    vertices[2] = Vertex(v.coordinates)
                    vertex = v
                    orientation = 'left'

            assert orientation is not None
            if orientation == 'right':
                lines[0] = Line([vertices[0], vertices[1]], inside=self,
                                outside=vertex.tent_below, inflow=True)
                lines[1] = Line([vertices[1], vertices[2]], inside=self)
                if (self.bottom_space_time_vertex.space_vertex.is_boundary_vertex()
                   and self.top_space_time_vertex.space_vertex.is_boundary_vertex()):
                    lines[2] = Line([vertices[2], vertices[0]], inside=self,
                                    outside=None, inflow=True)
                else:
                    lines[2] = Line([vertices[2], vertices[0]], inside=self)
            elif orientation == 'left':
                lines[0] = Line([vertices[0], vertices[1]], inside=self)
                lines[1] = Line([vertices[1], vertices[2]], inside=self)
                lines[2] = Line([vertices[2], vertices[0]], inside=self,
                                outside=vertex.tent_below, inflow=True)
            self.element = Triangle(lines)
        elif len(self.space_time_vertices) == 4:
            vertices = [Vertex(self.bottom_space_time_vertex.coordinates), None,
                        Vertex(self.top_space_time_vertex.coordinates), None]
            lines = [None, None, None, None]
            for v in self.space_time_vertices:
                if (v.space_vertex.coordinate >
                   self.bottom_space_time_vertex.space_vertex.coordinate):
                    vertices[1] = Vertex(v.coordinates)
                    right_vertex = v
                elif (v.space_vertex.coordinate <
                      self.bottom_space_time_vertex.space_vertex.coordinate):
                    vertices[3] = Vertex(v.coordinates)
                    left_vertex = v
            lines[0] = Line([vertices[0], vertices[1]], inside=self,
                            outside=right_vertex.tent_below, inflow=True)
            lines[1] = Line([vertices[1], vertices[2]], inside=self)
            lines[2] = Line([vertices[2], vertices[3]], inside=self)
            lines[3] = Line([vertices[3], vertices[0]], inside=self,
                            outside=left_vertex.tent_below, inflow=True)
            self.element = Quadrilateral(lines)

    def __str__(self):
        if self.number is not None:
            return (f"Tent number {self.number} pitched above of "
                    f"{self.bottom_space_time_vertex.space_vertex}")
        return f"Tent without number pitched above of {self.bottom_space_time_vertex.space_vertex}"

    def __contains__(self, point):
        x = point[0]
        t = point[1]
        if (x in self.get_space_patch()
           and (self.get_bottom_front_value(x) <= t <= self.get_top_front_value(x)
                or np.isclose(self.get_bottom_front_value(x), t)
                or np.isclose(self.get_top_front_value(x), t))):
            return True
        return False

    def inflow_tents(self):
        return [tent[0] for tent in self.neighboring_tents_below]

    def inflow_faces(self):
        return [f for f in self.element.get_subentities(codim=1) if f.inflow]

    def outflow_faces(self):
        return [f for f in self.element.get_subentities(codim=1) if not f.inflow]

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
            if np.isclose(self.potential_pitch_locations[0].potential_tent_height, 0.):
                self.potential_pitch_locations.pop(0)
                return self.get_feasible_vertex()
            else:
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
        self.dim = self.space_grid.dim + 1

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
