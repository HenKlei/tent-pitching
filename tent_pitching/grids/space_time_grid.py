import numpy as np


class SpaceTimeVertex:
    def __init__(self, space_vertex, time):
        self.space_vertex = space_vertex
        self.time = time
        self.potential_tent_height = 0.

        self.coordinates = [self.space_vertex.coordinate, self.time,]

    def __str__(self):
        return self.space_vertex.__str__() + f"; time: {self.time}"


class SpaceTimeTent:
    def __init__(self, bottom_space_time_vertex, top_space_time_vertex, space_time_vertices=None):
        self.bottom_space_time_vertex = bottom_space_time_vertex
        self.top_space_time_vertex = top_space_time_vertex
        assert self.bottom_space_time_vertex.space_vertex == self.top_space_time_vertex.space_vertex
        self.height = top_space_time_vertex.time - bottom_space_time_vertex.time

        self.space_time_vertices = space_time_vertices
        assert len(self.space_time_vertices) <= 4

    def get_left_space_time_vertex(self):
        for vertex in self.space_time_vertices:
            if vertex.space_vertex.coordinate < self.bottom_space_time_vertex.space_vertex.coordinate:
                return vertex
        return None

    def get_right_space_time_vertex(self):
        for vertex in self.space_time_vertices:
            if vertex.space_vertex.coordinate > self.bottom_space_time_vertex.space_vertex.coordinate:
                return vertex
        return None

    def get_space_patch(self):
        return self.bottom_space_time_vertex.space_vertex.patch

    def _get_front(self, central_space_time_vertex):
        def front(x):
            # Input x is in global coordinates of the underlying space patch!
            patch = self.get_space_patch()
            assert x in patch
            if patch.element_left is not None and x in patch.element_left:
                assert patch.element_left.vertex_right == central_space_time_vertex.space_vertex
                assert patch.element_left.vertex_left == self.get_left_space_time_vertex().space_vertex
                xl = patch.element_left.vertex_left.coordinate
                xr = patch.element_left.vertex_right.coordinate
                yl = self.get_left_space_time_vertex().time
                yr = central_space_time_vertex.time
            else:
                assert patch.element_right is not None and x in patch.element_right
                assert patch.element_right.vertex_left == central_space_time_vertex.space_vertex
                assert patch.element_right.vertex_right == self.get_right_space_time_vertex().space_vertex
                xl = patch.element_right.vertex_left.coordinate
                xr = patch.element_right.vertex_right.coordinate
                yl = central_space_time_vertex.time
                yr = self.get_right_space_time_vertex().time

            return yl + (yr - yl) * (x - xl) / (xr - xl)

        return front

    def get_bottom_front(self):
        return self._get_front(self.bottom_space_time_vertex)

    def get_top_front(self):
        return self._get_front(self.top_space_time_vertex)


class AdvancingFront:
    def __init__(self, space_grid, t_max, characteristic_speed):
        self.space_grid = space_grid
        self.t_max = t_max
        self.characteristic_speed = characteristic_speed

        self.space_time_vertices = [SpaceTimeVertex(vertex, 0.) for vertex in space_grid.get_vertices()]
        self.potential_pitch_locations = set(list(self.space_time_vertices))

        for vertex in self.space_time_vertices:
            vertex.potential_tent_height = np.min([element.length * self.space_grid.shape_regularity_constant / element.get_maximum_speed(characteristic_speed) for element in vertex.space_vertex.patch.get_elements()])

    def get_feasible_vertex(self):
        if len(self.potential_pitch_locations) > 0:
            return next(iter(self.potential_pitch_locations))
        return None


class SpaceTimeMesh:
    def __init__(self, space_grid, t_max, characteristic_speed, gamma=0.5):
        self.space_grid = space_grid
        self.t_max = t_max
        self.characteristic_speed = characteristic_speed
        assert 0. < gamma < 1.
        self.gamma = gamma

        self.advancing_front = AdvancingFront(self.space_grid, self.t_max, self.characteristic_speed)
        self.space_time_vertices = list(self.advancing_front.space_time_vertices)
        self.tents = []

    def pitch_tent(self, space_time_vertex):
        assert space_time_vertex in self.advancing_front.potential_pitch_locations
        self.advancing_front.space_time_vertices.remove(space_time_vertex)
        self.advancing_front.potential_pitch_locations.remove(space_time_vertex)
        new_space_time_vertex = SpaceTimeVertex(space_time_vertex.space_vertex, space_time_vertex.time + space_time_vertex.potential_tent_height)
        self.advancing_front.space_time_vertices.append(new_space_time_vertex)
        self.space_time_vertices.append(new_space_time_vertex)
        # create tent
        space_time_vertices_of_tent = [space_time_vertex, new_space_time_vertex]
        for vertex in self.advancing_front.space_time_vertices:
            if vertex.space_vertex in space_time_vertex.space_vertex.get_adjacent_vertices():
                space_time_vertices_of_tent.append(vertex)
        tent = SpaceTimeTent(space_time_vertex, new_space_time_vertex, space_time_vertices=space_time_vertices_of_tent)
        self.tents.append(tent)

        # update potential tent heights
        for vertex in self.advancing_front.space_time_vertices:
            if vertex.space_vertex in space_time_vertex.space_vertex.get_adjacent_vertices() or vertex.space_vertex == space_time_vertex.space_vertex:
                vertex.potential_tent_height = self.t_max - vertex.time
                for element in vertex.space_vertex.patch.get_elements():
                    other_vertex = None
                    for v in self.advancing_front.space_time_vertices:
                        if v.space_vertex in element.get_vertices() and v is not vertex:
                            other_vertex = v
                            break
                    assert other_vertex is not None
                    vertex.potential_tent_height = np.min([vertex.potential_tent_height, other_vertex.time - vertex.time + element.length * self.space_grid.shape_regularity_constant / element.get_maximum_speed(self.characteristic_speed)])

                tent_height_on_flat_front = np.min([element.length * self.space_grid.shape_regularity_constant / element.get_maximum_speed(self.characteristic_speed) for element in vertex.space_vertex.patch.get_elements()])
                if vertex.potential_tent_height >= np.min([self.gamma * tent_height_on_flat_front, self.t_max - vertex.time]) and vertex.potential_tent_height > 0.:
                    self.advancing_front.potential_pitch_locations.update([vertex,])
                elif vertex in self.advancing_front.potential_pitch_locations:
                    self.advancing_front.potential_pitch_locations.remove(vertex)

    def check_space_time_vertices_adjacent(self, v1, v2):
        for element in self.space_grid.elements:
            if v1.space_vertex in element.get_vertices() and v2.space_vertex in element.get_vertices():
                return True
        return False
