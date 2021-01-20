import numpy as np


class Vertex:
    def __init__(self, coordinates, label=""):
        self.coordinates = coordinates
        self.dim = len(self.coordinates)
        self.label = label

        self.patch_elements = []
        self.incident_edges = []

    def __str__(self):
        return self.label + f" at {self.coordinates}"

    def get_adjacent_vertices(self):
        adjacent_vertices = []
        for edge in self.incident_edges:
            adjacent_vertices.extend(edge.get_vertices())

        if self in adjacent_vertices:
            adjacent_vertices.remove(self)

        return list(filter((self).__ne__, adjacent_vertices))


class SpaceTimeVertex:
    def __init__(self, space_vertex, time):
        self.space_vertex = space_vertex
        self.time = time
        self.potential_tent_height = 0.

        self.coordinates = np.concatenate((self.space_vertex.coordinates, [self.time]))

    def __str__(self):
        return self.space_vertex.__str__() + f"; time: {self.time}"


class Edge:
    def __init__(self, vertex1, vertex2, label=""):
        assert vertex1.dim == vertex2.dim
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.label = label

        self.length = np.linalg.norm(vertex1.coordinates - vertex2.coordinates)

        self.vertex1.incident_edges.append(self)
        self.vertex2.incident_edges.append(self)

        self.incident_elements = []

    def __str__(self):
        return self.label + f" between ({self.vertex1}) and ({self.vertex2}); length: {self.length}"

    def get_vertices(self):
        return [self.vertex1, self.vertex2]

    def get_maximum_speed_on_incident_elements(self, characteristic_speed):
        speed = 0.
        for element in self.incident_elements:
            speed = np.max([speed, element.get_maximum_speed(characteristic_speed)])

        return speed


class Element:
    def __init__(self, edges, local_subgrid=None, label=""):
        self.edges = edges
        for edge in self.edges:
            edge.incident_elements.append(self)
        self.local_subgrid = local_subgrid
        self.label = label

        for vertex in self.get_vertices():
            vertex.patch_elements.append(self)

        assert len(self.get_vertices()) > 0
        assert len(set([vertex.dim for vertex in self.get_vertices()])) == 1

        self.dim = self.get_vertices()[0].dim

    def __str__(self):
        return self.label + f" with {len(self.edges)} edges"

    def get_vertices(self):
        return [vertex for edge in self.edges for vertex in edge.get_vertices()]

    def get_edges(self):
        return self.edges

    def get_maximum_speed(self, characteristic_speed):
        return characteristic_speed(self.get_vertices()[0].coordinates) # Do something more elaborate here!

    def uniform_subgrid(self, h):
        # !!!! Construct a uniform subgrid of the element !!!!
        pass


class Grid:
    def __init__(self, elements):
        assert len(elements) > 0
        assert len(set([element.dim for element in elements])) == 1
        self.elements = elements
        self.dim = self.elements[0].dim

        self.shape_regularity_constant = 1. # compute a reasonable value here!

    def get_vertices(self):
        return set([vertex for element in self.elements for vertex in element.get_vertices()])

    def get_edges(self):
        return set([edge for element in self.elements for edge in element.get_edges()])


class SpaceTimeTent:
    def __init__(self, base_space_time_vertex, top_space_time_vertex, space_time_vertices=None):
        self.base_space_time_vertex = base_space_time_vertex
        self.top_space_time_vertex = top_space_time_vertex
        self.height = top_space_time_vertex.time - base_space_time_vertex.time

        self.space_time_vertices = space_time_vertices

        def get_space_patch_elements(self):
            return self.base_space_time_vertex.space_vertex.patch_elements


class AdvancingFront:
    def __init__(self, space_grid, t_max, characteristic_speed):
        self.space_grid = space_grid
        self.t_max = t_max
        self.characteristic_speed = characteristic_speed

        self.space_time_vertices = [SpaceTimeVertex(vertex, 0.) for vertex in space_grid.get_vertices()]
        self.potential_pitch_locations = set(list(self.space_time_vertices))

        for vertex in self.space_time_vertices:
            vertex.potential_tent_height = np.min([edge.length * self.space_grid.shape_regularity_constant / edge.get_maximum_speed_on_incident_elements(characteristic_speed) for edge in vertex.space_vertex.incident_edges])

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
                for edge in vertex.space_vertex.incident_edges:
                    other_vertex = None
                    for v in self.advancing_front.space_time_vertices:
                        if v.space_vertex in edge.get_vertices() and v is not vertex:
                            other_vertex = v
                            break
                    assert other_vertex is not None
                    vertex.potential_tent_height = np.min([vertex.potential_tent_height, other_vertex.time - vertex.time + edge.length * self.space_grid.shape_regularity_constant / edge.get_maximum_speed_on_incident_elements(self.characteristic_speed)])

                tent_height_on_flat_front = np.min([edge.length * self.space_grid.shape_regularity_constant / edge.get_maximum_speed_on_incident_elements(self.characteristic_speed) for edge in vertex.space_vertex.incident_edges])
                if vertex.potential_tent_height >= np.min([self.gamma * tent_height_on_flat_front, self.t_max - vertex.time]) and vertex.potential_tent_height > 0.:
                    self.advancing_front.potential_pitch_locations.update([vertex,])
                elif vertex in self.advancing_front.potential_pitch_locations:
                    self.advancing_front.potential_pitch_locations.remove(vertex)

    def check_space_time_vertices_adjacent(self, v1, v2):
        for edge in self.space_grid.get_edges():
            if v1.space_vertex in edge.get_vertices() and v2.space_vertex in edge.get_vertices():
                return True
        return False
