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


class Edge:
    def __init__(self, vertex0, vertex1, label=""):
        assert vertex0.dim == vertex1.dim
        self.vertex0 = vertex0
        self.vertex1 = vertex1
        self.label = label

        self.length = np.linalg.norm(vertex0.coordinates - vertex1.coordinates)

        self.vertex0.incident_edges.append(self)
        self.vertex1.incident_edges.append(self)

        self.incident_elements = []

    def __str__(self):
        return self.label + f" between ({self.vertex0}) and ({self.vertex1}); length: {self.length}"

    def get_vertices(self):
        return [self.vertex0, self.vertex1]

    def get_maximum_speed_on_incident_elements(self, characteristic_speed):
        speed = 0.
        for element in self.incident_elements:
            speed = np.max([speed, element.get_maximum_speed(characteristic_speed)])

        return speed


class Element:
    def __init__(self, edges, label=""):
        self.edges = edges
        for edge in self.edges:
            edge.incident_elements.append(self)
        self.label = label

        for vertex in self.get_vertices():
            vertex.patch_elements.append(self)

        assert len(self.get_vertices()) > 0
        assert len(set([vertex.dim for vertex in self.get_vertices()])) == 1

        self.dim = self.get_vertices()[0].dim

    def __str__(self):
        return self.label + f" with {len(self.edges)} edges"

    def get_vertices(self):
        return list(set([vertex for edge in self.edges for vertex in edge.get_vertices()]))

    def get_edges(self):
        return self.edges

    def get_maximum_speed(self, characteristic_speed):
        return characteristic_speed(self.get_vertices()[0].coordinates) # Do something more elaborate here!


class Grid:
    def __init__(self, elements):
        assert len(elements) > 0
        assert len(set([element.dim for element in elements])) == 1
        self.elements = elements
        self.dim = self.elements[0].dim

        self.shape_regularity_constant = 1. # compute a reasonable value here!

    def get_vertices(self):
        return list(set([vertex for element in self.elements for vertex in element.get_vertices()]))

    def get_edges(self):
        return list(set([edge for element in self.elements for edge in element.get_edges()]))
