import numpy as np

from tent_pitching.grids import Patch


class Vertex:
    def __init__(self, coordinates, label=""):
        assert len(coordinates) == 1
        self.coordinates = coordinates
        self.label = label

        self.patch_elements = []
        self.patch = None

    def __str__(self):
        return self.label + f" at {self.coordinates}"

    def get_adjacent_vertices(self):
        adjacent_vertices = []
        for element in self.patch_elements:
            adjacent_vertices.extend(element.get_vertices())

        if self in adjacent_vertices:
            adjacent_vertices.remove(self)

        return list(filter((self).__ne__, adjacent_vertices))

    def init_patch(self):
        self.patch = Patch(self)


class Element:
    def __init__(self, vertex_left, vertex_right, label=""):
        assert vertex_left.coordinates < vertex_right.coordinates
        self.vertex_left = vertex_left
        self.vertex_right = vertex_right
        self.label = label

        self.length = np.linalg.norm(self.vertex_left.coordinates - self.vertex_right.coordinates)

        self.vertex_left.patch_elements.append(self)
        self.vertex_right.patch_elements.append(self)

    def __str__(self):
        return self.label

    def get_vertices(self):
        return [self.vertex_left, self.vertex_right,]

    def get_maximum_speed(self, characteristic_speed):
        return characteristic_speed(self.get_vertices()[0].coordinates) # Do something more elaborate here!


class Grid:
    def __init__(self, elements):
        assert len(elements) > 0
        self.elements = elements

        self.shape_regularity_constant = 1. # compute a reasonable value here!

        for vertex in self.get_vertices():
            vertex.init_patch()

    def get_vertices(self):
        return list(set([vertex for element in self.elements for vertex in element.get_vertices()]))
