import itertools
import numpy as np

from tent_pitching.grids import Patch


class Vertex:
    def __init__(self, coordinate, label=""):
        self.coordinate = coordinate
        self.label = label

        self.patch_elements = []
        self.patch = None

    def __str__(self):
        return self.label + f" at {self.coordinate:.3f}"

    def get_adjacent_vertices(self):
        adjacent_vertices = []
        for element in self.patch_elements:
            adjacent_vertices.extend(element.get_vertices())

        if self in adjacent_vertices:
            adjacent_vertices.remove(self)

        return list(filter((self).__ne__, adjacent_vertices))

    def init_patch(self):
        assert len(self.patch_elements) <= 2
        self.patch = Patch(self)

    def get_left_element(self):
        for element in self.patch_elements:
            if element.vertex_left.coordinate < self.coordinate:
                return element
        assert len(self.patch_elements) == 1
        return None

    def get_right_element(self):
        for element in self.patch_elements:
            if element.vertex_right.coordinate > self.coordinate:
                return element
        assert len(self.patch_elements) == 1
        return None

    def is_boundary_vertex(self):
        return self.get_right_element() is None or self.get_left_element() is None


class Element:
    def __init__(self, vertex_left, vertex_right, label=""):
        assert vertex_left.coordinate < vertex_right.coordinate
        self.vertex_left = vertex_left
        self.vertex_right = vertex_right
        self.label = label

        self.length = self.vertex_right.coordinate - self.vertex_left.coordinate

        self.vertex_left.patch_elements.append(self)
        self.vertex_right.patch_elements.append(self)

    def __str__(self):
        return self.label

    def __contains__(self, x):
        return self.vertex_left.coordinate <= x <= self.vertex_right.coordinate

    def get_vertices(self):
        return [self.vertex_left, self.vertex_right, ]

    def get_maximum_speed(self, characteristic_speed):
        # Do something more elaborate here!
        return max(characteristic_speed(self.get_vertices()[0].coordinate),
                   characteristic_speed(self.get_vertices()[1].coordinate))


class Grid:
    def __init__(self, elements):
        assert len(elements) > 0
        self.elements = elements

        self.shape_regularity_constant = 1.  # compute a reasonable value here!

        for vertex in self.get_vertices():
            vertex.init_patch()

    def get_vertices(self):
        seen = set()
        seen_add = seen.add
        return [vertex for element in self.elements
                for vertex in element.get_vertices() if not (vertex in seen or seen_add(vertex))]
        # If we change back to sets, use this instead:
        # return list(set([vertex for element in self.elements
        #                  for vertex in element.get_vertices()]))

    def get_space_bounds(self):
        min_coordinate = None
        max_coordinate = None

        for vertex in self.get_vertices():
            if min_coordinate is None or vertex.coordinate < min_coordinate:
                min_coordinate = vertex.coordinate
            if max_coordinate is None or vertex.coordinate > max_coordinate:
                max_coordinate = vertex.coordinate
        return min_coordinate, max_coordinate


def create_uniform_grid(global_space_grid_size, left=0., right=1.):
    num_vertices = int((right - left) / global_space_grid_size) + 1
    diff = (right - left) / (num_vertices - 1.)

    vertices = []
    for i in range(num_vertices):
        vertices.append(Vertex(np.array(left + i * diff), label=f"Vertex {i}"))

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        first, second = itertools.tee(iterable)
        next(second, None)
        return zip(first, second)

    elements = []
    for i, tmp in enumerate(pairwise(vertices)):
        elements.append(Element(tmp[0], tmp[1], label=f"Element {i}"))

    return Grid(elements)
