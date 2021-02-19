class Patch:
    def __init__(self, vertex):
        self.central_vertex = vertex
        self.element_left = self.central_vertex.get_left_element()
        self.element_right = self.central_vertex.get_right_element()

        assert not (self.element_left is None and self.element_right is None)

    def __contains__(self, x):
        if self.element_left is None:
            return x in self.element_right
        if self.element_right is None:
            return x in self.element_left
        return x in self.element_right or x in self.element_left

    def get_elements(self):
        return [elem for elem in [self.element_left, self.element_right, ] if elem is not None]

    def is_boundary_patch(self):
        return len(self.get_elements()) == 1
