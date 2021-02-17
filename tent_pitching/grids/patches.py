


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

    def to_local(self, x):
        assert x in self

        if self.element_right is None:
            return self.element_left.to_local(x)
        if self.element_left is None:
            return self.element_right.to_local(x)

        if x in self.element_right:
            return .5 + self.element_right.to_local(x) / 2.
        return self.element_left.to_local(x) / 2.

    def to_global(self, x_ref):
        assert 0. <= x_ref <= 1.

        if self.element_right is None:
            return self.element_left.to_global(x_ref)
        if self.element_left is None:
            return self.element_right.to_global(x_ref)

        if x_ref >= .5:
            return self.element_right.to_global((x_ref - .5) * 2.)
        return self.element_left.to_global(2. * x_ref)

    def to_global_dx(self, x_ref):
        assert 0. <= x_ref <= 1.

        if self.element_right is None:
            return self.element_left.to_global_dx(x_ref)
        if self.element_left is None:
            return self.element_right.to_global_dx(x_ref)

        if x_ref >= .5:
            return self.element_right.to_global_dx((x_ref - .5) * 2.)
        return self.element_left.to_global(2. * x_ref)

    def get_elements(self):
        return [elem for elem in [self.element_left, self.element_right,] if elem is not None]

    def is_boundary_patch(self):
        return len(self.get_elements()) == 1
