import numpy as np
from itertools import tee

from tent_pitching.grids import Vertex, Element, Grid


class Line:
    def __init__(self):
        self.dim = 1
        self.vertices = [Vertex(np.array([0.,])), Vertex(np.array([1.,]))]
        self.elements = [Element(*self.vertices),]
        self.volume = 1.

    def get_uniform_refinement(self, h):
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)
        sub_vertices = [Vertex(pos) for pos in np.linspace(0., 1., num=1.//h)]
        sub_elements = [Element(v0, v1) for v0, v1 in pairwise(sub_vertices)]
        refinement = Grid(sub_elements)
        return refinement
