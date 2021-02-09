


class DiscontinuousGalerkin:
    def __init__(self, flux, flux_derivative):
        self.flux = flux
        self.flux_derivative = flux_derivative

    def right_hand_side(self, tent, local_solution):
        raise NotImplementedError
