class NumericalFlux:
    def __init__(self, flux, flux_derivative=None, lambda_=None):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.lambda_ = lambda_


class LaxFriedrichsFlux(NumericalFlux):
    def __call__(self, u_1, u_2):
        assert self.lambda_ is not None
        return (self.flux(u_1) + self.flux(u_2)) / 2. + 1. / (2. * self.lambda_) * (u_1 - u_2)


class BackwardDifference(NumericalFlux):
    def __call__(self, u_1, u_2):
        return self.flux(u_1)
