class NumericalFlux:
    def __init__(self, flux, flux_derivative=None, flux_positive=None, flux_negative=None,
                 lambda_=None):
        self.flux = flux
        self.flux_derivative = flux_derivative
        self.flux_positive = flux_positive
        self.flux_negative = flux_negative
        self.lambda_ = lambda_


class LaxFriedrichsFlux(NumericalFlux):
    def __call__(self, u_1, u_2):
        assert self.lambda_ is not None
        return (self.flux(u_1) + self.flux(u_2)) / 2. + 1. / (2. * self.lambda_) * (u_1 - u_2)


class BackwardDifference(NumericalFlux):
    def __call__(self, u_1, u_2):
        return self.flux(u_1)


class EngquistOsherFlux(NumericalFlux):
    def __call__(self, u_1, u_2):
        assert self.flux_positive is not None and self.flux_negative is not None
        return self.flux_positive(u_1) + self.flux_negative(u_2)
