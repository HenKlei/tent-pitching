class LaxFriedrichsFlux:
    def __init__(self, flux, lambda_):
        self.flux = flux
        self.lambda_ = lambda_

    def __call__(self, u_1, u_2):
        return (self.flux(u_1) + self.flux(u_2)) / 2. + 1. / (2. * self.lambda_) * (u_1 - u_2)


class BackwardDifference:
    def __init__(self, flux):
        self.flux = flux

    def __call__(self, u_1, u_2):
        return self.flux(u_1)
