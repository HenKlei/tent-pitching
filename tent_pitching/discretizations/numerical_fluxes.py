

class LaxFriedrichsFlux:
    def __init__(self, flux, lambda_):
        self.flux = flux
        self.lambda_ = lambda_

    def __call__(self, u_1, u_2, n):
        return (self.flux(u_1) + self.flux(u_2)) / 2. * n + 1. / (2. * self.lambda_) * (u_1 - u_2)
