class TimeStepper:
    def __init__(self, discretization, dt):
        self.discretization = discretization
        self.dt = dt


class ExplicitEuler(TimeStepper):
    def __call__(self, tent, u_n, time):
        return self.discretization.right_hand_side(tent, u_n, time * self.dt)


class RungeKutta4(TimeStepper):
    def __call__(self, tent, u_n, time):
        t = time * self.dt

        k_1 = self.discretization.right_hand_side(tent, u_n, t)
        tmp = [f1 + self.dt / 2. * f2 for f1, f2 in zip(u_n, k_1)]
        k_2 = self.discretization.right_hand_side(tent, tmp, t + self.dt / 2.)
        tmp = [f1 + self.dt / 2. * f2 for f1, f2 in zip(u_n, k_2)]
        k_3 = self.discretization.right_hand_side(tent, tmp, t + self.dt / 2.)
        tmp = [f1 + self.dt * f2 for f1, f2 in zip(u_n, k_3)]
        k_4 = self.discretization.right_hand_side(tent, tmp, t + self.dt)

        return [f1 / 6. + f2 / 3. + f3 / 3. + f4 / 6. for f1, f2, f3, f4 in zip(k_1, k_2, k_3, k_4)]
