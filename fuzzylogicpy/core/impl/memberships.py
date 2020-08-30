import math

from fuzzylogicpy.core.membership_function import MembershipFunction


class Sigmoid(MembershipFunction):
    def __init__(self, center: float, beta: float):
        super().__init__()
        self.center = center
        self.beta = beta

    def evaluate(self, value) -> float:
        return (1 / (1 + (
            math.exp(-((math.log(0.99) - math.log(0.01)) / (self.center - self.beta)) * (value - self.center)))))

    def derive(self, param: str) -> float:
        pass


class FPG(MembershipFunction):
    def __init__(self, beta: float, gamma: float, m: float):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.m = m

    def evaluate(self, value) -> float:
        sigmoid = pow(Sigmoid(self.gamma, self.beta).evaluate(value), self.m)
        sigmoid2 = pow(1 - Sigmoid(self.gamma, self.beta).evaluate(value), 1 - m)
        m_ = pow(self.m, self.m) * pow((1 - self.m), (1 - self.m))
        return (sigmoid * sigmoid2) / m_

    def derive(self, param: str) -> float:
        pass
