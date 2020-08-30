import math

from core.membership_function import MembershipFunction


class Sigmoid(MembershipFunction):
    def __init__(self, center: float, beta: float):
        self.type = self.name()
        self.center = center
        self.beta = beta

    def evaluate(self, value) -> float:
        return (1 / (1 + (
            math.exp(-((math.log(0.99) - math.log(0.01)) / (self.center - self.beta)) * (value - self.center)))))

    def derive(self, param: str) -> float:
        pass
