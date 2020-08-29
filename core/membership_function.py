import math
from abc import ABC, abstractmethod


class MembershipFunction(ABC):

    @abstractmethod
    def evaluate(self, value) -> float:
        pass

    @abstractmethod
    def derive(self, param: str) -> float:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def get_values(self):
        return self.__dict__.values()

    def __str__(self):
        return str(self.__dict__)


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
