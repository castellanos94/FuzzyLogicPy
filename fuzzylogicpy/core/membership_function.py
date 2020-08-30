from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    def __init__(self):
        self.type = self.name()

    @abstractmethod
    def evaluate(self, value) -> float:
        pass

    @abstractmethod
    def derive(self, param: str) -> float:
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def repair(self):
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def get_values(self):
        return self.__dict__.values()

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()
