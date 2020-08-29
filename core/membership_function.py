from abc import ABC, abstractmethod


class MembershipFunction(ABC):

    @abstractmethod
    def evaluate(self, value) -> float:
        pass

    @abstractmethod
    def derive(self, param: str) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def get_values(self):
        return self.__dict__.values()
