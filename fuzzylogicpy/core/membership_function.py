from __future__ import annotations

from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    def __init__(self):
        self.type = self.name()

    @abstractmethod
    def evaluate(self, v) -> float:
        pass

    @abstractmethod
    def derive(self, v: float, param: str) -> float:
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def set_values(self, *args):
        pass

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def to_edn(self) -> str:
        pass
