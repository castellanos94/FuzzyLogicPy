from abc import ABC, abstractmethod


class Logic(ABC):
    @abstractmethod
    def not(self, value:float) -> float:
        pass
