from abc import ABC, abstractmethod


class Logic(ABC):
    @abstractmethod
    def not_(self, value: float) -> float:
        pass

    @abstractmethod
    def and_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def or_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def imp_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def eqv_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def for_all(self, values: List[float]) -> float:
        pass

    @abstractmethod
    def exist(self, values: List[float]) -> float:
        pass

    @abstractmethod
    def and_(self, values: List[float]) -> float:
        pass

    @abstractmethod
    def or_(self, values: List[float]) -> float:
        pass
