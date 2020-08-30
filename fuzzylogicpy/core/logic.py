from abc import ABC, abstractmethod


class Logic(ABC):
    @abstractmethod
    def not_(self, value: float) -> float:
        pass

    @abstractmethod
    def and_test(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def or_test(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def imp_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def eqv_(self, a: float, b: float) -> float:
        pass

    @abstractmethod
    def for_all(self, values) -> float:
        pass

    @abstractmethod
    def exist(self, values) -> float:
        pass

    @abstractmethod
    def and_(self, values) -> float:
        pass

    @abstractmethod
    def or_(self, values) -> float:
        pass

    def name(self):
        return str(self.__class__.__name__)

    def __str__(self):
        return self.name()
