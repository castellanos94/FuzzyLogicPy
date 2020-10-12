from abc import ABC, abstractmethod


class Logic(ABC):
    def __init__(self, natural_imp: bool = True):
        self.natural_imp = natural_imp

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

    def imp_(self, a: float, b: float) -> float:
        if self.natural_imp:
            return self.or_test(self.not_(a), b)
        return self.or_test(self.not_(a), self.and_test(a, b))

    def eqv_(self, a: float, b: float) -> float:
        return self.and_test(self.imp_(a, b), self.imp_(b, a))

    def __str__(self):
        return self.name()
