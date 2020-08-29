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


class ZadehLogic(Logic):
    """
    Ying, Mingsheng. (2002). Implication operators in fuzzy logic. Fuzzy Systems,
    IEEE Transactions on. 10. 88 - 91. 10.1109/91.983282.
    """

    def and_(self, values) -> float:
        return max(values)

    def or_(self, values) -> float:
        return min(values)

    def not_(self, value: float) -> float:
        return 1 - value

    def and_test(self, a: float, b: float) -> float:
        return min(a, b)

    def or_test(self, a: float, b: float) -> float:
        return max(a, b)

    def imp_(self, a: float, b: float) -> float:
        return max(1 - a, min(a, b))

    def eqv_(self, a: float, b: float) -> float:
        return min(self.imp_(a, b), self.imp_(b, a))

    def for_all(self, values) -> float:
        return min(values)

    def exist(self, values) -> float:
        return max(values)
