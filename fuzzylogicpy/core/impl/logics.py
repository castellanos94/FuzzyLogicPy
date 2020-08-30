import math
from functools import reduce

import numpy as np

from fuzzylogicpy.core.logic import Logic


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


class GMBC(Logic):
    """
    Logica Compensatoria Basada en la Media Geometrica.
    """

    def not_(self, value: float) -> float:
        return 1 - value

    def and_test(self, a: float, b: float) -> float:
        return pow(a * b, 0.5)

    def or_test(self, a: float, b: float) -> float:
        return 1 - pow((1 - a) * (1 - b), 0.5)

    def imp_(self, a: float, b: float) -> float:
        neg = 1 - a
        return 1 - pow((1 - neg) * (1 - b), 0.5)

    def eqv_(self, a: float, b: float) -> float:
        return pow(self.imp_(a, b) * self.imp_(b, a), 0.5)

    def for_all(self, values) -> float:
        return pow(math.e, (1 / len(values)) * sum([np.log(v) for v in values if v != 0]))

    def exist(self, values) -> float:
        return 1 - pow(math.e, ((1 / len(values)) * sum([np.log(1 - v) for v in values if v != 0 and v != 1])))

    def and_(self, values) -> float:
        return pow(reduce(lambda x, y: x * y, values), 1 / len(values))

    def or_(self, values) -> float:
        return 1 - pow(reduce(lambda x, y: (1 - x) * (1 - y), values), 1 / len(values))
