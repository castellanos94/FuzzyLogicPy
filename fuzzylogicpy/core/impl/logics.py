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
        return min(values)

    def or_(self, values) -> float:
        return max(values)

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

    def for_all(self, values) -> float:
        _exponent = (1 / len(values)) * sum([np.log(v) for v in values if v != 0])
        if abs(_exponent) > 0:
            return pow(math.e, _exponent)
        return 0

    def exist(self, values) -> float:
        return 1 - pow(math.e, ((1 / len(values)) * sum([np.log(1 - v) for v in values if v != 1 and v != 0])))

    def and_(self, values) -> float:
        return pow(reduce(lambda x, y: x * y, values), 1 / len(values))

    def or_(self, values) -> float:
        return 1 - pow(reduce(lambda x, y: (1 - x) * (1 - y), values), 1 / len(values))


class AMBC(Logic):
    """
    AMBC logic: https://doi.org/10.1142/S1469026811003070 -based implementation.
    """

    def not_(self, value: float) -> float:
        return 1 - value

    def and_test(self, a: float, b: float) -> float:
        return np.sqrt(min(a, b) * 0.5 * (a + b))

    def or_test(self, a: float, b: float) -> float:
        return 1.0 - np.sqrt(min((1.0 - a), (1.0 - b)) * 0.5 * ((1.0 - a) + (1.0 - b)))

    def for_all(self, values) -> float:
        return np.sqrt(min(values) * sum(values) * 1.0 / len(values))

    def exist(self, values) -> float:
        _values = [1 - v for v in values]
        return 1 - np.sqrt(min(_values) * sum(_values) * 1.0 / len(_values))

    def and_(self, values) -> float:
        return np.sqrt(min(values) * sum(values) * 1.0 / len(values))

    def or_(self, values) -> float:
        _values = [1 - v for v in values]
        return 1 - np.sqrt(min(_values) * (1.0 / len(values)) * sum(_values))

class ACFL(Logic):
    def _f(self, value:float, exp, base) -> float:
        valor = np.log(value) / np.log(base)
        valor = -1 * np.pow(valor, exp)
        return valor

    def _fInv(self,value, base, exp) -> float:
        valor = 1.0/(exp*1.0)
        valor =  pow(abs(value),valor)
        if (value < 0):
            valor *= -1.0
        valor = np.exp(-1* np.log(base)*valor)
        return valor
    
    def and_(self, values,base, exp) -> float:
        Ct = sum([self._f(v,base,exp) for v in values])
        return self._fInv(Ct/len(values),base,exp)

    def imp_(self, ant, cons, base, exp) -> float:
        return 1 - self._fInv(self._f(1-ant,base,exp)+ self._f(cons,base,exp),base, exp)
    
    def eqv_(self, A, B, base, exp) -> float:
        V1 = self.imp_(A,B, base, exp)
        V2 = self.imp_(B,A, base, exp)
        return self._fInv(self._f(V1,base,exp)+self._f(V2,base, exp), base, exp)
    
    def for_all(self, values) -> float:
        pass

    def exist(self, values) -> float:
        pass