from __future__ import annotations

from typing import List, Tuple

import numpy as np

from fuzzylogicpy.core.membership_function import MembershipFunction

_nan = float('nan')


class Sigmoid(MembershipFunction):
    def to_edn(self) -> str:
        return '[sigmoid {} {}]'.format(self.center, self.beta)

    def __init__(self, center: float, beta: float):
        super(Sigmoid, self).__init__()
        self.center = center
        self.beta = beta

    def get_values(self):
        return [self.center, self.beta]

    def set_values(self, values: List):
        self.center, self.beta = values[:2]

    def is_valid(self) -> bool:
        return self.center > self.beta

    def evaluate(self, value) -> float:
        return 1 / (
                1
                + (
                    np.exp(
                        -((np.log(0.99) - np.log(0.01)) / (self.center - self.beta))
                        * (value - self.center)
                    )
                )
        )

    def derive(self, value: float, param: str) -> float:

        if param == "b":
            #       ce^(c(b+x))
            #  - _________________
            #    (e^(bc) + e^(cx))^2
            result = (
                    -1
                    * (self.center * np.exp(self.center * (self.beta + value)))
                    / np.power(
                (np.exp(self.beta * self.center) + np.exp(self.center * value)), 2
            )
            )
        elif param == "c":
            #      b - x * e^(c(x-b))
            #  - _______________________
            #    ( e^(c * (x-b)) + 1 )^2
            result = (
                    -1
                    * ((self.beta - value) * np.exp(self.center * (value - self.beta)))
                    / np.power((np.exp(self.center * (value - self.beta))) + 1, 2)
            )
        return result


class FPG(MembershipFunction):
    def to_edn(self) -> str:
        return '[FPG {} {} {}]'.format(self.gamma, self.beta, self.m)

    def get_values(self):
        return [self.beta, self.gamma, self.m]

    def set_values(self, values: List):
        self.beta, self.gamma, self.m = values[0], values[1], values[2]

    def is_valid(self) -> bool:
        return self.beta < self.gamma and 0 <= self.m <= 1

    def __init__(self, beta: float, gamma: float = None, m: float = None):
        super(FPG, self).__init__()
        if isinstance(beta, List):
            self.beta, self.gamma, self.m = beta[:3]
        else:
            self.beta = beta
            self.gamma = gamma
            self.m = m

    def evaluate(self, value) -> float:
        sigmoid = pow(Sigmoid(self.gamma, self.beta).evaluate(value), self.m)
        sigmoid2 = pow(1 - Sigmoid(self.gamma, self.beta).evaluate(value), 1 - self.m)
        m_ = np.float(self.m) ** np.float(self.m) * np.float(1 - self.m) ** np.float(1 - self.m)
        return (sigmoid * sigmoid2) / m_

    def derive(self, value: float, param: string) -> float:
        gamma, beta, m = self.get_values()
        if param == 'gamma':
            #       e^(g(-b+x)) * (1-m)^(m-1) * m^-m * ( m* (1 / 1 + e^(g(b-x) )^(m-1) - 1) * (b-x)
            #  - _______________________________________________________________________
            #                              (1 + e^(g(-b + x))^2
            a = m * (1/(1+np.exp(gamma *(beta-value))))**(m-1)
            a -= 1
            a *= np.float(1-m)**np.float(m-1)
            a *= np.float(m)**np.float(-m)
            a *= np.exp(gamma*(value-beta))
            a *= (beta-value)
            b = (1 + np.exp(gamma*(value-beta)))**2
            result = -a/b 
            
           
        elif param == 'beta':
            #    ge^(g(x-b)) * (1-m)^(m-1) * m^-m * (m*( 1/ 1+e^(g(b-x)) )^(m-1) -1)
            #  - ________________________________________________________________
            #                       (1 + e^(c(x-b)))2
            a = m * (1 / (1 + np.exp(gamma * (beta - value)))) ** (m - 1)
            a -= 1
            a *= np.float(1 - m) ** np.float(m - 1)
            a *= np.float(m) ** np.float(-m)
            a *= gamma * np.exp(gamma * (value - beta))
            b = (1 + np.exp(gamma * (value - beta))) ** 2
            result = -a / b

        # elif param == 'm':
        #    Sg = Sigmoid(gamma, beta).evaluate(value)
        #    #  (1-m)^(-1+m) * m^-m * (1 -Sg)^(1-m) * Sg**m * (log(1-m) - log(m) - log(1-Sg) + log(Sg))
        #    a = (1 - m) ** (-1 + m)
        #    a *= np.float(m) ** np.float(-m)
        #    a *= (1 - Sg) ** (1 - m)
        #    a *= Sg ** np.float(m)
        #    a *= (np.log(1 - m) - np.log(m) - np.log(1 - Sg) + np.log(Sg))
        #    result = a

        return result


class Gamma(MembershipFunction):
    def to_edn(self) -> str:
        return '[gamma {} {}]'.format(self.a, self.b)

    def __init__(self, a: float, b: float):
        super(Gamma, self).__init__()
        self.a = a
        self.b = b

    def evaluate(self, v) -> float:
        if v <= self.a:
            return 0
        return 1 - np.exp(-self.b * pow(v - self.a, 2))

    def derive(self, value: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        return self.a > self.b

    def get_values(self):
        return [self.a, self.b]

    def set_values(self, *args):
        self.a, self.b = args[:2]


class Gaussian(MembershipFunction):
    def to_edn(self) -> str:
        return '[gaussian {} {}]'.format(self.center, self.deviation)

    def __init__(self, center: float, deviation: float):
        super(Gaussian, self).__init__()
        self.center = center
        self.deviation = deviation

    def evaluate(self, v) -> float:
        return np.exp(-pow(v - self.center, 2) / (2 * pow(self.deviation, 2)))

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        return self.center > self.deviation

    def get_values(self):
        return [self.center, self.deviation]

    def set_values(self, *args):
        self.center, self.deviation = args[:2]


class GBell(MembershipFunction):
    def __init__(self, width: float, slope: float, center: float):
        super(GBell, self).__init__()
        self.width = width
        self.slope = slope
        self.center = center

    def evaluate(self, v) -> float:
        return 1 / (1 + pow(abs(v - self.center) / self.width, 2 * self.slope))

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        pass

    def get_values(self):
        return [self.width, self.slope, self.center]

    def set_values(self, *args):
        self.width, self.slope, self.center = args[:3]


class Triangular(Gamma):
    def to_edn(self) -> str:
        return '[triangular {} {} {}]'.format(self.a, self.b, self.c)

    def __init__(self, a: float, b: float, c: float):
        super(Triangular, self).__init__(a, b)
        self.c = c

    def evaluate(self, v) -> float:
        low_a = self.b - self.a
        low_b = self.c - self.b
        if v <= self.a:
            return 0
        elif self.a <= v <= self.b:
            return (v - self.a) / low_a if low_a != 0 else _nan
        elif self.b <= v <= self.c:
            return (self.c - v) / low_b if low_b != 0 else _nan
        return 0

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        pass

    def get_values(self):
        return [self.a, self.b, self.c]

    def set_values(self, *args):
        self.a, self.b, self.c = args[:3]


class Trapezoidal(Triangular):
    def to_edn(self) -> str:
        return '[trapezoidal {} {} {} {}]'.format(self.a, self.b, self.c, self.d)

    def __init__(self, a: float, b: float, c: float, d: float):
        super(Trapezoidal, self).__init__(a, b, c)
        self.d = d

    def evaluate(self, v) -> float:
        if v < self.a:
            return 0
        elif self.a <= v <= self.b:
            return (v - self.a) / (self.b - self.a) if (self.b - self.a) != 0 else _nan
        elif self.b <= v < self.c:
            return 1
        elif self.c <= v < self.d:
            return 1 - (v - self.c) / (self.d - self.c)
        return 0

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        pass

    def get_values(self):
        return [self.a, self.b, self.c, self.d]

    def set_values(self, *args):
        self.a, self.b, self.c, self.d = args[:4]


class LGamma(Gamma):
    def to_edn(self) -> str:
        return '[Lgamma {} {}]'.format(self.a, self.b)

    def evaluate(self, v) -> float:
        if v <= self.a:
            return 0
        return self.b * pow(v - self.a, 2) / (1 + self.b * pow(v - self.a, 2))

    def derive(self, value: float, param: str) -> float:
        pass


class LTrapezoidal(Gamma):
    def to_edn(self) -> str:
        return '[Ltrapezoidal {} {}]'.format(self.a, self.b)

    def evaluate(self, v) -> float:
        if v < self.a:
            return 0
        if self.a <= v <= self.b:
            return (v - self.a) / (self.b - self.a) if (self.b - self.a) != 0 else _nan
        return 1

    def is_valid(self) -> bool:
        pass

    def derive(self, value: float, param: str) -> float:
        pass


class RTrapezoidal(Gamma):
    def to_edn(self) -> str:
        return '[Rtrapezoidal {} {}]'.format(self.a, self.b)

    def evaluate(self, v) -> float:
        if v < self.a:
            return 1
        if self.a <= v <= self.b:
            return 1 - (v - self.a) / (self.b - self.a) if (self.b - self.a) != 0 else _nan
        return 0

    def is_valid(self) -> bool:
        pass

    def derive(self, value: float, param: str) -> float:
        pass


class Singleton(MembershipFunction):
    def to_edn(self) -> str:
        return '[singleton {}]'.format(self.a)

    def __init__(self, a: float):
        super(Singleton, self).__init__()
        self.a = a

    def evaluate(self, v) -> float:
        return 1 if v == self.a else 0

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        return True

    def get_values(self):
        return [self.a]

    def set_values(self, *args):
        self.a = args[:1]


class Nominal(MembershipFunction):
    def __init__(self, param: Tuple[str, float], not_found_value: float = 0):
        super(Nominal, self).__init__()
        self.param = param
        self.not_found_value = not_found_value

    def evaluate(self, v) -> float:
        return 1 if v == self.param[0] else self.not_found_value

    def derive(self, v: float, param: str) -> float:
        pass

    def is_valid(self) -> bool:
        return 0 <= self.param[1] <= 1 and 0 <= self.not_found_value <= 1

    def get_values(self):
        return self.param

    def set_values(self, *args):
        self.param = args[:1]


class NSigmoid(Sigmoid):
    def to_edn(self) -> str:
        return '[-sigmoid {} {}]'.format(self.center, self.beta)

    def evaluate(self, v) -> float:
        return 1 - (1 / (
                1 + (np.exp(-((np.log(0.99) - np.log(0.01)) / (self.center - self.beta)) * (v - self.center)))))


class PseudoExp(Gaussian):
    def to_edn(self) -> str:
        return '[pseudo-exp {} {}]'.format(self.center, self.deviation)

    def evaluate(self, v) -> float:
        return 1 / (1 + self.deviation * pow((v - self.center), 2))


class SForm(Gamma):
    """
    S-shaped membership function MathWorks-based implementation
    """

    def to_edn(self) -> str:
        return '[Sform {} {}]'.format(self.a, self.b)

    def evaluate(self, v) -> float:
        if v <= self.a:
            return 0
        if self.a <= v <= (self.a + self.b) / 2:
            return 2 * pow((v - self.a) / (self.b - self.a) if (self.b - self.a) != 0 else _nan, 2)
        if (self.a + self.b) / 2 <= v <= self.b:
            return 1 - 2 * pow((v - self.b) / (self.b - self.a) if (self.b - self.a) != 0 else _nan, 2)
        return 1


class ZForm(Gamma):
    """
    Z-shaped memberhip function MathWorks-based implementation
    """

    def to_edn(self) -> str:
        return '[Zform {} {}]'.format(self.a, self.b)

    def evaluate(self, v) -> float:
        if v <= self.a:
            return 1
        if self.a <= v <= (self.a + self.b) / 2:
            return 1 - 2 * pow((v - self.a) / (self.b - self.a) if (self.b - self.a) != 0 else _nan, 2)
        if (self.a + self.b) / 2 <= v <= self.b:
            return 2 * pow((v - self.b) / (self.b - self.a) if (self.b - self.a) != 0 else _nan, 2)
        return 0
