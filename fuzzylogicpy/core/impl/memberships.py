from __future__ import annotations

from typing import List

import numpy as np

from fuzzylogicpy.core.membership_function import MembershipFunction


class Sigmoid(MembershipFunction):
    def get_values(self):
        return [self.center, self.beta]

    def set_values(self, values: List):
        self.center, self.beta = values[0], values[1]

    def is_valid(self) -> bool:
        pass

    def __init__(self, center: float, beta: float):
        super().__init__()
        self.center = center
        self.beta = beta

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
    def get_values(self):
        return [self.beta, self.gamma, self.m]

    def set_values(self, values: List):
        self.beta, self.gamma, self.m = values[0], values[1], values[2]

    def is_valid(self) -> bool:
        return self.beta < self.gamma and 0 <= self.m <= 1

    def __init__(self, beta: float, gamma: float, m: float):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.m = m

    def evaluate(self, value) -> float:
        sigmoid = pow(Sigmoid(self.gamma, self.beta).evaluate(value), self.m)
        sigmoid2 = pow(1 - Sigmoid(self.gamma, self.beta).evaluate(value), 1 - self.m)
        m_ = pow(self.m, self.m) * pow((1 - self.m), (1 - self.m))
        return (sigmoid * sigmoid2) / m_

    def derive(self, value: float, param: str) -> float:
        if param == "g":
            #       e^(g(-b+x)) * (1-m)^(m-1) * ((1 / 1 + e^(c(g-x)) )^(m-1) - 1) * (b-x)
            #  - _______________________________________________________________________
            #                              (1 + e^(c(-b + x))^2
            result = (
                    -1
                    * (
                            np.exp(self.gamma * (-self.beta + value))
                            * pow(1 - self.m, self.m - 1)
                            * (
                                    pow(
                                        (
                                            1 / (1 + np.exp(self.gamma * (self.beta - value))),
                                            self.m - 1,
                                        )
                                    )
                                    - 1
                            )
                            * (self.beta - value)
                    )
                    / pow(1 + np.exp(self.gamma * (-self.beta + value)), 2)
            )
        elif param == "b":
            #    ce^(c(x-b)) * (1-m)^(m-1) * m^-m * (( 1/ 1+e^(c(b-x)) )-1)^(m-1)
            #  - ________________________________________________________________
            #                       (1 + e^(c(x-b)))2
            result = (
                    -1
                    * (
                            self.gamma
                            * np.exp(self.gamma * (value - self.beta))
                            * pow(1 - self.m, self.m - 1)
                            * pow(self.m, -self.m)
                            * pow(
                        (1 / (1 + np.exp(self.gamma * (self.beta - value)))) - 1,
                        self.m - 1,
                    )
                    )
                    / pow(1 + np.exp(self.gamma * (value - self.beta)), 2)
            )
        elif param == "m":
            Sg = Sigmoid(self.gamma, self.beta).evaluate(value)
            #  (1-m)^(1+m) * m^-m * (1 -Sg )^(1-m) * (log(1-m) - log(m) - log(1-Sg) + log(Sg))
            result = (
                    pow(1 - self.m, self.m - 1)
                    * pow(self.m, -self.m)
                    * pow(1 - Sg, 1 - self.m)
                    * (np.log(1 - self.m) - np.log(self.m) - np.log(1 - Sg) + np.log(Sg))
            )
        return result
