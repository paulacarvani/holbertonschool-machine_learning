#!/usr/bin/env python3
"""Create a class Exponential that represents
an exponential distribution"""


pi = 3.1415926536
e = 2.7182818285


def factorial(n):
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Exponential:

    def __init__(self, data=None, lambtha=1.):

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        if x < 0:
            return 0

        return self.lambtha * e**(-self.lambtha * x)

    def cdf(self, x):
        if x < 0:
            return 0

        return 1 - e**(-self.lambtha * x)
