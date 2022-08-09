#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


pi = 3.1415926536
e = 2.7182818285


def factorial(n):
    """Finds the factorial of a given number"""
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Poisson:
    """Represents a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Initialize Poisson"""

        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of successes"""

        if k < 0:
            return 0
        if type(k) != int:
            k = int(k)
        result = (e**(-self.lambtha) * self.lambtha**k) / factorial(k)
        return result
