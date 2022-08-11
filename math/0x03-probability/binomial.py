#!/usr/bin/env python3
"""Binomial distribution"""


pi = 3.1415926536
e = 2.7182818285


def factorial(n):
    """Finds the factorial of a given number"""
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)


class Binomial:
    """Represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial"""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            s_dif = []
            for d in data:
                s_dif.append((d - mean)**2)
            stddev = (sum(s_dif) / len(s_dif))**(1/2)
            var = stddev**2
            p = -((var / mean) - 1)
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """Calculates the value of the PMF
        for a given number of successes"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        a = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        result = a * self.p**k * (1 - self.p)**(self.n - k)
        return result

    def cdf(self, k):
        """Calculates the value of the CDF
        for a given number of successes"""
        if type(k) != int:
            k = int(k)
        if k < 0:
            return 0
        p = 0
        for i in range(k + 1):
            p += self.pmf(i)
        return p
