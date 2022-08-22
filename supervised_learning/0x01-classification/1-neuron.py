#!/usr/bin/env python3
"""Based on 0-neuron
Replace public instance attribute for:
Private instance attributes:
__W: The weights vector for the neuron. Upon instantiation,
it should be initialized using a random normal distribution.
__b: The bias for the neuron. Upon instantiation,
it should be initialized to 0.
__A: The activated output of the neuron (prediction).
Upon instantiation, it should be initialized to 0.
Each private attribute should have a corresponding
getter function (no setter function)."""


import numpy as np


class Neuron:
    """That defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    """Each private attribute should have a corresponding
    getter function (no setter function)"""

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
