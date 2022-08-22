#!/usr/bin/env python3
"""Write a class Neuron that defines a single neuron
performing binary classification:

class constructor: def __init__(self, nx):
nx is the number of input features to the neuron
If nx is not an integer, raise a TypeError with the exception:
nx must be an integer
If nx is less than 1, raise a ValueError with the exception:
nx must be a positive integer
All exceptions should be raised in the order listed above
Public instance attributes:
W: The weights vector for the neuron. Upon instantiation,
it should be initialized using a random normal distribution.
b: The bias for the neuron. Upon instantiation,
it should be initialized to 0.
A: The activated output of the neuron (prediction).
Upon instantiation, it should be initialized to 0."""
import numpy as np


class Neuron:
    """That defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
