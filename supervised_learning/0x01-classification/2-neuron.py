#!/usr/bin/env python3
"""Based on 1-neuron.py
Add the public method def forward_prop(self, X):
Calculates the forward propagation of the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Updates the private attribute __A
The neuron should use a sigmoid activation function
Returns the private attribute __A"""


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

    def forward_prop(self, X):
        self.__A = sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A


# Sigmoid Activation Function
def sigmoid(y):
    return 1/(1+np.exp(-y))
