#!/usr/bin/env python3
"""Write a class DeepNeuralNetwork"""


import numpy as np


def sigmoid(y):
    """Sigmoid Activation Function"""
    return 1/(1+np.exp(-y))


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            if type(layers[i]) != int or layers[i] < 0:
                raise TypeError('layers must be a list of positive integers')
            kb = "b" + str(i + 1)
            kW = "W" + str(i + 1)

            self.__weights[kb] = np.zeros(layers[i]).reshape(layers[i], 1)
            if i > 0:
                aux = layers[i-1]
            else:
                aux = nx
            self.__weights[kW] = np.random.randn(
                layers[i], aux) * np.sqrt(2/aux)

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for layer in range(self.__L):
            Al = self.__cache["A" + str(layer)]
            Wl = self.__weights["W" + str(layer + 1)]
            bl = self.__weights["b" + str(layer + 1)]
            Zl = np.matmul(Wl, Al) + bl
            self.__cache["A" + str(layer + 1)] = sigmoid(Zl)
        return self.__cache["A" + str(layer + 1)], self.__cache
