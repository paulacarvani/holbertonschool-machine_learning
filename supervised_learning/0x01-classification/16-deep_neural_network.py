#!/usr/bin/env python3
"""Write a class DeepNeuralNetwork"""


import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.weights["W" + str(i + 1)] = np.random.normal(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights["W" + str(i+1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights["b" + str(i + 1)] = np.zeros(
                layers[i]).reshape(layers[i], 1)
