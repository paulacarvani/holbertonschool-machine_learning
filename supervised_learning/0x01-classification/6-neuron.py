#!/usr/bin/env python3
"""Based on 5-neuron.py
Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
Trains the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct
labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception
iterations must be an integer
if iterations is not positive, raise a ValueError with the exception
iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception
alpha must be a float
if alpha is not positive, raise a ValueError with the exception
alpha must be positive
All exceptions should be raised in the order listed above
Updates the private attributes __W, __b, and __A
You are allowed to use one loop
Returns the evaluation of the training data after iterations
of training have occurred"""

import numpy as np


def sigmoid(y):
    """Sigmoid Activation Function"""
    return 1/(1+np.exp(-y))


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
        """Calculates the forward propagation of the neuron"""
        self.__A = sigmoid(np.matmul(self.__W, X) + self.__b)
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = Y * np.log(A) + np.log(1.0000001 - A) * (1-Y)
        return (-cost.sum() / len(np.transpose(Y)))

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.rint(self.A).astype(int)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = (Y.shape[1])
        dz = A-Y
        db = (1/m) * np.sum(dz)
        dw = (1/m) * np.matmul(X, dz.T)
        self.__W = self.__W - alpha * dw.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
