#!/usr/bin/env python3
"""Based on 6-neuron.py
Readme file for more info"""

import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron - updated"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # Update for task 7
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            x = np.arange(0, iterations + 1, step)
            y = np.empty((iterations // step + 1, 4))

        for i in range(iterations + 1):
            self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph:
                    y[i // step] = cost

            self.gradient_descent(X, Y, self.A, alpha)

        if graph:
            y[-1] = cost
            plt.plot(x, y)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.suptitle("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
