#!/usr/bin/env python3
"""Write a class DeepNeuralNetwork"""


import numpy as np
import matplotlib as plt
import pickle


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
        for i in range(1, self.L + 1):
            z_tmp = np.matmul(self.weights["W{}".format(i)], self.__cache[
                "A{}".format(i - 1)]) + self.weights["b{}".format(i)]
            if i == self.L:
                t_exp = np.exp(z_tmp)
                A_tmp = t_exp / np.sum(t_exp, axis=0, keepdims=True)
            else:
                A_tmp = 1 / (1 + np.exp((-1) * z_tmp))
            self.__cache["A{}".format(i)] = A_tmp
        return self.cache["A" + str(self.L)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        evaluation = np.rint(A).astype(np.int)
        return evaluation, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = (Y.shape[1])
        Al = cache["A" + str(self.__L)]
        dAl = -(Y/Al) + (1-Y)/(1-Al)
        for layer in reversed(range(1, self.__L + 1)):
            Al = cache["A" + str(layer)]
            gl_d = Al * (1 - Al)
            dZl = np.multiply(dAl, gl_d)
            Al_1 = cache["A" + str(layer - 1)]
            dWl = (1/m) * np.matmul(dZl, Al_1.T)
            dbl = (1/m) * np.sum(dZl, axis=1, keepdims=True)
            Wl = self.__weights["W" + str(layer)]
            dAl = np.matmul(Wl.T, dZl)

            kW = "W" + str(layer)
            kb = "b" + str(layer)
            self.__weights[kW] = self.__weights[kW] - alpha * dWl
            self.__weights[kb] = self.__weights[kb] - alpha * dbl

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) != int:
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        g_x = []
        g_y = []
        for i in range(iterations):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            c = self.cost(Y, A)
            if not i % 100:
                if verbose:
                    print('Cost after {} iterations: {}'.format(i, c))
                g_x.append(i)
                g_y.append(c)

        A, cost = self.evaluate(X, Y)
        if verbose:
            print('Cost after {} iterations: {}'.format(iterations, cost))
        g_x.append(iterations)
        g_y.append(cost)
        if graph:
            plt.plot(g_x, g_y)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return A, cost

    def save(self, filename):
        """saves the instance object to a file in pickle format"""
        if ".pkl" not in filename:
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """loads a pickled neural network"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
