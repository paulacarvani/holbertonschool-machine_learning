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
    def __init__(self, nx, layers, activation='sig'):
        """class constructor"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__weights = {}
        prev_layer = nx
        for i, n in enumerate(layers, 1):
            if type(n) is not int or n < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i)] = (
                np.random.randn(n, prev_layer) * np.sqrt(2 / prev_layer))
            self.__weights["b{}".format(i)] = np.zeros((n, 1))
            prev_layer = n
        self.__L = len(layers)
        self.__cache = {}
        self.__activation = activation

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

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
                if self.__activation == 'sig':
                    A_tmp = 1 / (1 + np.exp((-1) * z_tmp))
                else:
                    A_tmp = np.tanh(z_tmp)
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
        m = Y.shape[1]
        m = 1 / m
        wc = self.__weights.copy()
        for i in range(self.L, 0, -1):
            A = cache["A{}".format(i)]
            if i == self.L:
                dz = A - Y
            else:
                if self.__activation == 'sig':
                    g = A * (1 - A)
                else:
                    g = 1 - (A ** 2)
                dz = (wc["W{}".format(i + 1)].T @ dz) * g
            dW = m * (dz @ cache["A{}".format(i - 1)].T)
            db = m * np.sum(dz, axis=1, keepdims=True)
            self.__weights["W{}".format(i)] = (
                self.weights["W{}".format(i)] - (alpha * dW))
            self.__weights["b{}".format(i)] = (
                self.weights["b{}".format(i)] - (alpha * db))

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
