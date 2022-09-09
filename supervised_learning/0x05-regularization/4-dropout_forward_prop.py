#!/usr/bin/env python3
"""File that contains the function """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """X is a numpy.ndarray of shape (nx, m) containing the input data for
    the network"""
    cache = {}
    cache["A0"] = X

    Layers = range(L + 1)[1: L + 1]

    for i in Layers:
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A = cache["A" + str(i - 1)]
        Z = np.matmul(W, A) + b

        row = Z.shape[0]
        col = Z.shape[1]
        dropout = np.random.rand(row, col)
        dropout = np.where(dropout >= keep_prob, 0, 1)

        if i == L:
            softmax = np.exp(Z)
            cache["A" + str(i)] = (softmax /
                                   np.sum(softmax, axis=0, keepdims=True))

        else:
            tanh = np.tanh(Z)
            cache["A" + str(i)] = tanh
            cache["D" + str(i)] = dropout
            cache["A" + str(i)] *= dropout
            cache["A" + str(i)] /= keep_prob

    return cache
