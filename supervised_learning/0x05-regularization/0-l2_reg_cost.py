#!/usr/bin/env python3
"""a function that calculates the cost of a NN with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization:"""
    L2 = 0
    for i in range(L):
        w = weights["W{}".format(i + 1)]
        norm = np.linalg.norm(w)
        L2 += (lambtha / 2 / m * norm)
    return L2 + cost
