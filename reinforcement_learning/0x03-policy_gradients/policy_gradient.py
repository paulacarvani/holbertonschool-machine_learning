#!/usr/bin/env python3
"""
Policiy gradient
"""
import numpy as np


def policy(matrix, weight):
    z = np.dot(matrix, weight)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def policy_gradient(state, weight):
    p = policy(state, weight)
    action = np.random.choice(len(p[0]), p=p[0])
    dsoftmax = p.copy()
    dsoftmax[0, action] -= 1
    dlog = dsoftmax / p
    grad = np.outer(state.T, dlog)
    return action, grad
