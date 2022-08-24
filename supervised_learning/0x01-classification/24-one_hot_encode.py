#!/usr/bin/env python3
"""function def one_hot_encode(Y, classes)"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    A = np.eye(classes)[Y]
    return A.T
