#!/usr/bin/env python3
"""function def one_hot_decode(one_hot)"""

import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    lables = one_hot.argmax(0)

    return lables
