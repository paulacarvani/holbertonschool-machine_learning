#!/usr/bin/env python3
"""Write the function def create_placeholders(nx, classes):
that returns two placeholders, x and y, for the neural network:"""

import tensorflow.compat.v1


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder("float", [None], nx, name="x")
    y = tf.placeholder("float", [None, classes], name="y")
    return x, y