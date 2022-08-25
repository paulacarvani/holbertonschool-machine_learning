#!/usr/bin/env python3
"""Write the function def create_layer(prev, n, activation)"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function"""
    A = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=A,
                            name="layer")
    nlayer = layer(prev)
    return nlayer
