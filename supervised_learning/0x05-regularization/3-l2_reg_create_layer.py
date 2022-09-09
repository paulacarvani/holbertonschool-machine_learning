#!/usr/bin/env python3
"""File That contains the function l2_reg_create_layer"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes L2
    regularization"""
    regularizer = tf.keras.regularizers.L2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))

    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=initializer,
                             kernel_regularizer=regularizer)

    output = tensor(prev)

    return output
