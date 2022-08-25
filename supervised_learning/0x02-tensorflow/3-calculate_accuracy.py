#!/usr/bin/env python3
"""function def calculate_accuracy(y, y_pred)"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    max_prediction_index = tf.argmax(y_pred, 1)
    equal = tf.equal(tf.argmax(y, 1), max_prediction_index)

    accuaracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return accuaracy
