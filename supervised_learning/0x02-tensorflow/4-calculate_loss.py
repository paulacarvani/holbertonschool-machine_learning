#!/usr/bin/env python3
"""function def calculate_loss(y, y_pred):"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
