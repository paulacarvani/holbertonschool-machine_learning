#!/usr/bin/env python3
"""Write the function def forward_prop(x, layer_sizes=[], activations=[])"""

import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    create_layer = __import__('1-create_layer').create_layer

    layer = create_layer(x, layer_sizes[0], activations[0])

    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer
