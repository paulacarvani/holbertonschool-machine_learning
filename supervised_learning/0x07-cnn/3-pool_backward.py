#!/usr/bin/env python3
"""3. Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    der = np.zeros_like(A_prev)
    for frame in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    if mode == 'avg':
                        avg_dA = dA[frame, h, w, ch] / kh / kw
                        der[frame, sh*h:sh*h+kh,
                            sw*w:sw*w+kw, ch] += (np.ones((kh, kw)) * avg_dA)
                    if mode == 'max':
                        box = A_prev[frame, sh*h:sh*h+kh, sw*w:sw*w+kw, ch]
                        mask = (box == np.max(box))
                        der[frame, sh*h:sh*h+kh, sw*w:sw*w+kw,
                            ch] += (mask * dA[frame, h, w, ch])
    return der
