#!/usr/bin/env python3
"""Write a function def matrix_shape(matrix): that calculates the shape of
a matrix:
You can assume all elements in the same dimension are of the same type/shape
The shape should be returned as a list of integers"""


def matrix_shape(matrix):
    """Returns Shape of a Matrix"""
    if type(matrix) == list:
        return [len(matrix), *matrix_shape(matrix[0])]
    return []
