#!/usr/bin/env python3
"""Write a function def add_matrices2D(mat1, mat2):
that adds two matrices element-wise

You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same type/shape
You must return a new matrix
If mat1 and mat2 are not the same shape, return None"""


def add_matrices2D(mat1, mat2):
    """adds two matrices element-wise"""
    mat3 = []
    if len(mat1[0]) != len(mat2[0]):
        return None
    for row in range(len(mat1)):
        lit = []
        for i in range(len(mat1[0])):
            lit.append(mat1[row][i] + mat2[row][i])
        mat3.append(lit)
    return mat3
