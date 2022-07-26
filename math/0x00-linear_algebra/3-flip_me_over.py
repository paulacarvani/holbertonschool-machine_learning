#!/usr/bin/env python3
"""Write a function def matrix_transpose(matrix):
that returns the transpose of a 2D matrix, matrix:

You must return a new matrix
You can assume that matrix is never empty
You can assum all elements in the same dimension are of the same type/shape"""


def matrix_transpose(matrix):
    """return a new matrix"""
    m2 = []
    for i in range(len(matrix[0])):
        row = []
        for item in matrix:
            row.append(item[i])
        m2.append(row)
    return m2
