#!/usr/bin/env python3
"""Write a function def cat_arrays(arr1, arr2):
that concatenates two arrays:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list"""


def cat_arrays(arr1, arr2):
    """concatenates two arrays"""
    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i])
    for i in range(len(arr2)):
        arr3.append(arr2[i])
    return arr3
