#!/usr/bin/env python3
"""Write a function def poly_derivative(poly)
that calculates the derivative of a polynomial:

poly is a list of coefficients representing a polynomial
the index of the list represents the power of x that the
coefficient belongs to
Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
If poly is not valid, return None
If the derivative is 0, return [0]
Return a new list of coefficients representing the derivative
of the polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if type(poly) != list:
        return None
    if poly == []:
        return None
    if len(poly) == 1:
        return [0]
    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])

    if result == [0] * len(result):
        return [0]

    return result
