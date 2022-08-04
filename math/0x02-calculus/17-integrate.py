#!/usr/bin/env python3
"""Write a function def poly_integral(poly, C=0):
that calculates the integral of a polynomial:

poly is a list of coefficients representing a polynomial
the index of the list represent the power of x that the coefficient belongs to
Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
C is an integer representing the integration constant
If a coefficient is a whole number, it should be represented as an integer
If poly or C are not valid, return None
Return a new list of coefficients representing the integral of the polynomial
The returned list should be as small as possible
"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) != list or type(C) != int:
        return None

    if poly == []:
        return None

    if poly == [0]:
        return [C]

    result = [C]

    for i in range(len(poly)):
        r = poly[i] / (i+1)
        if r.is_integer():
            r = int(r)
        result.append(r)

    return result
