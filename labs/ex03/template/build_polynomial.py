# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    poly_x = np.ones([x.shape[0]])
    for i in range(1,degree+1):
        x_ = np.power(x,i)
        poly_x = np.c_[poly_x, np.power(x,i)]
    poly_x = np.array(poly_x)
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return poly_x
    # ***************************************************
    raise NotImplementedError
