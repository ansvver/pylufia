# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython


def moving_average_simple_cy(np.ndarray[double,ndim=1] input, int n_points):
    """
    Calculate Simple Moving Average
    
    Paremeters:
      input: ndarray
        input signal
      n_points: int
        numbers of moving average points
    
    Returns:
      result: ndarray
        moving-averaged signal
    """
    output = np.zeros( len(input), dtype=float )

    output[0:n_points] = input[0:n_points]
    cdef int i
    cdef int n_input = len(input)
    for i from n_points <= i < n_input:
        output[i] = input[i-n_points:i].sum() / n_points

    return output

def moving_average_exp_cy(np.ndarray[double,ndim=1] input, int n_points):
    """
    Calculate Exponential Moving Average
    
    Paremeters:
      input: ndarray
        input signal
      n_points: int
        numbers of moving average points
    
    Returns:
      result: ndarray
        moving-averaged signal
    """
    cdef double alpha = 2.0 / (n_points + 1)
    cdef double d = 0.0

    cdef int n_input = len(input)
    output = np.zeros( n_input, dtype=np.double )
    cdef int i
    for i from 0 <= i < n_input:
        output[i] = alpha * input[i] + (1-alpha) * d
        d = output[i]

    return output
