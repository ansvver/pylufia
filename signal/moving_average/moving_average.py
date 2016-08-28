# -*- coding: utf-8 -*-

import scipy as sp
# from numba import jit


# @jit
def moving_average_simple(input, n_points):
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
    output = sp.zeros( len(input) )

    output[0:nPoints] = input[0:n_points]
    for i in range( n_points, len(input) ):
        output[i] = sum( input[i-n_points:i] ) / n_points

    return output

# @jit
def moving_average_exp(input, n_points):
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
    alpha = 2.0 / (n_points + 1)
    d = 0.0

    output = sp.zeros( len(input) )
    for i,_input in enumerate(input):
        output[i] = alpha * _input + (1-alpha) * d
        d = output[i]

    return output

# @jit
def moving_average_exp_numba(input, n_points):
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
    alpha = 2.0 / (n_points + 1)
    d = 0.0

    output = sp.zeros( len(input) )
    for i,_input in enumerate(input):
        output[i] = alpha * _input + (1-alpha) * d
        d = output[i]

    return output
