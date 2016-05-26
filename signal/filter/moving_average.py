# -*- coding: utf-8 -*-

"""
@file moving_average.py
@brief moving average functions
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def moving_average_simple(input, nPoints):
    """
    Calculate Simple Moving Average
    
    Paremeters:
      input: ndarray
        input signal
      nPoints: int
        numbers of moving average points
    
    Returns:
      result: ndarray
        moving-averaged signal
    """
    output = sp.zeros(len(input))

    output[0:nPoints] = input[0:nPoints]
    for i in xrange(nPoints, len(input)):
        output[i] = sum(input[i-nPoints:i]) / nPoints

    return output

def moving_average_exp(input, nPoints):
    """
    Calculate Exponential Moving Average
    
    Paremeters:
      inData: ndarray
        input signal
      nPoints: int
        numbers of moving average points
    
    Returns:
      result: ndarray
        moving-averaged signal
    """
    alpha = 2.0 / (nPoints + 1)
    d = 0.0

    output = sp.zeros(len(input))
    for i, _input in enumerate(input):
        output[i] = alpha * _input + (1-alpha) * d
        d = output[i]

    return output
