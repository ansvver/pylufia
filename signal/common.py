# -*- coding: utf-8 -*-

"""
@file common.py
@brief common functions of signal package
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def autocorr(input, n_lag):
    """
    Calculate autocorrelation
    
    Parameters:
      inData: ndarray
        input signal
      nLag: int
        number of lags
    
    Returns:
      result: ndarray
        autocorrelation
    """
    R = sp.zeros(n_lag)

    for lag in xrange(n_lag):
        R[lag] = sum(input[lag:len(input)] * input[0:len(input)-lag])

    return R

def crosscorr(input0, input1, n_lag):
    """
    Calculate crosscorrelation
    
    Parameters:
      inData0: ndarray
        input signal 1
      inData1: ndarray
        input signal 2
      nLag:
        number of lags
    
    Returns:
      result: ndarray
        crosscorrelation
    """
    R = sp.zeros(n_lag)

    for lag in xrange(n_lag):
        R[lag] = sum(input0[lag:len(input0)] * input1[0:len(input1)-lag])

    return R


def half_rect(input):
    """
    Half rectification
    
    Parameters:
      inData: ndarray
        input signal
    
    Returns:
      result: ndarray
        half-rectified signal
    """
    return (input + abs(input)) / 2.0
    
def all_rect(input):
    """
    All rectification
    
    Parameters:
      inData: ndarray
        input signal
    
    Returns:
      result: ndarray
        all-rectified signal
    """
    return abs(input)

def peak(input, threshold):
    """
    Peak-picking (for 1d array)
    
    Parameters:
      inData: ndarray
        input signal
      threshold: float
        threshold for peak picking
    
    Returns:
      result: ndarray
        peak amplitude array
    """
    peak_data = sp.zeros(len(input))
    for i in xrange(len(input)-1):
        if input[i] - input[i-1] > 0 and input[i+1] - input[i] < 0:
            if input[i] >= threshold[i]:
                peak_data[i] = input[i]
    
    return peak_data
