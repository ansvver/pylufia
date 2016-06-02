# -*- coding: utf-8 -*-

"""
@file common.py
@brief common functions of tonal extractors
@author ふぇいと (@stfate)

@description

"""

import pylufia.signal as signal
import scipy as sp


def envelope(input):
    """
    Calculate temporal envelope
    (LPF&Hilbert transform)
    
    Parameters:
      inData: ndarray
        input signal
    
    Returns:
      result: ndarray
        envelope signal
    """
    b, a = iirCalcCoef(2.0, 4000.0, 3.0, 'lpf2')
    fil_data = iirApply(input)

    env_data = abs(sig.hilbert(fil_data))
    env_data2 = calcEMA(env_data, 256)

    return env_data2

def flatnessDB(input):
    """
    Calculate flatness
    
    Parameters:
      inData: ndarray
        input signal
    
    Returns:
      result: ndarray
        flatness
    """
    arith_mean = reduce(lambda x, y: x + y, input) / len(input)
    geo_mean = reduce(lambda x, y: x * y, input) / len(input)

    flatness = geo_mean / arith_mean

    return 20 * sp.log(flatness)

def check_nan_1d(x):
    """
    NaN check for 1-D array
    
    Parameters:
      x: ndarray
        input data
    
    Returns:
      result: ndarray
        NaN removed data
    """
    index = sp.array(sp.where(x != x))
    x[index] = 0.0
    
    return x
    
def check_nan_2d(X):
    """
    NaN check for 2-D array
    
    Parameters:
      x: ndarray
        input data
    
    Returns:
      result: ndarray
        NaN removed data
    """
    index = sp.array(sp.where(X != X))
    X[index[0, :], index[1, :]] = 0.0
    
    return X
    
def check_inf_1d(x):
    """
    +/-inf check for 1-D array
    
    Parameters:
      x: ndarray
        input data
    
    Returns:
      result: ndarray
        Inf removed data
    """
    index = sp.array(sp.where(x == float('-inf')))
    x[index] = 0.0
    
    index = sp.array(sp.where(x == float('inf')))
    x[index] = 0.0
    
    return x
    
def check_inf_2d(X):
    """
    +/-inf check for 2-D array
    
    Parameters:
      x: ndarray
        input data
    
    Returns:
      result: ndarray
        Inf removed data
    """
    index = sp.array(sp.where(X == float('-inf')))
    X[index[0, :], index[1, :]] = 0.0
        
    index = sp.array(sp.where(X == float('inf')))
    X[index[0, :], index[1, :]] = 0.0
    
    return X
