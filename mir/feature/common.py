# -*- coding: utf-8 -*-

"""
====================================================================
Common functions for feature extractors
====================================================================
"""

import pylufia.signal as signal
import pylufia.signal.filter.iir as iir
import pylufia.signal.moving_average as ma
import scipy as sp
import scipy.signal as sp_sig


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
    b, a = iir.calc_coef(2.0, 4000.0, 3.0, 'lpf2')
    fil_data = iir.apply(input)

    env_data = abs( sp_sig.hilbert(fil_data) )
    env_data2 = ma.moving_average_exp(env_data, 256)

    return env_data2

def flatness_db(input):
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
