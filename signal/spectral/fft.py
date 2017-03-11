# -*- coding: utf-8 -*-

"""
====================================================================
fft.py

FFT,STFTの実装
====================================================================
"""

import scipy as sp


def fft(x, fftsize=512):
    """
    Apply FFT
    
    Parameters:
      x: ndarray
        input signal
      fftsize: int
        size of FFT
    
    Returns:
      fftData: ndarray
        result of FFT
    """
    X = sp.fftpack.fft(x, fftsize)
    n_freq = int(fftsize/2)+1
    # X = X.T[:n_freq].T / (fftsize/2.0)
    # X = X.T[:n_freq].T
    X = X[:n_freq]
    # X = sp.array(X, order="C")
    # if len(input.shape) == 1:
        # fftData = fftData[0:nFreq]
    # else:
        # fftData = fftData[:, 0:nFreq]
    
    return X

def ifft(X, fftsize):
    """
    Apply IFFT
    
    Parameters:
      X: ndarray
        input signal
      fftsize: int
        size of IFFT
    
    Returns:
      result: ndarray
        result of IFFT
    """
    rec_x = sp.fftpack.ifft(X, fftsize)
    output = rec_x

    return output
    
def irfft(X, fftsize):
    rec_x = sp.fftpack.irfft(X, fftsize)
    output = rec_x

    return output
    
def fft2d(X, shape=None):
    """
    Apply 2D-FFT
    
    Parameters:
      inData: ndarray
        input signal
      shape: 2d tuple
        shape of 2D-FFT result
    
    Results:
      result: ndarray
        result of 2D-FFT
    """
    return sp.fftpack.fft2(X, shape=shape)
