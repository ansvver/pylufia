# -*- coding: utf-8 -*-

"""
cepstrum.py

ケプストラム分析
"""

import scipy as sp
from pylufia.signal.common import *
from .fft import *


def cepstrum(x, framesize=256, fs=44100):
    """
    Calculate cepstrum from 1-frame signal
    
    Parameters:
      x: ndarray
        input signal
      framesize: int
        framesize of fft
      fs: int
        samplingrate of input signal
    
    Returns:
      result: ndarray
        cepstrum data
    """
    ## FFT
    X = fft(x, framesize)
    X = sp.absolute(X)
    
    ## Log
    log_X = sp.log10(X + 0.00001)
    
    ## IFFT
    ceps = ifft(log_X, framesize)
    
    return ceps
    
def cepstrogram(x, framesize=512, hopsize=256, window='hann', fs=44100):
    """
    Calculate time-invarianted cepstrum from long signal
    
    Parameters:
      input: ndarray
        input signal
      framesize: int
        framesize of analysis
      hopsize: int
        hopsize of analysis
      window: string
        type of window function
      fs: int
        samplingrate
    
    Returns:
      result: ndarray
        short-time cepstrum
    """
    n_frames = int(sp.ceil((len(x) - framesize) / hopsize))
    n_freq = sp.ceil((framesize + 1) / 2.0)
    framed_x = makeFramedData(x, framesize, hopsize, window)
    X = cepstrum(framed_x, framesize)
    X = X.T
    
    return X
