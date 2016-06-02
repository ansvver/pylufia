# -*- coding: utf-8 -*-

"""
@file dynamics.py
@brief Dynamics feature extractors
@author ふぇいと (@stfate)

@description

"""

from pylufia.signal.segment import *
from ..mir.feature.common import *
import scipy as sp


def loudness(input, framesize=1024, hopsize=512):
    """
    Calculate loudness
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
    
    Returns:
      result: ndarray
        loudness of inData
    """
    framed_data = makeFramedData(input, framesize, hopsize, 'hann')
    
    J = 1000.0
    L = sp.log(1 + J/framesize * sum(framed_data.T ** 2))
    
    return L

def rms(input, framesize=1024, hopsize=512):
    """
    Calculate RMS
    
    Parameters:
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
    
    Returns:
      result: ndarray
        RMS of inData
    """
    framed_data = makeFramedData(input, framesize, hopsize, 'hann')
    
    rms = sp.sqrt(sp.mean(framed_data, axis=1)**2)
    
    return rms
