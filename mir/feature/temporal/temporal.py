# -*- coding: utf-8 -*-

"""
@file temporal.py
@brief temporal feature extractors
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.signal as sig
from plifia.signal import *
from plifia.mir.feature.rhythm import *


def zerocross_rate(input):
    """
    Calculate zero-crossing rate
    
    Parameters:
      inData: ndarray
        input signal
    
    Returns:
      result: int
        zero-crossing rate
    """
    zero_idx = []
    for i in xrange(len(input)):
        print input[i]
        if input[i] == 0:
            zero_idx.append(i)
            
    #zeroIdx = sp.where(inData == 0.0)
    
    return len(zeroIdx)

def temporal_centroid(input, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate temporal centroid
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
    
    Returns:
      result: ndarray
        temporal centroid
      
    """
    framed = make_framed_data(input, framesize, hopsize, 'hann')
    power = framed ** 2
    t = sp.arange(framesize)
    TC = sp.sum(power * t, axis=1) / sp.sum(power, axis=1)
    
    return TC
    
def log_attack_time(input, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate log-attack-time
    
    Parameters:
    
    Returns:
    
    """
    onset_peak_idx,onset_peak_data,odf,threshold = onset(input, framesize=framesize, hopsize=hopsize, fs=fs, method='logspe')
    
    attack_time = (onset_peak_idx[:, 1] - onset_peak_idx[:, 0]) * hopsize
    log_attack_time = sp.log10(attack_time)
    
    return log_attack_time
