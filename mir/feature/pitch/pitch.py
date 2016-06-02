# -*- coding: utf-8 -*-

"""
====================================================================
Pitch feature extractors
====================================================================
"""

import scipy as sp
from pylufia.mir.feature.common import *
from pylufia.signal import *
from pylufia.signal.spectral import *
from pylufia.stats.distance import *


def pitch(input, framesize=8192, hopsize=220, fs=44100):
    """
    Pitch detection
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
      fmin: float
        minimum frequency of f0 candidate
      fmax: float
        maximum frequency of f0 candidate
    
    Returns:
      result: ndarray
        pitch contour array
    
    Notes:
      とりあえず周波数領域のMaximum likelihoodで実装してみる．
      2013.03.22 実装してみたがsin波でもピッチを間違える，というか窓の振幅で結果が変わる．
      アルゴリズムの理解を誤っているところがあるやも．
      2013.03.26 Cepstrumによる手法で実装しなおし
    
    """
    ceps = cepstrogram(input, framesize, hopsize, fs, 'hann')
    ceps_fft = fft(ceps.T, framesize)
    f0_contour = sp.argmax(ceps_fft, axis=1) * (fs/2) / (framesize/2)
    
    return f0_contour

def inharmonicity(input, framesize=8192, hopsize=220, fs=44100):
    """
    Calculate Inharmonicity(Divergence of the signal spectral components from a purely harmonic signal)
    
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
        time-variant inharmonicity
    """
    S, F, T = stft(input, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    f0_contour = pitch(input, framesize, hopsize, fs)
    
    ## W[n*f0] = 0, W[(n*f0 + (n+1)*f0)/2] = 1 となるような三角窓を生成
    f0_window = sp.zeros( (len(f0contour), framesize/2) )
    for idx, f0 in enumerate(f0contour):
        n_f0s = sp.arange(f0, 10*f0, f0)
        n_f0_idxs = sp.around(n_f0s * framesize/2 / fs/2).astype(int)
        # f0_window = sp.zeros(framesize/2)
        for i in xrange(len(n_f0_idxs)-1):
            cur_idx = n_f0_idxs[i]
            next_idx = n_f0_idxs[i+1]
            peak_idx = int((next_idx+cur_idx)/2)
            f0_window[idx, cur_idx:peak_idx+1] = sp.arange(0, 1.0, 1.0/(peak_idx-cur_idx+1))
            f0_window[idx, peak_idx+1:next_idx] = sp.arange(1.0, 0, -1.0/(next_idx-peak_idx-1))
    
    ## 上記の三角窓をかけてinharmonicity計算
    inharmo = sp.dot(S.T, f0_window.T) / sp.sum(S, axis=0)
    inharmo = sp.diag(inharmo)
    
    return inharmo
