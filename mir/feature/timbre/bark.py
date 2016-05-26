# -*- coding: utf-8 -*-

"""
@file bark.py
@brief barkband feature extractors
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from plifia.signal.spectral import *


def _make_bark_filterbank(fs, framesize):
    """
    Calculate Bark-band filterbank
    """
    f_centers = sp.array([50,150,250,350,450,570,700,840,1000,1170,1370,1600,1850,2150,2500,2900,3400,4000,4800,5800,7000,8500,10500,13500])
    f_lowers = sp.array([20,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000])
    f_uppers = sp.array([100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500])
    
    n_freqs = framesize/2
    n_bark_band = len(f_centers)
    
    fidx_centers = (framesize * f_centers / float(fs)).astype('int')
    fidx_lowers = (framesize * f_lowers / float(fs)).astype('int')
    fidx_uppers = (framesize * f_uppers / float(fs)).astype('int')
    
    filterbank = sp.zeros( (n_bark_band, n_freqs) )
    
    for n in xrange(n_bark_band):
        inc = 1.0 / (fidx_centers[n] - fidx_lowers[n])
        idxs = sp.arange(fidx_lowers[n], fidx_centers[n])
        filterbank[n, fidx_lowers[n]:fidx_centers[n]] = (idxs - fidx_lowers[n]) * inc
        # filterbank[n, fidx_lowers[n]:fidx_centers[n]] = 1.0
        dec = 1.0 / (fidx_uppers[n] - fidx_centers[n])
        idxs = sp.arange(fidx_centers[n], fidx_uppers[n])
        filterbank[n, fidx_centers[n]:fidx_uppers[n]] = 1.0 - (idxs - fidx_centers[n]) * dec
        # filterbank[n, fidx_centers[n]:fidx_uppers[n]] = 1.0
        
    return filterbank
    
def bark_spectrogram(input, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate bark-scaled spectrogram
    
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
        bark-scaled spectrogram
    """
    S,F,T = stft(input, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    # bark_idx = [int(_hz2bark(F[i])) for i in xrange(len(F))]
    
    # bark_spe = sp.zeros((S.shape[1], n_bark_band))
    # for i in xrange(S.shape[0]):
        # bark_spe[:, bark_idx[i]] += S[i, :]
        
    # for i in xrange(n_bark_band):
        # count = bark_idx.count(i)
        # bark_spe[:, i] /= count
    bark_filterbank = _make_bark_filterbank(fs, framesize)
    bark_spe = sp.dot(S.T, bark_filterbank.T)
        
    return bark_spe

def _hz2bark(f):
    """
    Hz -> Bark
    """
    return 13 * sp.arctan(f / 1315.8) + 3.5 * sp.arctan(f / 7518.0)