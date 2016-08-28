# -*- coding: utf-8 -*-

"""
====================================================================
Tonal feature extractors
====================================================================
"""

import scipy as sp
from pylufialib.signal.spectral import *
from pylufialib.feature.common import *


def semitone_pow(input, framesize=2048, hopsize=512, fs=44100, fmin=50, fmax=1400):
    """
    Calculate semitone-power
    
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
        min freq
      fmax: float
        max freq
    
    Returns:
      result: ndarray
        semitone power
    """
    S,F,T = stft(input, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S) ** 2

    n_frames = S.shape[1]
    semitone_idx = [_freq2semitone(f) for f in F]
    
    STP = sp.zeros( (n_frames, sp.amax(semitone_idx)+1) )
    
    fidx_min = sp.where(F <= fmin)[0][0] + 1
    fidx_max = sp.where(F > fmax)[0][0] - 1
    for frm in range(n_frames):
        #for f in xrange(S.shape[0]):
        for f in range(fidx_min, fidx_max):
            if semitone_idx[f] >= 0:
                STP[frm, semitone_idx[f]] += S[f, frm]
                
    STP /= sp.amax(STP)
                
    return STP
    
    #semitoneFreqBin = [440.0*2**((minNoteId+i-69.0)/12.0) for i in xrange(nSemitones)]
    #semitoneFreqBinLow = [semitoneFreqBin[i] / (2**(0.5/12.0)) for i in xrange(nSemitones)]
    #semitoneFreqBinHigh = [semitoneFreqBin[i] * (2**(0.5/12.0)) for i in xrange(nSemitones)]
    #semitoneFreqMap = sp.zeros(len(F))

    #for i, freq in enumerate(F):
    #    curFreq = freq
    #    if (curFreq < semitoneFreqBinLow[0] or curFreq > semitoneFreqBinHigh[-1]):
    #        semitoneFreqMap[i] = -1
    #    else:
    #        for j, val in enumerate(semitoneFreqBin):
    #            if (curFreq >= semitoneFreqBinLow[j] and curFreq < semitoneFreqBinHigh[j]):
    #                semitoneFreqMap[i] = j

    #semitonePowData = sp.zeros( (nFrames, nSemitones) )

    #for j in range(S.shape[0] - 1):
    #    if semitoneFreqMap[j] != -1:
    #        semitonePowData[:, semitoneFreqMap[j]] += S[j, :]

    #return semitonePowData

def chroma(input, framesize=2048, hopsize=512, fs=44100, fmin=50, fmax=1400):
    """
    Calculate chroma vector
    
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
        min freq
      fmax: float
        max freq
    
    Returns:
      result: ndarray
        chroma vector
    """
    semitone_pow_data = semitone_pow(input, framesize, hopsize, fs, fmin, fmax)
    n_frames = semitone_pow_data.shape[0]
    n_semitones = semitone_pow_data.shape[1]

    chroma_data = sp.zeros( (n_frames, 12) )

    for j in range(n_semitones):
        chroma_data[:, j%12] += semitone_pow_data[:, j]
        
    #chromaData = sp.log10(chromaData)
    # chromaData = _normChroma(chromaData)
    chroma_data = checkNaN2D(chroma_data)

    return chroma_data


## Local functions

def _freq2semitone(f):
    """
    周波数 -> semitone index(C2=130.81Hz origin)
    """
    # if f < 130.81: # min=C2
        # return -1
    # else:
        # return int(sp.around(21.0 + 12.0 * sp.log2(f/440.0)))
    if f < 32.70: # min=C0
        return -1
    else:
        return int(sp.around(21.0 + 12.0 * sp.log2(f/110.0)))

def _normChroma(input):
    """
    chroma正規化(全次元の総和=1)
    """
    for i in range(input.shape[0]):
        frm_sum = sum(input[i])
        input[i] /= float(frm_sum)

    return input
