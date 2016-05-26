# -*- coding: utf-8 -*-

"""
@file spectral.py
@brief spectral feature extractors
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from plifia.signal.spectral import *
from plifia.mir.feature.common import *


def dim_reduced_spectrogram(x, framesize=1024, hopsize=512, fs=44100, window='hann', max_freq=22050, n_dims=88):
    """
    周波数次元数を削減した振幅スペクトログラム
    
    次元削減の方法は単純にframesize/n_dimごとに平滑化するだけ．
    """
    X,F,T = stft(x, framesize, hopsize, fs, window)
    X = sp.absolute(X)
    
    n_fbins = int(framesize/2 * max_freq*2/float(fs))
    n_fbin_group = n_fbins/float(n_dims)
    X = X[:n_fbins,:]
    
    X_dim_reduced = sp.zeros( (n_dims,X.shape[1]) )
    
    f_idx_start = 0
    for i in xrange(n_dims):
        f_idx_start += n_fbin_group
        f_idx_end = f_idx_start + n_fbin_group
        # X_dim_reduced[i,:] = sp.mean(X[int(f_idx_start):int(f_idx_end),:], axis=0)
        # X_dim_reduced[i,:] = sp.median(X[int(f_idx_start):int(f_idx_end),:], axis=0)
        X_dim_reduced[i,:] = sp.amax(X[int(f_idx_start):int(f_idx_end),:], axis=0)
        
    return X_dim_reduced

def octave_spectrogram(x, framesize=1024, hopsize=512, fs=44100, fmin=60, n_per_octave=1, n_octaves=8):
    """
    Calculate octave-band spectrum
    
    Parameters:
      input: ndarray
        input data (waveform or spectrogram)
      itype: string
        input data type
          wav: waveform
          spe: spectrogram
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
      fmin: float
        min freqency
      n_per_octaves: int
        n of dimensions per octaves
      n_octaves: int
    
    Returns:
      result: ndarray
        octave-band spectrogram
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)

    n_frames = S.shape[1]
    n_dims = n_per_octave * n_octaves
    # freq_low = [fmin * 2**(i/n_per_octave) for i in xrange(n_dims)]
    # freq_high = [fmin * 2**((i+1)/n_per_octave) for i in xrange(n_dims)]
    # idx_low = [int(freq_low[i] * framesize/2 / (fs/2)) for i in xrange(n_dims)]
    # idx_high = [int(freq_high[i] * framesize/2 / (fs/2)) for i in xrange(n_dims)]
    
    freq_low = fmin * 2**(sp.arange(n_dims)/float(n_per_octave))
    freq_high = fmin * 2**(sp.arange(1,n_dims+1)/float(n_per_octave))
    idx_low = (freq_low * framesize/2 / (fs/2)).astype('int')
    idx_high = (freq_high * framesize/2 / (fs/2)).astype('int')
    
    oct_spe_data = sp.zeros((n_frames, n_dims))
    for i in xrange(n_dims):
        oct_spe_data[:, i] = sp.sum(S[idx_low[i]:idx_high[i]+1, :], axis=0)
        
    return oct_spe_data

def flux(x, framesize=512, hopsize=256, window='hann', fs=44100):
    """
    Calculate flux (Spectral difference)
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      window: string
        type of window function
      fs: int
        samplingrate
    
    Returns:
      result: ndarray
        flux of input signal
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    diff_S = S[1:S.shape[0], :] - S[0:S.shape[0]-1, :]
    flux_data = (halfRect(diff_S)**2).sum()

    return flux_data

def spectral_centroid(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral centroid
    
    Parameters:
      input: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
    
    Returns:
      result: ndarray
        spectral centroid of input signal
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hamming')
    S = sp.absolute(S)
    
    n_frames = S.shape[1]
    SC = sp.zeros(n_frames)

    normalized_freq = F
    SC = (normalized_freq[:,sp.newaxis] * S).sum(0) / S.sum(0)
    
    # nan check & inf check
    SC = checkNaN1D(SC)
    SC = checkInf1D(SC)

    return SC

def hfc(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate HFC (High Frequency Content)
    
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
        HFC
    
    Notes:
      Spectral Centroidとの違いはスペクトログラムのエネルギーで正規化するか否か，のみ．
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    n_frames = S.shape[1]
    hfc_data = sp.zeros(n_frames)

    hfc_data = (F * S.T).T.sum(0)

    return hfc_data
    
def spectral_spread(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral spread
    
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
        spectral spread
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hamming')
    S = sp.absolute(S)
    
    u = spectral_centroid(x, framesize=framesize, hopsize=hopsize, fs=fs)
    
    prob_S = S / S.sum(0)
    normalized_freq = F
    centremoved_F = sp.array([normalized_freq-cur_u for cur_u in u])
    
    spread = (centremoved_F**2 * prob_S.T).sum(1)

    return spread
    
def spectral_entropy(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral entropy
    
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
        spectral entropy [frame * 1]
    
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    pmf = S / S.sum(0)
    entropy = - ( pmf * sp.log2(pmf) ).sum(0)
    
    return entropy
    
def spectral_skewness(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral skewness
    
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
        spectral skewness
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    n_frames = S.shape[1]
    
    u = spectralCentroid(x, framesize=framesize, hopsize=hopsize, fs=fs)
    
    m = sp.zeros(n_frames)
    for i in xrange(n_frames):
        m[i] = ( (F - u[i])**3 * S[:, i] / S[:, i].sum() ).sum()
        
    ss = spectralSpread(x, framesize, hopsize, fs)
    
    skewness = m / (ss**3)
    
    return skewness
    
def spectral_kurtosis(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral kurtosis
    
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
        spectral kurtosis
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hamming')
    S = sp.absolute(S)
    
    n_frames = S.shape[1]
    
    u = spectral_centroid(x, framesize=framesize, hopsize=hopsize, fs=fs)
    
    normalized_freq = F
    m = sp.zeros(n_frames)
    for i in xrange(n_frames):
        m[i] = ( (normalized_freq - u[i])**4 * S[:, i] / S[:, i].sum() ).sum()
        
    ss = spectral_spread(x, framesize=framesize, hopsize=hopsize, fs=fs)
    
    kurtosis = m / (ss**4)
    
    return kurtosis

def spectral_slope(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral slope
    """
    pass

def spectral_peak(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral peaks
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    # spePeak = sp.array(map(peak, S.T, sp.amax(S, axis=0)/20.0))
    # numPeaks = sp.array(map(sp.count_nonzero, spePeak))
    n_peaks = peak2d_cy(S.T)
    
    return n_peaks
    
def spectral_flatness(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral flatness
    
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
        spectral flatness
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hamming')
    S = sp.absolute(S)
    
    # [250-500Hz],[500-1000Hz],[1000-2000Hz],[2000-4000Hz]の4バンドでみる
    freq_bands = sp.array( [ [250,500], [500,1000], [1000,2000], [2000,4000] ] )
    freq_bands_idx = ( freq_bands * (framesize/2) / (fs/2) ).astype('int')
    n_freq_bands = freq_bands.shape[0]
    
    SFM = sp.zeros( (S.shape[1], n_freq_bands) )
    for i in xrange(SFM.shape[1]):
        sub_S = S[freq_bands_idx[i,0]:freq_bands_idx[i,1]+1]
        K = sub_S.shape[0]
        SFM[:,i] = sub_S.prod(0) ** (1/float(K)) / sub_S.mean(0)
        
    # tonalityへの変換
    SFM_db = 10*sp.log10(SFM)
    tonality = SFM_db/-60.0
    floor_idx = sp.where(tonality > 1.0)
    tonality[floor_idx] = 1.0
    
    return SFM, tonality

def spectral_rolloff(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral roll-off
    
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
        spectral roll-off
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    P = sp.absolute(S)**2
    aFc = sp.zeros(P.shape[1])

    sum_P = P.sum(0)
    
    for i in xrange(P.shape[1]):
        cur_pow = 0.0
        fc = 0
        for f in xrange(P.shape[0]):
            cur_pow += P[f, i]
            if cur_pow >= 0.95 * sum_P[i]:
                fc = F[f]
                break
        aFc[i] = fc
        
    return aFc

def spectral_irregularity(x, framesize=1024, hopsize=512, fs=44100):
    """
    Calculate spectral irregularity
    
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
        spectral irregularity
    """
    S,F,T = stft(x, framesize, hopsize, fs, 'hann')
    S = sp.absolute(S)
    
    out = 0.0
    for i in xrange(S.shape[1]-1):
        out += (S[:, i+1] - S[:, i]) ** 2
        
    out /= (S**2).sum(1)
    
    return out
