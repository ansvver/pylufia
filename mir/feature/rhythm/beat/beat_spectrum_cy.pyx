# -*- coding: utf-8 -*-

import cython
import numpy as np
cimport numpy as np
from scipy import interpolate

from ymh_mir.signal.spectral import *
from ymh_mir.mir.feature.structure import *
from ymh_mir.mir.feature.spectral import *
from ymh_mir.mir.feature.common import *
import ymh_mir.signal.filter.iir as iir
import ymh_mir.signal as signal
import ymh_mir.signal.segment as segment

import itertools
import time


def beat_spectrum_cy(x, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Beat spectrogramの実装
    (Kurth, ''The Cyclic Beat Spectrum: Tempo-Related Audio Features for Time-Scale Invariant Audio Identification''のスペクトル差分を用いたもの)
    """
    cdef double alpha = 0.5
    cdef double min_bpm = 15.0
    cdef double max_bpm = 320.0
    cdef double r_len = 10.0

    ## Compute novelty curve
    X,F,T = stft(x, framesize, hopsize, fs, window)
    X = np.absolute(X)
    dX = X[:,1:] - X[:,:-1]
    dX[sp.where(dX < 0)] = 0
    N = dX.sum(0)

    ## Apply comb filter
    p_start = np.ceil((60.0/max_bpm * fs - framesize) / float(hopsize)).astype('int')
    p_end = np.ceil((60.0/min_bpm * fs - framesize) / float(hopsize)).astype('int')
    # nP = p_end+1 - p_start
    cdef int nP = p_end
    cdef int nT = len(N)
    cdef np.ndarray[double,ndim=2] Y = np.zeros( (nT, nP), dtype=np.double )
    Y[:,0] = N

    cdef int p,t
    # for p,t in itertools.product(xrange(1,nP), xrange(nT)):
    for p in range(1,nP):
        for t in range(nT):
            if t > p:
                Y[t,p] = (1-alpha) * N[t] + alpha * Y[t-p,p]
            else:
                Y[t,p] = N[t]

    ## Compute beat spectrum
    cdef int r = int((r_len*fs - framesize) / hopsize)
    B = (Y[:2*r,:]**2).mean(0) # Y(t-r:t+r)にすべき

    ## BS値が0以上&総和1になるように正規化
    #B = (B - B.min())
    #B /= B.sum() + 1e-10
    B /= B[0]
    
    return B

def beat_spectrum_bpm_normalized_cy(x, bpm, n_lags, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Beat spectrogramの実装
    (Kurth, ''The Cyclic Beat Spectrum: Tempo-Related Audio Features for Time-Scale Invariant Audio Identification''のスペクトル差分を用いたもの)
    """
    cdef double alpha = 0.5
    cdef double min_bpm = 15.0
    cdef double max_bpm = 320.0
    cdef double r_len = 10.0
    cdef int beatunit = n_lags

    ## Compute novelty curve
    X,F,T = stft(x, framesize, hopsize, fs, window)
    X = np.absolute(X)
    Xsm = segment.normalize_time_axis_by_bpm(X.T, bpm, beatunit, framesize, hopsize, fs).T
    dXsm = Xsm[:,1:] - Xsm[:,:-1]
    dXsm[sp.where(dXsm<0)] = 0
    N = dXsm.sum(0)
    #dX = X[:,1:] - X[:,:-1]
    #dX[sp.where(dX < 0)] = 0
    #dXsm = segment.smoothByBpm(dX.T, bpm, beatunit, framesize, hopsize, fs).T
    #N = dXsm.sum(0)

    ## Apply comb filter
    n_frames_per_beat = ( ( 60.0/bpm * fs / (beatunit/4) ) ) / float(hopsize)
    p_start = np.ceil( (60.0/max_bpm * fs - framesize) / float(hopsize) / n_frames_per_beat ).astype('int')
    p_end = np.ceil( (60.0/min_bpm * fs - framesize) / float(hopsize) / n_frames_per_beat ).astype('int')
    # nP = p_end+1 - p_start
    cdef int nP = p_end
    cdef int nT = len(N)
    cdef np.ndarray[double,ndim=2] Y = np.zeros( (nT, nP), dtype=np.double )
    Y[:,0] = N

    cdef int p,t
    # for p,t in itertools.product(xrange(1,nP), xrange(nT)):
    for p in range(1,nP):
        for t in range(nT):
            if t > p:
                Y[t,p] = (1-alpha) * N[t] + alpha * Y[t-p,p]
            else:
                Y[t,p] = N[t]

    ## Compute beat spectrum
    #cdef int r = int( (r_len*fs - framesize) / hopsize )
    cdef int r = int( (r_len*fs - framesize) / float(hopsize) / n_frames_per_beat )
    B = (Y[:2*r,:]**2).mean(0) # Y(t-r:t+r)にすべき

    ## BS値が0以上&総和1になるように正規化
    #B = (B - B.min())
    #B /= B.sum() + 1e-10
    B /= B[0]
    B = B[:beatunit]
    
    return B

def beat_spectrum_bpm_normalized_force20sec_cy(x, bpm, n_lags, framesize=512, hopsize=256, fs=44100, window='hann'):
    x = _repeat_wave_to_20sec(x, fs)
    B = beat_spectrum_bpm_normalized_cy(x, bpm, n_lags, framesize, hopsize, fs, window)
    return B

def beat_spectrum_force20sec_cy(x, framesize=512, hopsize=256, fs=44100, window='hann'):
    x = _repeat_wave_to_20sec(x, fs)
    B = beat_spectrum_cy(x, framesize, hopsize, fs, window)
    return B

def beat_spectrum_4beat_cy(x, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    lag timeの次元数を一定長に正規化するbeat spectrogram
    (4拍分先までのBSをとり，その次元数をn_lagsに補間/間引きする)
    """
    B = beat_spectrum_cy(x, framesize, hopsize, fs, window)
    nP_4beat = np.around( (60.0/bpm * fs * 4) / hopsize ).astype('int')
    B = B[:nP_4beat]

    intp_func = interpolate.interp1d( np.arange(len(B) ), B, kind='linear')
    lastP = len(B)-1
    new_p = np.arange( 0, lastP, lastP/float(n_lags) )
    B2 = intp_func(new_p)
    #B2 /= B2.sum() + 1e-10
    B2 /= B2[0]

    return B2

def beat_spectrum_4beat_force20sec_cy(x, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    入力が20secに足りない場合は20secを超えるように繰り返してから
    Beat spectrumを計算する
    """
    x = _repeat_wave_to_20sec(x, fs)
    B = beat_spectrum_4beat_cy(x, framesize, hopsize, fs, window, bpm, n_lags)

    return B

def beat_spectrogram_cy(x, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Beat spectrogramの実装
    (Kurth, ''The Cyclic Beat Spectrum: Tempo-Related Audio Features for Time-Scale Invariant Audio Identification''のスペクトル差分を用いたもの)
    """
    cdef double alpha = 0.5
    cdef double min_bpm = 15.0
    cdef double max_bpm = 320.0
    cdef double r_len = 10.0

    ## Compute novelty curve
    X,F,T = stft(x, framesize, hopsize, fs, window)
    X = np.absolute(X)
    dX = X[:,1:] - X[:,:-1]
    dX[sp.where(dX < 0)] = 0
    N = dX.sum(0)

    ## Apply comb filter
    p_start = np.ceil((60.0/max_bpm * fs - framesize) / float(hopsize)).astype('int')
    p_end = np.ceil((60.0/min_bpm * fs - framesize) / float(hopsize)).astype('int')
    # nP = p_end+1 - p_start
    cdef int nP = p_end
    cdef int nT = len(N)
    cdef np.ndarray[double,ndim=2] Y = np.zeros( (nT, nP), dtype=np.double )
    Y[:,0] = N

    cdef int p,t
    # for p,t in itertools.product(xrange(1,nP), xrange(nT)):
    for p in range(1,nP):
        for t in range(nT):
            if t > p:
                Y[t,p] = (1-alpha) * N[t] + alpha * Y[t-p,p]
            else:
                Y[t,p] = N[t]

    ## Compute beat spectrum
    cdef int r = int((r_len*fs - framesize) / hopsize)
    # cdef np.ndarray[double,ndim=2] B = np.zeros( (nT-2*r,nP), dtype=np.double )
    B = np.zeros( (nT-2*r,nP), dtype=np.double )
    for t in range(r, nT-r):
        # for p in xrange(nP):
            # B[t,p] = (Y[t:t+r,p]**2).sum()
        # B[t-r] = (Y[t-r:t+r,:]**2).sum(0)
        B[t-r] = (Y[t-r:t+r,:]**2).mean(0)

    ## 各時刻フレームごとにBS値が[0,1]の範囲に収まるよう正規化
    cdef int nB = B.shape[0]
    for t in range(nB):
        #B[t] = (B[t] - B[t].min())
        #B[t] /= B[t].sum() + 1e-6
        B[t] /= B[t,0]
    
    return B

def beat_spectrogram_force20sec_cy(x, framesize=512, hopsize=256, fs=44100, window='hann'):
    x = _repeat_wave_to_20sec(x, fs)
    B = beat_spectrogram_cy(x, framesize, hopsize, fs, window)
    return B

def beat_spectrogram_4beat_cy(x, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    lag timeの次元数を一定長に正規化するbeat spectrogram
    (4拍分先までのBSをとり，その次元数をn_lagsに補間/間引きする)
    """
    B = beat_spectrogram_cy(x, framesize, hopsize, fs, window)
    nP_4beat = np.around( (60.0/bpm * fs * 4) / hopsize ).astype('int')
    B = B[:,:nP_4beat]
    B2 = np.zeros( (B.shape[0], n_lags), dtype=np.double )
    cdef int t
    for t in xrange(B.shape[0]):
        intp_func = interpolate.interp1d(np.arange(len(B[t])), B[t], kind='linear')
        lastP = len(B[t])-1
        new_p = np.arange(0, lastP, lastP/float(n_lags))
        B2[t] = intp_func(new_p)

    return B2

def beat_spectrogram_4beat_force20sec_cy(x, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    入力が20secに足りない場合は20secを超えるように繰り返してから
    Beat spectrumを計算する
    """
    x = _repeat_wave_to_20sec(x, fs)
    B = beat_spectrogram_4beat_cy(x, framesize, hopsize, fs, window, bpm, n_lags)

    return B

def find_beat_spectrum_period(BS):
    """
    ビートスペクトルから繰り返し周期を見つける
    """
    sigma = 3

    # peak picking
    gain = filter.calcEMA(BS, 32)
    peak = signal.peak(BS, gain)
    peak_pos = np.where(peak > 0)[0]

    # 先頭1/3区間のピークを抽出(繰り返し周期候補)
    peak_cands = peak[:int( len(BS)*0.55 )]
    peak_cands_pos = np.where(peak_cands > 0)[0]
    peak_cands_pos = peak_cands_pos[1:]
    
    accum_mean_powers = np.zeros( len(peak_cands_pos) )
    for i,p in enumerate(peak_cands_pos):
        p_harmonics = sp.arange(p,peak_pos[-1],p)
        target_peak_idx_list = [0]
        for ph in p_harmonics:
            idx = sp.where(sp.absolute( peak_pos - ph ) <= sigma)[0]
            if len(idx) > 0:
                target_peak_idx_list.append(peak_pos[idx[0]])
        accum_mean_powers[i] = peak[target_peak_idx_list].mean()

    maxidx = sp.argmax(accum_mean_powers)
    
    return peak_cands_pos[maxidx]

def decimate_beat_spectrum(BS, n_dim=64):
    intp_func = interpolate.interp1d(np.arange(len(BS)), BS, kind='linear')
    lastP = len(BS)-1
    new_p = np.arange(0, lastP, lastP/float(n_dim))
    BS_dec = intp_func(new_p)
    return BS_dec


""" helper functions """

def _repeat_wave_to_20sec(x, fs):
    #n_repeats = np.ceil(fs*32.0 / len(x)).astype('int')
    n_repeats = np.ceil(fs*10.0 / len(x)).astype('int')
    y = np.tile(x, n_repeats)
    return y
