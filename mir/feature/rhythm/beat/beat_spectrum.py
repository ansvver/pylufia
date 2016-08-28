# -*- coding: utf-8 -*-

import scipy as sp
from pylufia.signal.spectral import *
from pylufia.mir.feature.structure import *
from pylufia.mir.feature import *
from pylufia.signal.segment import *
import functools
import itertools
import time
from scipy import interpolate


def beat_spectrum_foote(input, framesize=256, hopsize=128, fs=44100, bpm=120.0, n_beat_seg=64, method='autocorr'):
    """
    Compute beat spectrum
    """
    # MFCC
    # X = mfcc(input, framesize, hopsize, fs, 40)[1:]

    # log-spectrum
    # X,F,T = stft(input, framesize, hopsize, fs, 'hann')
    # X = sp.absolute(X)
    # X = sp.log10(X)
    # X = X.T

    # Power spectrum
    X,F,T = stft(input, framesize, hopsize, fs, 'hann')
    X = sp.absolute(X)**2
    # X = X[:,1:] - X[:,:-1]
    X = X.T

    X = normalize_time_axis_by_bpm(X, framesize, hopsize, fs, bpm, n_beat_seg)

    # similarity matrix
    t1 = time.clock()
    S = similarity_matrix_cy(X, X, 'cos')
    t2 = time.clock()
    print( 'SM computation: {0}'.format(t2-t1) )

    # l_max = int((len(input)/2 - framesize) / hopsize) + 1
    l_max = S.shape[0]/2
    
    if method == 'diag':
        B = sp.zeros(l_max)
        # for l in xrange(l_max):
            # for k in xrange(S.shape[0]-l_max):
        for l,k in itertools.product(xrange(l_max), xrange(S.shape[0]-l_max)):
            B[l] += S[k,k+l]
    elif method == 'autocorr':
        t1 = time.clock()
        B = sp.zeros((l_max,l_max), dtype=sp.double)
        r_ed,c_ed = S.shape[0]-l_max,S.shape[1]-l_max
        for k,l in itertools.product(xrange(l_max), xrange(l_max)):
            B[k,l] = (S[:r_ed,:c_ed] * S[k:r_ed+k,l:c_ed+l]).sum()
        B = B.sum(0)
        
        t2 = time.clock()
        print( 'beat spectrum computation: {0}'.format(t2-t1) )

    return B

def beat_spectrogram_foote(input, framesize=256, hopsize=128, fs=44100, bpm=120.0, n_beat_seg=64, method='autocorr'):
    # MFCC
    # X = mfcc(input, framesize, hopsize, fs, 20)

    # log-spectrum
    X,F,T = stft(input, framesize, hopsize, fs, 'hann')
    X = sp.absolute(X)
    X = sp.log10(X+0.000001)
    X = X.T

    X = normalize_time_axis_by_bpm(X, framesize, hopsize, fs, bpm, n_beat_seg)

    # similarity matrix
    S = similarity_matrix_cy(X, X, 'cos')

    group = 10.0 / (1/float(bpm) * 60.0 / (n_beat_seg/4))
    hop = 1.0 / (1/float(bpm) * 60.0 / (n_beat_seg/4))
    n_frames = int(sp.ceil((S.shape[0] - group) / hop))
    start_pos = 0
    end_pos = int(group)
    BS = sp.zeros( (n_frames, (end_pos-start_pos)/2) )
    for i in xrange(n_frames):
        print(i) #debug
        start_pos = int(i*hop)
        end_pos = min(hop*i+group,S.shape[0])

        cur_S = S[start_pos:end_pos, start_pos:end_pos]
        l_max = cur_S.shape[0]/2
        if method == 'diag':
            B = sp.zeros(l_max)
            # for l in xrange(l_max):
            #     for k in xrange(cur_S.shape[0]-l_max):
            for l,k in itertools.product(xrange(l_max), xrange(cur_S.shape[0]-l_max)):
                B[l] += cur_S[k,k+l]
        elif method == 'autocorr':
            B = sp.zeros((l_max,l_max))
            r_ed,c_ed = cur_S.shape[0]-l_max,cur_S.shape[1]-l_max
            # for k in xrange(l_max):
                # for l in xrange(l_max):
            for k,l in itertools.product(xrange(l_max), xrange(l_max)):
                # print 'l={0}'.format(l)
                B[k,l] = (cur_S[:r_ed,:c_ed] * cur_S[k:k+r_ed,l:l+c_ed]).sum()
            B = B.sum(0)
        BS[i] = B

    return BS

def beat_spectrogram_kurth(input, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Beat spectrogramの実装
    (Kurth, ''The Cyclic Beat Spectrum: Tempo-Related Audio Features for Time-Scale Invariant Audio Identification''のスペクトル差分を用いたもの)
    """
    alpha = 0.5
    min_bpm = 15.0
    max_bpm = 320.0
    r_len = 10.0

    ## Compute novelty curve
    X,F,T = stft(input, framesize, hopsize, fs, window)
    X = sp.absolute(X)
    dX = X[:,1:] - X[:,:-1]
    dX[sp.where(dX < 0)] = 0
    N = dX.sum(0)

    ## Apply comb filter
    p_start = sp.ceil((60.0/max_bpm * fs - framesize) / float(hopsize)).astype('int')
    p_end = sp.ceil((60.0/min_bpm * fs - framesize) / float(hopsize)).astype('int')
    # nP = p_end+1 - p_start
    nP = p_end
    nT = len(N)
    Y = sp.zeros( (nT, nP) )
    Y[:,0] = N

    t1 = time.clock()
    for p,t in itertools.product(xrange(1,nP), xrange(nT)):
        if t > p:
            Y[t,p] = (1-alpha) * N[t] + alpha * Y[t-p,p]
        else:
            Y[t,p] = N[t]
    t2 = time.clock()
    print(t2-t1)

    ## Compute beat spectrum
    t1 = time.clock()
    r = int((r_len*fs - framesize) / hopsize)
    B = sp.zeros( (nT-2*r,nP) )
    for t in xrange(r, nT-r):
        # for p in xrange(nP):
            # B[t,p] = (Y[t:t+r,p]**2).sum()
        # B[t-r] = (Y[t-r:t+r,:]**2).sum(0)
        B[t-r] = (Y[t-r:t+r,:]**2).mean(0)

    ## 各時刻フレームごとにBS値が[0,1]の範囲に収まるよう正規化
    for t in xrange(B.shape[0]):
        B[t] = (B[t] - B[t].min())
        B[t] /= B[t].max() + 0.0000001
    t2 = time.clock()
    print(t2-t1)
    
    return B

def beat_spectrogram_kurth_4beat(input, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    lag timeの次元数を一定長に正規化するbeat spectrogram
    (4拍分先までのBSをとり，その次元数をn_lagsに補間/間引きする)
    """
    B = beat_spectrogram_kurth(input, framesize, hopsize, fs, window)
    nP_4beat = sp.around((60.0/bpm * fs * 4) / hopsize).astype('int')
    B = B[:,:nP_4beat]
    B2 = sp.zeros( (B.shape[0], n_lags) )
    for t in xrange(B.shape[0]):
        intp_func = interpolate.interp1d(sp.arange(len(B[t])), B[t], kind='linear')
        lastP = len(B[t])-1
        new_p = sp.arange(0, lastP, lastP/float(n_lags))
        B2[t] = intp_func(new_p)

    return B2

def beat_spectrogram_kurth_4beat_force20sec(input, framesize=512, hopsize=256, fs=44100, window='hann', bpm=120.0, n_lags=256):
    """
    入力が20secに足りない場合は20secを超えるように繰り返してから
    Beat spectrumを計算する
    """
    n_repeats = sp.ceil(fs*20.0 / len(input)).astype('int')
    input = sp.tile(input, n_repeats)
    B = beat_spectrogram_kurth_4beat(input, framesize, hopsize, fs, window, bpm, n_lags)

    return B

def cyclic_beat_spectrum():
    pass