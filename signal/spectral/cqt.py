# -*- coding: utf-8 -*-

"""
cqt.py

CQTの実装
"""

import scipy as sp
from .fft import *


def make_cqt_kernels(framesize, fs, n_cq, n_per_semitone, fmin, n_wavs=30):
    """
    CQTのカーネル関数作成
    """
    wmin = 2*sp.pi*fmin/fs
    n_win = float(float(n_wavs) * fs / float(1+fmin) * wmin)
    
    freqs = sp.zeros(n_cq)
    kernels = sp.zeros( (n_cq, int(framesize/2)) )
    scaling = fs/(2*sp.pi)
    
    for kcq in range(n_cq):
        wkcq = (2**(1/float(12*n_per_semitone)))**kcq * wmin
        n = sp.arange(framesize)
        cur_win = _window(n, framesize, wkcq, n_win)
        r = cur_win * sp.exp(1.0j*wkcq*(n - framesize/2))
        freqs[kcq] = wkcq * scaling
        kernels[kcq,:] = sp.real(fft(r, framesize))[:int(framesize/2)]
        
    return kernels, freqs
    
def cqt(x, framesize=8192, hopsize=1024, fs=44100, window="hann", n_cq=88, n_per_semitone=1, fmin=60.0):
    """
    CQT計算
    
    kernelとスペクトログラムの乗算を行列演算で一気にやってしまえば高速化されるが
    大量のメモリを食うようになる．
    """
    x_zpad = sp.r_[sp.zeros(framesize/2), x, sp.zeros(framesize/2)]
    
    kernels,freqs = make_cqt_kernels(framesize, fs, n_cq, n_per_semitone, fmin)
    kernels = kernels.astype(float)
    
    X,F,T = stft(x_zpad, framesize, hopsize, fs, window)
    X = X.astype(complex).T
    
    CQ = sp.zeros( (X.shape[0], n_cq), dtype=complex)
    try:
        CQ[:] = sp.dot( kernels, X[:,:kernels.shape[1]].T ).T
    except MemoryError:
        print("catch MemoryError.")
        for k in xrange(kernels.shape[1]):
            CQ += kernels[sp.newaxis,:,k] * X[:,sp.newaxis,k]
    
    CQ = sp.absolute(CQ)
    CQ = CQ.astype(float).T
    # CQ = CQ[:]
    CQ = sp.array(CQ, order="C")
    
    return CQ, freqs, T


""" helper functions """

def _hamming(n, N, alpha=25/46.0):
    """
    ハミング窓
    """
    output = alpha - (1.0-alpha)*sp.cos(2*sp.pi*n/N)
    output[n<0] = 0
    output[n>=N] = 0
    return output
    
def _window(n, N, wkcq, n_win=2048):
    """
    CQT用窓関数作成
    """
    Nkcq = n_win/float(wkcq)
    _n = n - (N/2.0 - Nkcq/2.0)
    win = _hamming(_n, Nkcq)
    return win/win.sum()
