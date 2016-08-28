# -*- coding: utf-8 -*-

import scipy as sp
from pylufialib.signal.spectral import *


def cyclic_tempogram(input, framesize=1024, hopsize=512, fs=44100):
    """
    Compute Cyclic tempogram
    """
    # log-spectrum差分
    X,freq,time = stft(input, framesize, hopsize, fs)
    X = sp.absolute(X)
    log_spe = sp.log10(1+X)
    d_log_spe = log_spe[:, 1:] - log_spe[:, 0:-1]
    negative_idxs = sp.where(d_log_spe < 0)
    d_log_spe[negative_idxs] = 0
    delta = d_log_spe.sum(0)

    # Fourier tempogram
    F,time,freq = stft(delta, 512, 1, fs, 'hann')
    bpm = sp.arange(60, 480)
    fourier_tempogram = sp.absolute(F[bpm/60, :])
    
    # Autocorrelation tempogram
    

    return fourier_tempogram, delta

def _autocorrelation_tempogram(delta, win_len=1024):
    window = sp.sig.get_window(win_len*2, 'hann')
