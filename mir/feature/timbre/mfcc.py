# -*- coding: utf-8 -*-

import scipy as sp
import scipy.signal as sp_sig
from scipy.fftpack.realtransforms import dct
import pylufia.signal.spectral as sigspe
import pylufia.mir.feature as feature


def _make_mel_filterbank(n_mel_bands, fs, framesize, min_freq, max_freq):
    """ Calculate mel-filterbank
    @param n_mel_bands number of mel bands
    @param fs sampling rate
    @param framesize frame size
    @param min_freq minimum frequency of mel filterbank
    @param max_freq maximum frequency of mel filterbank
    @return (filterbank, center frequencies)
    """
    
    min_mel = _hz2mel(min_freq)
    max_mel = _hz2mel(max_freq)
    n_freqs = int(framesize / 2) + 1 # 周波数インデックスの最大数
    reso_mel = (max_mel - min_mel) / (n_mel_bands + 1) # 周波数解像度(mel)
    mel_centers = min_mel + sp.arange(0, n_mel_bands+2) * reso_mel # 中心周波数(mel)
    fc = _mel2hz(mel_centers) # 中心周波数(Hz)
    f_lowers = fc[:-2]
    f_centers = fc[1:-1]
    f_uppers = fc[2:]
    
    # fidx_centers = sp.around(f_centers / reso_freq) # 各binの中心
    # fidx_start = sp.hstack(([0], fidx_centers[0:n_channels-1])) # 各binの開始点
    # fidx_end = sp.hstack((fidx_centers[1:n_channels], [n_bins])) # 各binの終了点
    fidx_centers = ( f_centers / float(fs/2) * n_freqs ).astype('int')
    fidx_lowers = ( f_lowers / float(fs/2) * n_freqs ).astype('int')
    fidx_uppers =  ( f_uppers / float(fs/2) * n_freqs ).astype('int')

    filterbank = sp.zeros( (n_mel_bands, n_freqs) )
    for n in range(n_mel_bands):
        inc = 1.0 / (fidx_centers[n] - fidx_lowers[n])
        idxs = sp.arange(fidx_lowers[n], fidx_centers[n])
        filterbank[n, fidx_lowers[n]:fidx_centers[n]] = (idxs - fidx_lowers[n]) * inc
        dec = 1.0 / (fidx_uppers[n] - fidx_centers[n])
        idxs = sp.arange(fidx_centers[n], fidx_uppers[n])
        filterbank[n, fidx_centers[n]:fidx_uppers[n]] = 1.0 - (idxs - fidx_centers[n]) * dec

    return filterbank, f_centers
    
def mel_spectrogram(x, framesize=1024, hopsize=512, fs=44100, window="hamming", min_freq=0, max_freq=22050, n_mel_bands=40):
    """ Calculate Mel-scale spectrogram
    @param x input signal
    @param framesize STFT frame size
    @param hopsize STFT hop size
    @param fs sampling rate
    @param window type of window function
    @param min_freq minimum frequency of mel filterbank
    @param max_freq maximum frequency of mel filterbank
    @param n_mel_bands number of bands of mel filterbank
    @return (mel spectrogram, center frequencies)
    """

    # Spectrogram
    S,F,T = sigspe.stft_amp(x, framesize, hopsize, fs, window)

    # mel-spectrum
    mel_filterbank,center_freqs = _make_mel_filterbank(n_mel_bands, fs, framesize, min_freq, max_freq)
    mel_spe = sp.dot(S.T, mel_filterbank.T)

    return mel_spe,center_freqs

def mfcc(x, framesize=1024, hopsize=512, fs=44100, window="hamming", min_freq=0, max_freq=22050, n_mel_bands=40, n_ceps=13, preemp=False):
    """ Calculate MFCC
    @param x input signal
    @param framesize STFT frame size
    @param hopsize STFT hop size
    @param fs sampling rate
    @param window type of window function
    @param min_freq minimum frequency of mel filterbank
    @param max_freq maximum frequency of mel filterbank
    @param n_mel_bands number of channels of mel filterbank
    @param n_ceps number of coefficients
    @param preemp flag for using pre-emphasis
    @return (MFCC coefficients, center frequencies)
    """

    # プリエンファシス
    if preemp:
        coef = 0.97
        xemp = _pre_emphasis(x, coef)
    else:
        xemp = x

    # mel-scale spectrogram
    mel_spe,center_freqs = mel_spectrogram(xemp, framesize, hopsize, fs, window, min_freq, max_freq, n_mel_bands)
    mel_spe = sp.log10(mel_spe+1e-10)

    # DCT (ケプストラムに変換=MFCC)
    ceps = dct(mel_spe, type=2, norm="ortho", axis=-1)[:, :n_ceps]

    # nan check & inf check
    ceps = feature.check_nan_2d(ceps)
    ceps = feature.check_inf_2d(ceps)

    return ceps,center_freqs

def delta_mfcc(mfccdata, n_delta=2):
    n_frames,n_dims = mfccdata.shape
    dmfccdata = sp.zeros( (n_frames, n_dims) )
    for t in range(n_delta, n_frames-n_delta):
        for n in range(1, n_delta+1):
            dmfccdata[t] += n * (mfccdata[t+n] - mfccdata[t-n])
    dmfccdata /= 2 * ( (sp.arange(1, n_delta+1)**2).sum() )
    return dmfccdata

def _pre_emphasis(input, coef):
    """
    Pre-emphasis for MFCC
    """
    return sp_sig.lfilter([1.0, -coef], 1, input)

def _hz2mel(f):
    """
    Hz- > Mel
    """
    return 1127.010480 * sp.log(f / 700.0 + 1)

def _mel2hz(mel):
    """
    Mel -> Hz
    """
    return 700.0 * (sp.exp(mel / 1127.010480) - 1)
