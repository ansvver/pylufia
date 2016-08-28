# -*- coding: utf-8 -*-

import scipy as sp
from scipy.fftpack.realtransforms import dct
import pylufia.signal.spectral as sigspe
import pylufia.mir.feature as feature


def _make_mel_filterbank(n_ceps, fs, framesize, freq_max):
    """
    Calculate mel-filterbank
    
    Parameters:
      freq_max: int
        max frequency
        周波数軸上のindex:framesize-1での周波数値がこの値になるようにすること
      framesize: int
        frame size
      nChannels: int
        number of channels of mel-filterbank
    
    Returns:
      filterBank: ndarray
        mel-filterbank
      f_centers: ndarray
        center frequencies of mel-filterbank
    """
    # freq_max = fs / 2 # ナイキスト周波数
    mel_max = _hz2mel(freq_max) # ナイキスト周波数 (mel-scale)
    n_freqs = int(framesize / 2) + 1 # 周波数インデックスの最大数
    n_bins = int(n_freqs * freq_max*2 / float(fs))
    # reso_freq = freq_max*2 / float(framesize) # 周波数解像度
    reso_freq = freq_max / float(n_bins) # 周波数解像度
    reso_mel = mel_max / (n_ceps + 1) # 周波数解像度(mel)
    mel_centers = sp.arange(0, n_ceps+2) * reso_mel # 中心周波数(mel)
    fc = _mel2hz(mel_centers) # 中心周波数(Hz)
    # fc = sp.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1149, 1320,
                # 1516, 1741, 2000, 2297, 2639, 3031, 3482, 4000, 4595, 5278, 6063, 6964])
    f_lowers = fc[:-2]
    f_centers = fc[1:-1]
    f_uppers = fc[2:]
    
    # fidx_centers = sp.around(f_centers / reso_freq) # 各binの中心
    # fidx_start = sp.hstack(([0], fidx_centers[0:n_channels-1])) # 各binの開始点
    # fidx_end = sp.hstack((fidx_centers[1:n_channels], [n_bins])) # 各binの終了点
    fidx_centers = ( f_centers / float(fs/2) * n_freqs ).astype('int')
    fidx_lowers = ( f_lowers / float(fs/2) * n_freqs ).astype('int')
    fidx_uppers =  ( f_uppers / float(fs/2) * n_freqs ).astype('int')

    filterbank = sp.zeros( (n_ceps, n_freqs) )
    for n in range(n_ceps):
        inc = 1.0 / (fidx_centers[n] - fidx_lowers[n])
        idxs = sp.arange(fidx_lowers[n], fidx_centers[n])
        filterbank[n, fidx_lowers[n]:fidx_centers[n]] = (idxs - fidx_lowers[n]) * inc
        dec = 1.0 / (fidx_uppers[n] - fidx_centers[n])
        idxs = sp.arange(fidx_centers[n], fidx_uppers[n])
        filterbank[n, fidx_centers[n]:fidx_uppers[n]] = 1.0 - (idxs - fidx_centers[n]) * dec

    return filterbank, f_centers
    
def mel_spectrogram(x, framesize=1024, hopsize=512, fs=44100, window="hamming", freq_max=22050, n_ceps=22):
    """
    Calculate Mel-scale spectrogram
    
    Parameters:
      x: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
    
    Returns:
      result: ndarray
        mel-scaled spectrogram
    """

    # Spectrogram
    S,F,T = sigspe.stft(x, framesize, hopsize, fs, window)
    S = sp.absolute(S)

    # mel-spectrum
    mel_filterbank,center_freqs = _make_mel_filterbank(n_ceps, fs, framesize, freq_max)
    mel_spe = sp.dot(S.T, mel_filterbank.T)

    return mel_spe,center_freqs

def mfcc(x, framesize=1024, hopsize=512, fs=44100, window="hamming", max_freq=22050, n_ceps=13, preemp=False):
    """
    Calculate MFCC
    
    Parameters:
      x: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      fs: int
        samplingrate
      n_ceps: int
        number of dimensions of MFCC
    
    Returns:
      result: ndarray
        mfcc
    """
    # プリエンファシス
    if preemp:
        coef = 0.97
        xemp = _pre_emphasis(x, coef)
    else:
        xemp = x

    # mel-scale spectrogram
    mel_spe,center_freqs = mel_spectrogram(xemp, framesize, hopsize, fs, window, max_freq, n_ceps)
    mel_spe = sp.log(mel_spe+1e-10)

    # DCT (ケプストラムに変換=MFCC)
    ceps = dct(mel_spe, type=2, norm="ortho", axis=-1)[:, :n_ceps]

    # nan check & inf check
    ceps = feature.check_nan_2d(ceps)
    ceps = feature.check_inf_2d(ceps)

    return ceps,center_freqs

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
