# -*- coding: utf-8 -*-

"""
====================================================================
fft.py

FFT,STFTの実装
====================================================================
"""

import scipy as sp
import scipy.signal as sp_sig
import pylufia.signal.segment as segment


def fft(x, fftsize=512):
    """
    Apply FFT
    
    Parameters:
      inData: ndarray
        input signal
      fftsize: int
        size of FFT
    
    Returns:
      fftData: ndarray
        result of FFT
    """
    X = sp.fftpack.fft(x, fftsize)
    n_freq = int(fftsize/2)+1
    # X = X.T[:n_freq].T / (fftsize/2.0)
    X = X.T[:n_freq].T
    # X = sp.array(X, order="C")
    # if len(input.shape) == 1:
        # fftData = fftData[0:nFreq]
    # else:
        # fftData = fftData[:, 0:nFreq]
    
    return X

def ifft(X, fftsize):
    """
    Apply IFFT
    
    Parameters:
      X: ndarray
        input signal
      fftsize: int
        size of IFFT
    
    Returns:
      result: ndarray
        result of IFFT
    """
    rec_x = sp.fftpack.ifft(X, fftsize)
    output = rec_x

    return output
    
def irfft(X, fftsize):
    rec_x = sp.fftpack.irfft(X, fftsize)
    output = rec_x

    return output
    
def fft2d(X, shape=None):
    """
    Apply 2D-FFT
    
    Parameters:
      inData: ndarray
        input signal
      shape: 2d tuple
        shape of 2D-FFT result
    
    Results:
      result: ndarray
        result of 2D-FFT
    """
    return sp.fftpack.fft2(X, shape=shape)

def stft(x, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Compute spectrogram
    
    Parameters:
      x: ndarray
        input signal
      framesize: int
        framesize of STFT
      hopsize: int
        hopsize: of STFT
      window: string
        type of window function
      fs: int
        samplingrate
    
    Returns:
      S: ndarray
        result of STFT
      F: ndarray
        frequency map to fft index
      T: ndarray
        time map to frame index
    """
    n_frames = int( sp.ceil( (len(x) - framesize) / float(hopsize) ) ) + 1
    n_freqs = sp.ceil( (framesize + 1) / 2.0)
    times = sp.arange(0, len(x), hopsize)
    freqs = sp.arange(0, n_freqs) * fs / float(framesize)

    # 全フレーム分の波形データをあらかじめ作成しておく計算方法(メモリ使用量大)
    # framed_x = segment.make_framed_data(x, framesize, hopsize, window)
    # X = fft(framed_x, framesize)

    # 1フレームずつ波形を切り出し逐次計算(メモリ使用量小)
    # memo: Xは(dim,frame)で直接格納するより(frame,dim)で格納して後で転置する方が速い
    X = sp.zeros( (n_frames, int(framesize/2)+1), dtype=complex )
    cur_start_pos = 0
    cur_end_pos = framesize
    inbuf = sp.zeros(framesize, dtype=float)
    win_func = sp_sig.get_window(window, framesize)
    for i in range(n_frames):
        inbuf[:cur_end_pos-cur_start_pos] = x[cur_start_pos:cur_end_pos]
        inbuf = inbuf * win_func
        X[i,:] = fft(inbuf, framesize)
        cur_start_pos += hopsize
        cur_end_pos = min( cur_end_pos+hopsize, len(x) )
    X = X.T
    # X = sp.array(X, order="C")

    return X, freqs, times
    
def istft(X, framesize=512, hopsize=256, window='hann'):
    """
    逆短時間フーリエ変換
    """
    n_frames = X.shape[1]
    recovered_x = sp.zeros((n_frames-1)*hopsize+framesize)
    win_func = sp.signal.get_window(window, framesize)

    for i, _X in enumerate(X.T):
        start_frame = i * hopsize
        end_frame = min(start_frame+framesize, len(recovered_x))
        recX = sp.concatenate( (_X, sp.flipud(_X[1:-1].conjugate())) )
        ifft_data = ifft(recX, framesize)
        recovered_x[start_frame:end_frame] += sp.real((ifft_data * win_func)[:end_frame-start_frame])
        
    recovered_x *= hopsize / float(framesize)

    return recovered_x
    
def rep_istft(X, framesize, hopsize, fs, window, n_iter=100):
    """
    反復STFT法によるスペクトログラムからの信号復元
    """
    init_phase = sp.zeros(X.shape, dtype=complex)
    for f in range(init_phase.shape[0]):
        for t in range(init_phase.shape[1]):
            init_phase[f,t] = sp.exp(1j*(f*t + sp.random.random()))
    # X2 = X.copy()
    # Xmag = X.view()
    S = X * init_phase
    # S = X.copy()
    
    for i in range(n_iter):
        s = istft(S, framesize, hopsize, window).real
        S = stft(s, framesize, hopsize, fs, window)[0]
        S = X * S / (1e-6 + sp.absolute(S))
    
    return istft(S, framesize, hopsize, window).real
    
