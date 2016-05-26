# -*- coding: utf-8 -*-

"""
@file segment.py
@brief basic segmentation functions
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.signal as sp_sig
import matplotlib.pyplot as pp


def make_framed_data(input, framesize=512, hopsize=256, window='boxcar'):
    """
    Slice waveform per frame
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize
      hopsize: int
        hopsize
      window: string
        type of window function
        
    Returns:
      result: ndarray
        matrix which contains frame-cutted signal
    """

    cur_start_pos = 0
    cur_end_pos = framesize
    n_frames = int(sp.ceil((len(input) - framesize) / float(hopsize))) + 1
    framed_data = sp.zeros( (n_frames, framesize) )
    # win_func = sig.get_window(window, framesize)

    for i in xrange(n_frames):
        win_func = sp_sig.get_window(window, cur_end_pos-cur_start_pos)
        # pp.plot(win_func)
        # pp.show()
        framed_data[i,:cur_end_pos-cur_start_pos] = input[cur_start_pos:cur_end_pos] * win_func
        cur_start_pos += hopsize
        cur_end_pos = min(cur_end_pos+hopsize, len(input))

    return framed_data
    
def smooth_by_beat(X, framesize=1024, hopsize=512, fs=44100, bpm=120, segsize=16):
    """
    Segment and smooth feature per 1/n beat
    
    Parameters:
      X: ndarray
        input feature data
      framesize: int
        framesize of feature analysis
      hopsize: int
        hopsize of feature analysis
      fs: int
        samplingrate
      bpm: float
        BPM of input
      nBeat: int
        segment unit(n of 1/n beat)
    
    Returns:
      result: ndarray
        1/n beat segmented feature
    """
    #group = (fs * 60.0) / (bpm * nBeat/4 * framesize * 0.5) # 合ってるのかこれ？
    group = (60.0/bpm * fs / (segsize/4)) / hopsize
    n_seg = int(sp.ceil(X.shape[0] / group))
    start = 0.0
    end = 0
    
    if len(X.shape) == 1:
        n_dim = 1
    else:
        n_dim = X.shape[1]
        
    smoothed = sp.zeros( (n_seg, n_dim) )
    for seg in xrange(n_seg):
        end = min(X.shape[0], int(start+group))
        smoothed[seg] = sp.median(X[int(start):end], axis=0)
        # smoothed[seg] = sp.mean(X[int(start):end], axis=0)
        # smoothed[seg] = sp.amax(X[int(start):end], axis=0)
        start += group
        
    return smoothed
    
def smooth_by_beat_with_time(X, T, fs=44100, bpm=120, segsize=16):
    """
    ビート単位での特徴量時間方向平滑化 (時間軸のサンプル数を別ベクトルTで指定)
    """
    n_smp_per_beat = int(60.0/bpm * fs / (segsize/4))
    n_seg = int(T[-1]/n_smp_per_beat)
    if len(X.shape) == 1:
        n_dim = 1
    else:
        n_dim = X.shape[1]
        
    smoothed = sp.zeros( (n_seg, n_dim) )
    for seg in xrange(n_seg):
        start = sp.where(T >= seg*n_smp_per_beat)[0][0]
        end = sp.where(T >= (seg+1)*n_smp_per_beat)[0][0]
        smoothed[seg] = sp.mean(X[start:end], axis=0)
        
    return smoothed