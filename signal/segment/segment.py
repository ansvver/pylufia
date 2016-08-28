# -*- coding: utf-8 -*-

import scipy as sp
import scipy.signal as sp_sig
import scipy.interpolate as interpolate
import pylufia.mir.common as common
import copy


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
    framed_data = sp.zeros((n_frames, framesize))
    # win_func = sig.get_window(window, framesize)

    for i in range(n_frames):
        win_func = sp_sig.get_window(window, cur_end_pos-cur_start_pos)
        framed_data[i,:cur_end_pos-cur_start_pos] = input[cur_start_pos:cur_end_pos] * win_func
        cur_start_pos += hopsize
        cur_end_pos = min(cur_end_pos+hopsize, len(input))

    return framed_data
    
def make_bar_segmented_data(input, beat_pos_arr):
    bar_pos_arr = beat_pos_arr[::4]
    barseg_wav = []
    for i in range(len(bar_pos_arr)):
        st = bar_pos_arr[i]
        if i == len(bar_pos_arr) - 1:
            ed = len(input)
        else:
            ed = bar_pos_arr[i+1]
        cur_wav = input[st:ed]
        if len(cur_wav) > 0:
            barseg_wav.append(cur_wav)

    return barseg_wav
    
def normalize_time_axis_by_beat_old(X, framesize=1024, hopsize=512, fs=44100, bpm=120, segsize=16):
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
    n_frames_per_beatseg = ( ( 60.0/bpm * fs / (segsize/4) ) ) / hopsize
    n_beatseg = int(sp.ceil(X.shape[0] / n_frames_per_beatseg))
    start = 0.0
    end = 0
    
    if len(X.shape) == 1:
        n_dim = 1
    else:
        n_dim = X.shape[1]
        
    smoothed = sp.zeros( (n_beatseg, n_dim) )
    for seg in range(n_beatseg):
        end = min(X.shape[0], int(start+n_frames_per_beatseg))
        smoothed[seg] = sp.mean(X[int(start):end], axis=0)
        start += n_frames_per_beatseg
        
    return smoothed

def normalize_time_axis_by_beat(X, beat_pos_arr, beatunit=16, framesize=1024, hopsize=512, fs=44100):
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
    n_frames = X.shape[0]
    if len(X.shape) == 1:
        n_dims = 1
    else:
        n_dims = X.shape[1]
    n_smp = framesize + hopsize * (X.shape[0] - 1)
    
    interp_beat_pos_arr = _interpSubBeat(beat_pos_arr, beatunit, n_smp)
    beat_pos_in_frm = ( interp_beat_pos_arr / float(hopsize) ).astype('int')
    n_beatseg = len(beat_pos_in_frm)

    smoothed = sp.zeros( (n_beatseg, n_dims) )
    for b in range(n_beatseg):
        st = min(beat_pos_in_frm[b],n_frames-1)
        if b < n_beatseg-1:
            ed = beat_pos_in_frm[b+1]
        else:
            ed = n_frames

        if st == ed:
            smoothed[b] = X[st]
        else:
            smoothed[b] = X[st:ed].mean(0)

    return smoothed

def normalize_time_axis_by_bpm(X, bpm=120, beatunit=16, framesize=1024, hopsize=512, fs=44100):
    """
    Segment and smooth feature per 1/n beat (specify by BPM)
    """
    n_smp = framesize + hopsize * (X.shape[0] - 1)
    beat_pos_arr = common.bpm_to_beat_pos(bpm, n_smp, fs)
    return normalize_time_axis_by_beat(X, beat_pos_arr, beatunit, framesize, hopsize, fs)

def transform_time_axis_to_one_bar(X, beatunit):
    """
    1小節単位に特徴量をまとめる
    """
    n_frames_old,n_dims_old = X.shape
    n_frames_new = int( n_frames_old / beatunit )
    n_frames_old_traverse = n_frames_old - n_frames_old % beatunit
    n_dims_new = n_dims_old * beatunit

    new_X = sp.zeros( (n_frames_new,n_dims_new) )
    for t in range(n_frames_old_traverse):
        s = t % beatunit
        cur_seg = int(t / beatunit)
        new_X[cur_seg,s*n_dims_old:(s+1)*n_dims_old] = X[t]

    return new_X


""" helper functions """

# def _bpmToBeatPos(bpm, length, fs=44100):
#     n_smp_beat = 60.0/float(bpm) * fs
#     beat_pos_in_smp = sp.arange(0, length, n_smp_beat)
#     return beat_pos_in_smp

def _interp_subbeat(beat_pos_arr, beatunit, length):
    """
    Interpolate sub-beat to beat position array
    """
    if beat_pos_arr[-1] == length:
        _beat_pos_arr = copy.deepcopy(beat_pos_arr)
    else:
        _beat_pos_arr = sp.zeros(len(beat_pos_arr)+1)
        _beat_pos_arr[:len(beat_pos_arr)] = beat_pos_arr
        _beat_pos_arr[-1] = length
    interp_func = interpolate.interp1d(sp.arange( len(_beat_pos_arr) ), _beat_pos_arr, kind='linear')
    n_beats = len(_beat_pos_arr)
    new_t = sp.arange( 0, n_beats-1+1e-10, 4/float(beatunit) )
    tlen = min( 4*beatunit,len(new_t) )
    new_t = new_t[:tlen]
    interp_beat_pos_arr = interp_func(new_t)
    return interp_beat_pos_arr
