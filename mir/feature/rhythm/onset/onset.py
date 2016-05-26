# -*- coding: utf-8 -*-

"""
@file onset.py
@brief onset detection
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from plifia.mir.feature.timbre import *
from plifia.signal import *
from plifia.signal.moving_average import moving_average_exp_cy
from plifia.signal.spectral import *


def onset(input, framesize=1024, hopsize=512, fs=44100, method='mfcc'):
    """
    Compute onset
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize of onset analysis
      hopsize: int
        hopsize of onset analysis
      fs: int
        samplingrate
      method: string
        onset detection method (mfcc|logspe|sol|klapuri)
    
    Results:
      result: ndarray
        onset detection result array
    """
    if method == 'mfcc':
        return _onset_by_mfcc(input, framesize, hopsize, fs)
    elif method == 'logspe':
        return _onset_by_logspe(input, framesize, hopsize, fs)
    elif method == 'sol':
        return _onset_by_sol(input, framesize, hopsize, 'hann', fs)
    elif method == 'klapuri':
        return _onset_by_klapuri(input, framesize, hopsize, fs)
    else:
        return None

""" local functions """

def _onset_by_logspe(input, framesize=1024, hopsize=180, fs=44100):
    S,F,T = stft(input, framesize, hopsize, fs, "hamming")
    logS = sp.log10(sp.absolute(S)+1e-10)
        
    diff_logS = logS[:,1:] - logS[:,:-1]
    diff_logS[diff_logS<0] = 0

    odf = diff_logS.sum(0)
    thresholds = moving_average_exp_cy(odf, 32) * 2
    
    onset_peak_idx, onset_peak_data = _findOnsetAndPeak(odf, thresholds)
    
    return onset_peak_idx, onset_peak_data, odf, thresholds
    
def _findOnsetAndPeak(input, thresholds):
    """
    Find onset and peak from wave data
    
    Parameters:
    
    Returns:
    
    """
    onset_peak_idx = []
    onset_peak_data = []
    for i in xrange(len(input)-1):
        onset_idx = 0
        peak_idx = 0
        onset_val = 0.0
        peak_val = 0.0
        if input[i] - input[i-1] > 0 and input[i+1] - input[i] < 0:
            if input[i] >= thresholds[i]:
                peak_idx = i
                peak_val = input[i]
                for j in xrange(i, 0, -1):
                    if input[j] <= thresholds[j]:
                        onset_idx = j
                        onset_val = input[j]
                        onset_peak_idx.append(sp.array([onset_idx, peak_idx]))
                        onset_peak_data.append(sp.array([onset_val, peak_val]))
                        break
                
    onset_peak_idx = sp.array(onset_peak_idx)
    onset_peak_data = sp.array(onset_peak_data)
                
    return onset_peak_idx, onset_peak_data

def _onset_by_mfcc(input, framesize=1024, hopsize=512, fs=44100):
    """
    Onset detection by delta MFCC
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize of MFCC
      hopsize: int
        hopsize of MFCC
      fs: int
        samplingrate
    
    Returns:
      onsetidx: ndarray
        index of onsets
      onsetdata: ndarray
        onset amplitudes
      odf: ndarray
        onset detection function
    """
    th = 0.6 # onset検出のodf閾値 (移動平均あり)
    #th = 1.0 # onset検出のodf閾値 (移動平均なし)

    input_zero_padded = sp.r_[sp.zeros(framesize+hopsize*1), input] # 先頭onset検出のためダミーデータ挿入
    mfcc_data = mfcc(input_zero_padded, framesize, hopsize, fs, 13)[:,1:]
    d_mfcc_data = mfcc_data[1:] - mfcc_data[0:mfcc_data.shape[0]-1]
    odf = d_mfcc_data.sum(1)
    # odf = calcEMA(odf, 5)
    
    # peakを捉えて直前のdipをonsetとする
    peak_data = _peakpick(odf, th)
    peak_idx = sp.where(peak_data > 0)[0]
    
    onset_idx = []
    for cur_idx in peak_idx:
        for i in range(cur_idx, -1, -1):
            if (odf[i]-odf[i-1] <= 0 and odf[i+1]-odf[i] > 0) or i == 0:
                onset_idx.append(i)
                break

    onset_data = odf[onset_idx]
    
    return onset_idx, onset_data, odf
    
def _onset_by_sol(input, framesize=512, hopsize=256, fs=44100, window='hann'):
    """
    Onset detection by SOL slicer algorithm
    
    Parameters:
      inData: ndarray
        input signal
      framesize: int
        framesize of flux analysis
      hopsize: int
        hopsize of flux analysis
      window: string
        type of window function
      fs: int
        samplingrate
    
    Returns:
      onsetdata: ndarray
        onset amplitudes
      odf: ndarray
        onset detection function
      filData: ndarray
        filtered input signal
    """

    # HPSSで非調波成分のみ取り出すと前処理フィルタなしでもわりあい上手くいく．
    # が，HPSSなしだと持続音が取りきれずonsetが抽出できない．
    w = 3
    m = 3
    alpha = 0.5
    flag1 = 0
    flag2 = 0
    flag3 = 0

    # 前処理フィルタ
    bef_b, bef_a = iirCalcCoefFs(2.0, 200.0, -6.0, 'bpf', fs)
    hpf_b, hpf_a = iirCalcCoefFs(2.0, 4000.0, 3.0, 'hpf2', fs)

    fil_data = iirApply(input, bef_b, bef_a)
    fil_data = iirApply(fil_data, hpf_b, hpf_a)

    odf = flux(fil_data, framesize, hopsize, window, fs)
    odf = moving_average_exp_cy(odf, 10)

    onset_data = []
    g = _calcOnsetThreshold(odf, alpha) # Threshold function

    sigma = sp.mean(odf) / 12.0

    # Onset Detection Revisited (S. Dixon, 2006)の#marupakuri．
    # オリジナルの判定法に差し替える．
    for i in xrange(len(odf)):
        flag1 = 0
        flag2 = 0
        flag3 = 0

        rng_start = max(i-w, 0)
        rng_end = min(i+w+1, len(odf))
        max_idx = sp.argmax(odf[rng_start:rng_end])
        if i == rng_start + max_idx:
            flag1 = 1

        rng_start = max(i-m*w, 0)
        rng_end = min(i+w+1, len(odf))
        if odf[i] >= (sum(odf[rng_start:rng_end]) / (m*w + w + 1) + sigma):
            flag2 = 1

        if odf[i] >= g[i-1]:
            flag3 = 1

        onset_data.append(1.0 * flag1 * flag2 * flag3)

    onset_data = sp.array(onset_data)

    return onset_data, odf, fil_data

def _onset_by_klapuri(input, framesize=4410, hopsize=512, fs=44100):
    """
    Onset detection by Klapuri algorithm(1999)
    
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
    """

    decim_factor = 128
    fbank_coefs = _makeFilterbank(fs)

    n_dim = len(fbank_coefs)
    fil_data = sp.zeros((n_dim, len(input)))
    #envData = sp.zeros( (nDim, int(len(inData)/decim_factor) ) )
    #logEnvData = sp.zeros((nDim, int(len(inData)/decim_factor)))
    #dEnvData = sp.zeros((nDim, int(len(inData)/decim_factor)-1))
    
    # OnsetDetectionFunctionの作成
    env_data = []
    d_env_data = []
    for i in xrange(n_dim):
        fil_data[i] = sp_sig.lfilter(fbank_coefs[i]['b'], fbank_coefs[i]['a'], input)
        cur_env_data = _odf(fil_data[i], fs, 128)
        cur_d_env_data = cur_env_data[1:] - cur_env_data[0:len(cur_env_data)-1]
        cur_d_env_data = (cur_d_env_data + abs(cur_d_env_data)) / 2.0 # 半波整流
        #logEnvData[i] = sp.log10(envData[i])
        
        env_data.append(cur_env_data)
        d_env_data.append(cur_d_env_data)
        
    env_data = sp.array(env_data)
    d_env_data = sp.array(d_env_data)
    
    # Onsetの抽出
        
        

    #tframesize = int(framesize / decim_factor)
    #nFrames = int(sp.ceil(len(rsmpData[0]) / tframesize))
    #st = 0
    #ed = tframesize
    #winFunc = sig.get_window('hann', tframesize*2)
    #winFunc = winFunc[0:tframesize+1]

    #for i in xrange(nFrames):
    #    curData = rsmpData[:, st:ed]
    #    for j in xrange(len(fbCoefs)):
    #        # 畳込み
    #        #conv = sp.convolve(curData[j], winFunc)
    #        #envData[j, st*2:ed*2] = conv
    #
    #        # hilbert
    #        envData[j, st:ed] = abs(sig.hilbert(curData[j]))
    #        #envData[j, st:ed] = calcEMA(envData[j, st:ed], 32)
    #    st += tframesize
    #    ed += tframesize
    #    if ed > len(rsmpData[0]):
    #        ed = len(rsmpData[0])

    return env_data, d_env_data

def _odf(input, fs=44100, decim_factor=128):
    """
    ODF calculation for onsetKlapuri()
    
    Parameters:
      indata: ndarray
        input signal
      fs: int
        samplingrate
      decim_factor: int
        decimation factor
    
    Returns:
      result: ndarray
        odf data
    """
    hrect_data = halfRect(input)
    rsmp_data = sp_sig.decimate(hrect_data, q=decim_factor, n=2, ftype='iir')
    env_data = abs(sp_sig.hilbert(rsmp_data))
    env_data = moving_average_exp_cy(env_data, 16)
    
    return env_data

def _peakpick(input, threshold):
    """
    Peak picking for onset detection
    
    Parameters:
      inData: ndarray
        input signal
      threshold: float
        threshold for peak picking
    
    Returns:
      result: ndarray
        peak picked data
    """
    peak_data = sp.zeros(len(input))
    
    gain = moving_average_exp_cy(input, 64)

    flg_det = True
    for i in xrange(1, len(input)-1):
        if input[i] < gain[i]:
            flg_det = True
    
        if (input[i] > input[i-1] and input[i] > input[i+1]):
            if flg_det == True and input[i] >= threshold:
                peak_data[i] = input[i]
                flg_det = False
            else:
                peak_data[i] = 0
        else:
            peak_data[i] = 0

    return peak_data
    
def _makeFilterbank(fs):
    """
    BPF Filterbank for onsetKlapuri()
    
    Parameters:
      fs: int
        samplingrate
    
    Returns:
      result: list of dictionary
        coefficients of filterbank
    """
    fbank_coefs = []

    Wn1 = 44 * 2**sp.arange(4)
    Wn2 = Wn1[-1] * 2**(sp.arange(1, 18) * 1/3.)
    Wn = sp.r_[Wn1, Wn2, sp.array([18000.0])]
    Wn /= (fs/2.0)

    for i in xrange(21):
        coef = {'b': [], 'a': []}
        coef['b'], coef['a'] = sp_sig.iirfilter(2, Wn[i:i+2], btype='bandpass')
        fbank_coefs.append(coef)

    return fbank_coefs

def _calcOnsetThreshold(odf, alpha):
    """
    Calculate threshold for onset()
    
    Parameters:
      odf: ndarray
        onset detection function
      alpha: float
        parameter of moving average
    
    Returns:
      result: ndarray
        threshold array
    """
    g = sp.zeros(len(odf))
    for i in xrange(1, len(odf)):
        g[i] = max(odf[i], alpha*g[i-1] + (1 - alpha) * odf[i])

    return g