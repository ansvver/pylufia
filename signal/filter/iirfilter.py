# -*- coding: utf-8 -*-

"""
@file iirfilter.py
@brief IIR filter computation functions
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def iir_calc_coef_fs(q=3.0, f=440.0, g=2.0, type='lpf2', fs=44100):
    """
    Calculate IIR Filter coefficients
    
    Parameters:
      q: float
        Q factor
      f: float
        cutoff frequency
      g: float
        gain
      type: string
        type of filter
      fs: int
        samplingrate
    
    Returns:
      b: ndarray
        filter coefficient b
      a: ndarray
        filter coefficient a
    """
    CQ0 = 0.70710678118654752
    PI = sp.pi

    a = sp.zeros(3)
    b = sp.zeros(3)
    a[0] = 1.0

    if (type == 'peakdip'): # Peak/Dip
        if (g == 0.0):
            pass
        elif (g > 0.0):
            qq = q
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = 1.0 + tx / qq + tx * tx
            a[0] = (1.0 + kk * tx / qq + tx * tx ) / t0
            a[1] = 2.0 * ( tx * tx - 1.0) / t0
            a[2] = (1.0 - kk * tx / qq + tx * tx) / t0
            b[1] = -1.0 * a[1]
            b[2] = (-1.0 + tx / qq - tx * tx) / t0
        else:
            qq = q
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = 1.0 + kk * tx / qq + tx * tx
            a[0] = ( 1.0 + tx / qq + tx * tx) / t0
            a[1] = 2.0 * ( tx * tx - 1.0) / t0
            a[2] = ( 1.0 - tx / qq + tx * tx) / t0
            b[1] = -1.0 * a[1]
            b[2] = ( -1.0 + kk * tx / qq - tx * tx) / t0

    elif (type == 'bpf'): # BPF/BEF
        qq = q
        tx = sp.tan(PI * f / fs)
        if ( g >= 0.0):
            t0 = 1.0 + tx / qq + tx * tx
            t1 = 1.0 - tx / qq + tx * tx
            a[0] = ( tx / qq) / t0
            a[1] = 0.0
            a[2] = -1.0 * a[0]
            b[1] = 2.0 * ( 1.0 - tx * tx) / t0
            b[2] = -1.0 * t1 / t0
        else:
            t0 = 1.0 + tx / qq + tx * tx
            t1 = 1.0 - tx / qq + tx * tx
            a[0] = ( 1.0 + tx * tx) / t0
            a[1] = 2.0 * ( tx * tx - 1.0) / t0
            a[2] = a[0]
            b[1] = -1.0 * a[1]
            b[2] = -1.0 * t1 / t0

    elif (type == 'lba'): # LBA
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = ( 1.0 + tx) * ( 1.0 + tx)
            a[0] = ( 1.0 + sp.sqrt(kk) * tx) * ( 1.0 + sp.sqrt(kk) * tx) / t0
            a[1] = 2.0 *( kk * tx * tx - 1.0) / t0
            a[2] = ( 1.0 - sp.sqrt(kk) * tx) * (1.0 - sp.sqrt(kk) * tx) / t0
            b[1] = 2.0 * ( 1.0 - tx * tx) / t0
            b[2] = -1.0 *( 1.0 - tx) * ( 1.0 - tx) / t0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10 ** (-g / 20.0)
            t0 = ( 1.0 + sp.sqrt(kk) * tx) * ( 1.0 + sp.sqrt(kk) * tx)
            a[0] = ( 1.0 + tx) * ( 1.0 + tx) / t0
            a[1] = 2.0 *( tx * tx - 1.0) / t0
            a[2] = ( 1.0 - tx) * ( 1.0 - tx) / t0
            b[1] = 2.0 * ( 1.0 - kk * tx * tx) / t0
            b[2] = -1.0 * ( 1.0 - sp.sqrt(kk) * tx) * ( 1.0 - sp.sqrt(kk) * tx) / t0

    elif (type == 'hba'): #HBA
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = (1.0 + tx) * (1.0 + tx)
            a[0] = (sp.sqrt(kk) + tx) * (sp.sqrt(kk) + tx) / t0
            a[1] = 2.0 * (tx * tx - kk) / t0
            a[2] = (sp.sqrt(kk) - tx) * (sp.sqrt(kk) - tx) / t0
            b[1] = 2.0 * (1.0 - tx * tx) / t0
            b[2] = -1.0 * (1.0 - tx) * (1.0 - tx) / t0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = ( sp.sqrt(kk) + tx) * ( sp.sqrt(kk) + tx)
            a[0] = ( 1.0 + tx) * ( 1.0 + tx) / t0
            a[1] = 2.0 * ( tx * tx - 1.0) / t0
            a[2] = ( 1.0 - tx) * ( 1.0 - tx) / t0
            b[1] = 2.0 * ( kk - tx * tx) / t0
            b[2] = -1.0 * ( sp.sqrt(kk) - tx) * ( sp.sqrt(kk) - tx) / t0

    elif (type == 'lpf2'): # LPF (2-dim)
        if (g < 0.0):
            pass
        else:
            qq = CQ0
            tx = sp.tan(PI * f / fs)
            t0 = 1.0 + tx / qq + tx * tx
            a[0] = ( 1.0 * tx * tx) / t0
            a[1] = a[0] * 2.0
            a[2] = a[0]
            b[1] = ( 2.0 * ( 1.0 - tx * tx)) / t0
            b[2] = ( -1.0 + tx / qq - tx * tx) / t0

    elif (type == 'hpf2'): # HPF (2-dim)
        if (g < 0.0):
            pass
        else:
            qq = CQ0
            tx = sp.tan(PI * f / fs)
            t0 = 1.0 + tx / qq + tx * tx
            a[0] = 1.0 / t0
            a[1] = -2.0 * a[0]
            a[2] = a[0]
            b[1] = ( 2.0 * ( 1.0 - tx * tx)) / t0
            b[2] = ( -1.0 + tx / qq - tx * tx) / t0

    elif (type == 'lpf1'): # LPF (1-dim)
        if (g < 0.0):
            pass
        else:
            tx = sp.tan(PI * f / fs)
            t0 = 1.0 + tx
            a[0] = tx / t0;
            a[1] = a[0]
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0

    elif (type == 'hpf1'): # HPF (1-dim)
        if (g < 0.0 or f < 0.0):
            pass
        else:
            tx = sp.tan(PI * f / fs)
            t0 = 1.0 + tx
            a[0] = 1.0 / t0
            a[1] = -1.0 * a[0]
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0

    elif (type == 'lsf1'): # Low shelving (1-dim)
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = 1.0 + tx
            a[0] = ( 1.0 + kk * tx) / t0
            a[1] = ( kk * tx - 1.0) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = 1.0 + kk * tx
            a[0] = ( 1.0 + tx) / t0
            a[1] = ( tx - 1.0) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - kk * tx) / t0
            b[2] = 0.0

    elif (type == 'hsf1'): # High shelving (1-dim)
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = 1.0 + tx
            a[0] = ( kk + tx) / t0
            a[1] = ( tx - kk) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = kk + tx
            a[0] = ( 1.0 + tx) / t0
            a[1] = ( tx - 1.0) / t0
            a[2] = 0.0
            b[1] = ( kk - tx) / t0
            b[2] = 0.0

    elif (type == 'ltone1'): # Low tone (1-dim)
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = 1.0 + tx / kk
            a[0] = ( 1.0 + tx) / t0
            a[1] = ( tx - 1.0) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - tx / kk) / t0
            b[2] = 0.0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = 1.0 + tx
            a[0] = ( 1.0 + tx / kk) / t0
            a[1] = ( tx / kk - 1.0) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0

    elif (type == 'htone1'): # High tone (1-dim)
        if (g == 0.0):
            pass
        elif (g > 0.0):
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (g / 20.0)
            t0 = 1.0 + kk * tx
            a[0] = kk * ( 1.0 + tx) / t0
            a[1] = kk * ( tx - 1.0) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - kk * tx) / t0
            b[2] = 0.0
        else:
            tx = sp.tan(PI * f / fs)
            kk = 10.0 ** (-g / 20.0)
            t0 = 1.0 + tx
            a[0] = ( 1.0 / kk + tx) / t0
            a[1] = ( tx - 1.0 / kk) / t0
            a[2] = 0.0
            b[1] = ( 1.0 - tx) / t0
            b[2] = 0.0

    else:
        a[0] = 1.0
        a[1] = 0.0
        a[2] = 0.0
        b[1] = 0.0
        b[2] = 0.0

    return b, a

def iir_calc_coef(q, f, g, type):
    """
    Calculate IIR Filter coefficients (fs=44.1kHz fixed)
    
    Parameters:
      q: float
        Q factor
      f: float
        cutoff frequency
      g: float
        gain
      type: string
        type of filter
      fs: int
        samplingrate
    
    Returns:
      b: ndarray
        filter coefficient b
      a: ndarray
        filter coefficient a
    """
    return iir_calc_coef_fs(q, f, g, type, 44100.0)

def iir_apply(input, b, a):
    """
    Apply IIR filter
    
    Parameters:
      inData: ndarray
        input signal
      b: ndarray
        filter coefficient b
      a: ndarray
        filter coefficient a
    
    Returns:
      result: ndarray
        filtered signal
    """
    d0 = 0.0
    d1 = 0.0
    output = sp.zeros(len(input))
    for i, _input in enumerate(input):
        _output = _input * a[0] + d0
        output[i] = _output
        d0 = a[1] * _input + b[1] * _output + d1
        d1 = a[2] * _input + b[2] * _output

    return output
