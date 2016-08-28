# -*- coding: utf-8 -*-

"""
====================================================================
wavefile I/O functions
====================================================================
"""

# どちらかをTrueにすること
USE_SCIPY = False
USE_WAVIO = True

import wave
import scipy as sp
if USE_SCIPY:
    import scipy.io.wavfile
if USE_WAVIO:
    import wavio


def wavread_old(fname, dtype='float'):
    """
    Read wave file
    
    Parameters:
      fname: string
        input filename
      
    Returns:
      outData: ndarray
        wave data of fname
      fs: float
        sampling rate of fname
      bit: int
        bit depth of fname
    """ 
    wp = wave.open(fname, 'rb')
    fs = wp.getframerate()
    bit = wp.getsampwidth() * 8
    ch = wp.getnchannels()
    length = wp.getnframes()
    if dtype == 'float':
        scaling = 32768.0
    elif dtype == 'int':
        scaling = 1
    else:
        scaling = 32768.0

    str_data = wp.readframes(wp.getnframes())
    if bit == 16:
        data = sp.frombuffer(str_data, 'int16') / scaling # 24bit wavはどうすればいい？
    elif bit == 24:
        #data = sp.fromstring(strData, sp.int32) # 24bit wavはどうすればいい？
        data = sp.zeros(wp.getnframes())
        
    if (ch == 2):
        #lData = sp.int32(data[::2])
        #rData = sp.int32(data[1::2])
        l_data = data[::2]
        r_data = data[1::2]
        out_data = sp.array([l_data, r_data])
    else:
        #lData = sp.int32(data)
        l_data = data
        out_data = sp.array(l_data)

    return out_data, fs, bit

def wavread(fname):
    if USE_SCIPY:
        fs,x = scipy.io.wavfile.read(fname)
        x = x.T
        bit = 16
    elif USE_WAVIO:
        w = wavio.read(fname)
        fs = w.rate
        bit = 8*w.sampwidth
        x = w.data.T
    x = x / float( 2**(bit-1) ) # -1.0 to 1.0に正規化 
    return x,fs,bit
    
def wavread_header(fname):
    """
    Read wave file header chunk
    
    Parameters:
      fname: string
        input wavfile
    
    Returns:
      results: dict
        metadata of fname header
    """
    wp = wave.open(fname)
    fs = wp.getframerate()
    bit = wp.getsampwidth() * 8
    ch = wp.getnchannels()
    length = wp.getnframes()
    
    rtn = {'fs': fs, 'bit': bit, 'channel': ch, 'length': length}
    return rtn

def wavwrite_old(data, fs, bit, fname):
    """
    Write wave data to file
    
    Parameters:
      data: ndarray
        wave data
      fs: float
        samplingrate
      bit: int
        bit depth
      fname: string
        output filename
        
    Returns:
     result: int
       error code (0: OK -1: NG)
    """
     
    wp = wave.open(fname, 'w')

    wp.setsampwidth(bit/8)
    wp.setframerate(fs)
    if (len(data.shape) == 2):
        n_channels = 2
        wp.setnchannels(n_channels)
        n_smp = data.shape[1]
        interleaved = sp.zeros(n_channels*n_smp)
        for i, (ldata,rdata) in enumerate(zip(data[0], data[1])):
            interleaved[2*i] = ldata
            interleaved[2*i+1] = rdata
        
    elif (len(data.shape) == 1):
        n_channels = 1
        n_smp = data.shape[0]
        wp.setnchannels(n_channels)
        interleaved = data
    else:
        return -1

    interleaved = interleaved.astype('int')

    wData = sp.int16(interleaved).tostring()

    wp.writeframes(wData)
    
    return 0

def wavwrite(data, fs, bit, fname):
    if USE_SCIPY:
        outdata = ( data * 2**(bit-1) ).astype(sp.int16)
        scipy.io.wavfile.write(fname, fs, outdata.T)
    elif USE_WAVIO:
        data_wavio = data * 2**(bit-1)
        sampwidth = bit/8
        wavio.write(fname, data_wavio, fs, sampwidth)
