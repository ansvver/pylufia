# -*- coding: utf-8 -*-

"""
@file cqt.pyx
@brief CQT(Constant-Q Transform) implementation (cython version)
@author ふぇいと (@stfate)

@description

"""

import cython
import numpy as np
cimport numpy as np
import scipy as sp
import time

from fft import *
from plifia.signal import *


cdef make_cqt_kernels_cy(int framesize, float fs, int n_cq, int n_per_semitone, float fmin, int n_wavs=30):
    """
    CQTのカーネル関数作成
    """
    cdef float wmin = 2*np.pi*fmin/fs
    cdef float n_win = float(float(n_wavs) * fs / float(1+fmin) * wmin)
    
    cdef np.ndarray[double, ndim=1] freqs = np.zeros(n_cq, dtype=np.double)
    cdef np.ndarray[complex, ndim=2] kernels = np.zeros( (n_cq, int(framesize/2)), dtype=complex )
    cdef float scaling = fs/(2*np.pi)
    cdef int kcq
    cdef float wkcq
    cdef np.ndarray[int, ndim=1] n = np.zeros(framesize, dtype=int)
    cdef np.ndarray[double, ndim=1] cur_win = np.zeros(framesize, dtype=np.double)
    cdef np.ndarray[complex, ndim=1] r = np.zeros(framesize, dtype=complex)
    
    for kcq from 0 <= kcq < n_cq:
        wkcq = (2**(1/float(12*n_per_semitone)))**kcq * wmin
        n = np.arange(framesize)
        cur_win = _window(n, framesize, wkcq, n_win)
        r = cur_win * np.exp(1.0j*wkcq*(n - framesize/2))
        freqs[kcq] = wkcq * scaling
        kernels[kcq,:] = np.real(fft(r, framesize))[:int(framesize/2)]
        
    return kernels, freqs
    
def cqt_cy(x, framesize=8192, hopsize=1024, fs=44100, window='hann', n_cq=88, n_per_semitone=1, fmin=60.0):
    """
    CQT実行
    
    Notes:
      Cythonで実装してみたがほとんど高速化されない．
    """
    cdef np.ndarray[double, ndim=1] x_zpad = np.r_[np.zeros(framesize/2), x, np.zeros(framesize/2)]
    cdef np.ndarray[complex, ndim=2] kernels = np.zeros( (n_cq, int(framesize/2)), dtype=complex )
    cdef np.ndarray[double, ndim=2] kernels_f = np.zeros( (n_cq, int(framesize/2)), dtype=np.double )
    cdef np.ndarray[double, ndim=1] freqs = np.zeros(n_cq, dtype=np.double)
    cdef int k
    
    kernels,freqs = make_cqt_kernels_cy(framesize, fs, n_cq, n_per_semitone, fmin)
    kernels_f = kernels.astype(float)
    
    X,F,T = stft(x_zpad, framesize, hopsize, fs, window)
    X = X.astype(complex).T

    cdef np.ndarray[complex, ndim=2] XX = np.zeros((X.shape[0],X.shape[1]), dtype=complex)
    XX[:] = X
    
    CQ = np.zeros( (XX.shape[0], n_cq), dtype=complex)
    cdef int kernel_len = kernels_f.shape[1]
    
    t1 = time.clock()
    for k from 0 <= k < kernel_len:
        CQ += kernels_f[np.newaxis,:,k] * XX[:,np.newaxis,k]
    # CQ = (kernels_f[np.newaxis,:,:] * XX[:,np.newaxis,:framesize/2]).sum(2)
    t2 = time.clock()
    print 't(calcCQ)={0}'.format(t2-t1)

    CQ = np.absolute(CQ)
    CQ = CQ.astype(float).T
    CQ = CQ[:]
    
    return CQ, freqs, T

def make_oct_cqt_kernels_cy(fmax, n_bins, fs, q=1.0, atom_hop_factor=0.25, thresh=0.0005, window='blackmanharris', perf_rast=False):
    """
    CQTのKernel設計(Klapuri CQT用)
    """

    # define
    fmin = (fmax/2)*2**(1.0/float(n_bins))
    cdef float Q = 1.0/(2**(1/float(n_bins))-1)
    Q = Q*q
    cdef float Nk_max = Q * fs / float(fmin)
    Nk_max = round(Nk_max) # length of the largest atom [samples]

    # Compute FFT size, FFT hop, atom hop,
    cdef float Nk_min = round( Q * fs / (fmin*2**((n_bins-1)/float(n_bins))) ) # length of the shortest atom [samples]
    cdef float atom_hop = round(Nk_min*atom_hop_factor) # atom hop size
    cdef float first_center = sp.ceil(Nk_max/2) # first possible center position within the frame
    first_center = atom_hop * sp.ceil(first_center/atom_hop) # lock the first center to an integer multiple of the atom hop size
    # nextpow2(x)はScipyにはないので代替手段を考える。要はx以上で最小の2のべき乗数を見つけるだけ。
    # fftsize = 2**nextpow2(first_center+sp.ceil(Nk_max/2)) # use smallest possible FFT size (increase sparsity)
    cdef int fftsize = 2**_nextpow2(first_center+sp.ceil(Nk_max/2))
    cdef float winlen = 0.0

    if perf_rast:
        winlen = sp.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop) # number of temporal atoms per FFT Frame
        if winlen == 0:
            fftsize = fftsize * 2
            winlen = np.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop)
    else:
        winlen = np.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop)+1 # number of temporal atoms per FFT Frame

    cdef float last_center = first_center + (winlen-1)*atom_hop
    cdef float hopsize = (last_center + atom_hop) - first_center # hop size of FFT frames
    cdef float fft_overlap = (fftsize-hopsize/fftsize)*100 # overlap of FFT frames in percent ***AK:needed?

    # init variables
    cdef np.ndarray[complex,ndim=1] temp_kernel = sp.zeros( fftsize, dtype=complex )
    cdef np.ndarray[complex,ndim=2] spec_kernel = sp.zeros( (n_bins*winlen,fftsize), dtype=complex )

    # Compute kernel
    cdef int atom_ind = 0
    cdef int idx = 0
    cdef int k,i
    for k in xrange(n_bins):
        Nk = int(round( Q * fs / float(fmin*2**(k/float(n_bins))) )) # N[k] = (fs/fk)*Q. Rounding will be omitted in future versions
        
        win_func = np.sqrt(sp.signal.get_window(window, Nk))
        # win_func = sp.sqrt(sp.signal.blackmanharris(Nk))
        
        fk = fmin*2**(k/float(n_bins))
        temp_kernel_bin = (win_func/float(Nk)) * sp.exp(2*sp.pi*1j*fk*sp.arange(Nk)/float(fs))
        atom_offset = first_center - sp.ceil(Nk/2.0)

        for i in xrange(int(winlen)):
            shift = int(atom_offset + (i * atom_hop))
            temp_kernel[shift:Nk+shift] = temp_kernel_bin
            atom_ind = atom_ind+1
            _spec_kernel = sp.fftpack.fft(temp_kernel)
            _spec_kernel[sp.absolute(_spec_kernel)<=thresh] = 0
            spec_kernel[idx] = _spec_kernel
            # spar_kernel = sparse(sp.r_[spar_kernel,spec_kernel])
            temp_kernel = np.zeros( fftsize, dtype=complex ) # reset window
            idx += 1

    # spar_kernel = (spar_kernel.T)/fftsize
    spec_kernel = (spec_kernel.T)/fftsize

    # Normalize the magnitudes of the atoms
    wx1 = np.argmax(np.absolute(spec_kernel[:,0])) #matlabのmaxと挙動を合わせるためabsを取る
    wx2 = np.argmax(np.absolute(spec_kernel[:,-1])) #上に同じ
    wK = spec_kernel[wx1:wx2+1,:]
    wK = np.diag(np.dot(wK, wK.conjugate().T))
    wK = wK[int(round(1/q)):int(len(wK)-round(1/q)-2)]
    weight = 1./np.mean(np.absolute(wK))
    weight = weight *(hopsize/float(fftsize))
    weight = np.sqrt(weight) # sqrt because the same weight is applied in icqt again
    spec_kernel = weight * spec_kernel

    params = {'fftsize':fftsize,
              'hopsize':hopsize,
              'fft_overlap':fft_overlap,
              'perf_rast':perf_rast,
              'n_bins':n_bins,
              'firstcenter':first_center,
              'atom_hop':atom_hop,
              'winlen':winlen,
              'Nk_max':Nk_max,
              'Q':Q,
              'fmin':fmin}
                 
    return spec_kernel, params

# def cqt_sk(x, framesize=8192, hopsize=1024, fs=44100, n_cq=88, n_per_semitone=1, fmin=60.0):
def cqt_oct_cy(x, fmin, fmax, n_bins, fs, q=1.0, atom_hop_factor=0.25, thresh=0.0005, window='blackmanharris'):
    """
    CQT計算(Schorkhuber,Klapuriによるオクターブごとに一気に計算する手法)
    CQT toolbox(http://www.eecs.qmul.ac.uk/~anssik/cqt/)のMATLABコードを移植したもの。

    """
    # define
    cdef int n_octaves = np.ceil(sp.log2(fmax/fmin)).astype('int')
    fmin = (fmax/2**n_octaves) * 2**(1/n_bins) # set fmin to actual value
    cdef int xlen_init = len(x)

    # design lowpass filter
    cdef int LPorder = 6 # order of the anti-aliasing filter
    cdef float cutoff = 0.5
    B,A = sp.signal.butter(LPorder, cutoff, 'low') # design f_nyquist/2-lowpass filter

    # design kernel for one octave 
    cqt_kernel,params = make_oct_cqt_kernels_cy(fmax, n_bins, fs, q=q, atom_hop_factor=atom_hop_factor, thresh=thresh, window=window)

    # calculate CQT
    listCQ = []
    cdef int max_block = params['fftsize'] * 2**(n_octaves-1) # largest FFT Block (virtual)
    cdef int prefix_zeros = max_block
    cdef int suffix_zeros = max_block
    cdef np.ndarray[double,ndim=1] x_zp = np.r_[np.zeros(prefix_zeros), x, np.zeros(suffix_zeros)] # zeropadding
    cdef float overlap = params['fftsize'] - params['hopsize']
    cdef np.ndarray[complex,ndim=2] K = cqt_kernel.conjugate().T # conjugate spectral kernel for cqt transformation
    
    cdef int i
    time_make_frame = 0.0
    time_fft = 0.0
    time_dot = 0.0
    for i in xrange(n_octaves):
        t1 = time.clock()
        framed = make_framed_data(x_zp, params['fftsize'], params['hopsize'], 'boxcar')
        t2 = time.clock()
        time_make_frame += t2-t1

        t1 = time.clock()
        XX = sp.fftpack.fft(framed, params['fftsize']) # applying fft to each column (each FFT frame)
        t2 = time.clock()
        time_fft += t2-t1

        t1 = time.clock()
        CQ_oct = np.dot(K, XX.T)
        t2 = time.clock()
        time_dot += t2-t1

        listCQ.append(CQ_oct) # calculating cqt coefficients for all FFT frames for this octave

        if i != n_octaves-1:
            x_zp = sp.signal.filtfilt(B, A, x_zp) # anti aliasing filter
            x_zp = x_zp[0::2] # drop samplerate by 2

    print time_make_frame
    print time_fft
    print time_dot

    # map to sparse matrix representation
    arrayCQ = _list2ndarray(listCQ, n_octaves, n_bins, params['firstcenter'], params['atom_hop'], params['winlen'])

    intCQ = interpolate_cqt(arrayCQ, n_octaves, n_bins)
    
    # discard prefix suffix zero spectrum
    empty_hops = params['firstcenter'] / params['atom_hop']
    max_drop = empty_hops*2**(n_octaves-1) - empty_hops
    dropped_samples = (max_drop-1)*params['atom_hop'] + params['firstcenter']
    output_time_vec = sp.arange(intCQ.shape[1])*params['atom_hop']-prefix_zeros + dropped_samples
    start_frame = sp.where(output_time_vec >= 0)[0][0]
    end_frame = sp.where(output_time_vec >= len(x))[0][0]
    intCQ_dropped = intCQ[:,start_frame:end_frame]
    T = output_time_vec[start_frame:end_frame]
    
    # パラメータ辞書に逆変換に必要なパラメータを追加
    params['n_octaves'] = n_octaves
    params['coef_B'] = B
    params['coef_A'] = A
    params['prefix_zeros'] = prefix_zeros
    params['xlen_init'] = xlen_init
    
    return arrayCQ, intCQ_dropped, cqt_kernel, T, params
    
def interpolate_cqt(CQ, n_octaves, n_bins):
    intCQ = sp.zeros( CQ.shape )
    
    for k in xrange(CQ.shape[0]):
        oct = int(n_octaves-sp.floor((k+1-0.1)/float(n_bins)))
        step_vec = sp.arange(0,CQ.shape[1],2**(oct-1))
        Xbin = CQ[k,step_vec]
        intCQ[k,:] = sp.interp(sp.arange(CQ.shape[1]), step_vec, sp.absolute(Xbin))
        
    return intCQ

def icqt_oct(arrayCQ, cqt_kernel, params):
    """
    inverse constant-q transform
    """
    listCQ = _ndarray2list(arrayCQ, params['n_bins'], params['n_octaves'], params['winlen'], params['firstcenter'], params['atom_hop'])
    
    fftsize = params['fftsize']
    hopsize = params['hopsize']
    n_octaves = params['n_octaves']
    
    K_inv = cqt_kernel
    y = []
    for i in xrange(n_octaves-1,-1,-1):
        CQ_oct = listCQ[i]
        
        Y = sp.dot(K_inv, CQ_oct)

        y_oct_temp = sp.fftpack.ifft(Y.T).T
        y_oct = 2*y_oct_temp.real
        
        n_blocks = Y.shape[1]
        sig_len = fftsize + (n_blocks-1)*hopsize
        
        if sig_len >= len(y):
            y = sp.r_[y, sp.zeros(sig_len-len(y))]
        for n in xrange(n_blocks-1):
            y[n*hopsize:n*hopsize+fftsize] = y_oct[:,n] + y[n*hopsize:n*hopsize+fftsize] #overlap-add
            
        # upsample & filtering
        if i != 0:
            y2 = sp.zeros(len(y)*2)
            y2[::2] = y
            y = sp.signal.filtfilt(params['coef_B'], params['coef_A'], y2)
            y *= 2

    # trim prefix&suffix zeros
    y = y[params['prefix_zeros']:params['prefix_zeros']+params['xlen_init']+1]
        
    return y


""" helper functions """

cdef _hamming(np.ndarray[int, ndim=1] n, int N, double alpha=25/46.0):
    """
    ハミング窓
    """
    output = alpha - (1.0-alpha)*sp.cos(2*sp.pi*n/N)
    output[n<0] = 0
    output[n>=N] = 0
    return output
    
cdef _window(np.ndarray[int, ndim=1] n, int N, float wkcq, float n_win=2048):
    """
    CQT用窓関数作成
    """
    cdef int Nkcq = int(n_win/float(wkcq))
    cdef np.ndarray[int, ndim=1] _n = n - (N/2 - Nkcq/2)
    win = _hamming(_n, Nkcq)
    return win/win.sum()

cdef _nextpow2(float i):
    n = int(sp.ceil(sp.log2(i)))
    return n

cdef _list2ndarray(listCQ, int n_octaves, int n_bins, float firstcenter, float atom_hop, float n_atoms):
    """
    オクターブごとのCQT結果が格納されたlistを全オクターブを統合したひとつのndarrayに変換する
    """
    cdef float empty_hops = firstcenter/atom_hop
    cdef float drop = empty_hops*2**(n_octaves-1)-empty_hops
    cdef np.ndarray[complex,ndim=2] CQ = np.zeros((n_bins*n_octaves,listCQ[0].shape[1]*n_atoms-drop), dtype=complex)

    cdef int i,u
    for i in xrange(n_octaves):
        drop = empty_hops*2**(n_octaves-i-1)-empty_hops
        X = listCQ[i]
        if n_atoms > 1: # more than one atom per bin --> reshape
            Xoct = np.zeros( (n_bins,n_atoms*X.shape[1]-drop), dtype=complex )
            for u in xrange(n_bins):
                # reshape to continous windows for each bin (for the case of several wins per frame)
                octX_bin = X[u*n_atoms:(u+1)*n_atoms,:]
                Xcont = np.reshape(octX_bin,octX_bin.shape[0]*octX_bin.shape[1], order='F')
                Xoct[u,:] = Xcont[drop:]
            X = Xoct
        else:
            X = X[:,drop:]
        bin_st = n_bins*n_octaves-n_bins*(i+1)
        bin_ed = n_bins*n_octaves-n_bins*i
        frm_st = 0
        frm_ed = X.shape[1]*2**i
        frm_step = 2**i
        CQ[bin_st:bin_ed,frm_st:frm_ed:frm_step] = X.copy()

    return CQ

cdef _ndarray2list(np.ndarray[complex,ndim=2] arrayCQ, int n_bins, int n_octaves, float n_atom, float firstcenter, float atom_hop):
    """
    ndarrayに統合されたCQTをオクターブごとのCQT結果が格納されたlistに変換する
    """
    cdef float empty_hops = firstcenter / atom_hop
    listCQ = []

    cdef int i,u
    cdef float dropped = 0.0
    for i in xrange(n_octaves):
        dropped = empty_hops*2**(n_octaves-i-1)-empty_hops
        X = arrayCQ[n_bins*n_octaves-(i+1)*n_bins:n_bins*n_octaves-i*n_bins,::2**i]
        
        X = np.c_[sp.zeros( (n_bins,dropped)), X]
        X = np.c_[X, sp.zeros((n_bins,sp.ceil(X.shape[1]/n_atom)*n_atom-X.shape[1]))]

        if n_atom > 1: #reshape
            Xcell = np.zeros( (n_bins*n_atom,np.ceil(X.shape[1]/n_atom)), dtype=complex)
            for u in xrange(n_bins):
                Xbin = np.reshape(X[u,:], (n_atom,len(X[u,:])/n_atom), order='F').copy()
                Xcell[u*n_atom:(u+1)*n_atom,:] = Xbin

            listCQ.append(Xcell)
        else:
            listCQ.append(X)
            
    return listCQ
