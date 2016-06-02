# -*- coding: utf-8 -*-

"""
@file cqt.py
@brief CQT(Constant-Q Transform) implementation
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from pylufia.signal.common import *
from fft import *

import time
import cPickle


def _hamming(n, N, alpha=25/46.0):
    """
    ハミング窓
    """
    output = alpha - (1.0-alpha)*sp.cos(2*sp.pi*n/N)
    output[n<0] = 0
    output[n>=N] = 0
    return output
    
def _window(n, N, wkcq, n_win=2048):
    """
    CQT用窓関数作成
    """
    Nkcq = n_win/float(wkcq)
    _n = n - (N/2.0 - Nkcq/2.0)
    win = _hamming(_n, Nkcq)
    return win/win.sum()

def make_cqt_kernels(framesize, fs, n_cq, n_per_semitone, fmin, n_wavs=30):
    """
    CQTのカーネル関数作成
    """
    wmin = 2*sp.pi*fmin/fs
    n_win = float(float(n_wavs) * fs / float(1+fmin) * wmin)
    
    freqs = sp.zeros(n_cq)
    kernels = sp.zeros( (n_cq, int(framesize/2)) )
    scaling = fs/(2*sp.pi)
    
    for kcq in xrange(n_cq):
        wkcq = (2**(1/float(12*n_per_semitone)))**kcq * wmin
        n = sp.arange(framesize)
        cur_win = _window(n, framesize, wkcq, n_win)
        r = cur_win * sp.exp(1.0j*wkcq*(n - framesize/2))
        freqs[kcq] = wkcq * scaling
        kernels[kcq,:] = sp.real(fft(r, framesize))[:int(framesize/2)]
        
    return kernels, freqs
    
def cqt(x, framesize=8192, hopsize=1024, fs=44100, n_cq=88, n_per_semitone=1, fmin=60.0):
    """
    CQT計算
    
    kernelとスペクトログラムの乗算を行列演算で一気にやってしまえば高速化されるが
    大量のメモリを食うようになる．
    """
    x_zpad = sp.r_[sp.zeros(framesize/2), x, sp.zeros(framesize/2)]
    
    kernels,freqs = make_cqt_kernels(framesize, fs, n_cq, n_per_semitone, fmin)
    kernels = kernels.astype(float)
    
    X,F,T = stft(x_zpad, framesize, hopsize, fs, 'hamming')
    X = X.astype(complex).T
    
    CQ = sp.zeros( (X.shape[0], n_cq), dtype=complex)
    # for k in xrange(kernels.shape[1]):
        # CQ += kernels[sp.newaxis,:,k] * X[:,sp.newaxis,k]
    CQ[:] = sp.dot( kernels, X[:,:kernels.shape[1]].T ).T
    
    CQ = sp.absolute(CQ)
    CQ = CQ.astype(float).T
    CQ = CQ[:]
    
    return CQ, freqs, T

def _list2ndarray(listCQ, n_octaves, n_bins, firstcenter, atom_hop, n_atoms):
    """
    オクターブごとのCQT結果が格納されたlistを全オクターブを統合したひとつのndarrayに変換する
    """
    empty_hops = firstcenter/atom_hop
    drop = empty_hops*2**(n_octaves-1)-empty_hops
    CQ = sp.zeros((n_bins*n_octaves,listCQ[0].shape[1]*n_atoms-drop), dtype=complex)

    for i in xrange(n_octaves):
        drop = empty_hops*2**(n_octaves-i-1)-empty_hops
        X = listCQ[i]
        if n_atoms > 1: # more than one atom per bin --> reshape
            Xoct = sp.zeros( (n_bins,n_atoms*X.shape[1]-drop), dtype=complex )
            for u in xrange(n_bins):
                # reshape to continous windows for each bin (for the case of several wins per frame)
                octX_bin = X[u*n_atoms:(u+1)*n_atoms,:]
                Xcont = sp.reshape(octX_bin, octX_bin.shape[0]*octX_bin.shape[1], order='F')
                # print Xcont.shape #debug
                Xoct[u,:] = Xcont[drop:]
            X = Xoct
        else:
            X = X[:,drop:]
        bin_st = n_bins*n_octaves-n_bins*(i+1)
        bin_ed = n_bins*n_octaves-n_bins*i
        frm_st = 0
        frm_ed = X.shape[1]*2**i
        frm_step = 2**i
        if frm_ed > CQ.shape[1]:
            # CQ = sp.resize(CQ, (CQ.shape[0],frm_vec[-1]+1))
            # CQ[:,CQ.shape[1]+1:frm_vec[-1]] = 0.0
            CQ = sp.c_[CQ, sp.zeros((CQ.shape[0],frm_ed-CQ.shape[1]))]
            
        # CQ[bin_st:bin_ed, frm_st:frm_ed:frm_step] = X.copy()
        CQ[bin_st:bin_ed, frm_st:frm_ed:frm_step] = X.copy()

    return CQ

def _ndarray2list(arrayCQ, n_bins, n_octaves, n_atom, firstcenter, atom_hop):
    """
    ndarrayに統合されたCQTをオクターブごとのCQT結果が格納されたlistに変換する
    """
    empty_hops = firstcenter / atom_hop
    listCQ = []

    for i in xrange(n_octaves):
        dropped = empty_hops*2**(n_octaves-i-1)-empty_hops
        X = arrayCQ[n_bins*n_octaves-(i+1)*n_bins:n_bins*n_octaves-i*n_bins,::2**i]
        
        X = sp.c_[sp.zeros( (n_bins,dropped)), X]
        X = sp.c_[X, sp.zeros((n_bins,sp.ceil(X.shape[1]/n_atom)*n_atom-X.shape[1]))]

        if n_atom > 1: #reshape
            Xcell = sp.zeros( (n_bins*n_atom,sp.ceil(X.shape[1]/n_atom)), dtype=complex)
            for u in xrange(n_bins):
                Xbin = sp.reshape(X[u,:], (n_atom,len(X[u,:])/n_atom), order='F').copy()
                Xcell[u*n_atom:(u+1)*n_atom,:] = Xbin

            listCQ.append(Xcell)
        else:
            listCQ.append(X)
            
    return listCQ
    
def make_oct_cqt_kernels(fmax, n_bins, fs, q=1.0, atom_hop_factor=0.25, thr=0.0005, window='blackmanharris', perf_rast=False):
    """
    CQTのKernel設計(Klapuri CQT用)
    """
    def nextpow2(i):
        # n = 2
        # while n < i:
            # n = n * 2
        n = int(sp.ceil(sp.log2(i)))
        return n

    # define
    fmin = (fmax/2)*2**(1.0/float(n_bins))
    Q = 1.0/(2**(1/float(n_bins))-1)
    Q = Q*q
    Nk_max = Q * fs / float(fmin)
    Nk_max = round(Nk_max) # length of the largest atom [samples]

    # Compute FFT size, FFT hop, atom hop,
    Nk_min = round( Q * fs / float(fmin*2**((n_bins-1)/float(n_bins))) ) # length of the shortest atom [samples]
    atom_hop = round(Nk_min*atom_hop_factor) # atom hop size
    first_center = sp.ceil(Nk_max/2) # first possible center position within the frame
    first_center = atom_hop * sp.ceil(first_center/atom_hop) # lock the first center to an integer multiple of the atom hop size
    # nextpow2(x)はScipyにはないので代替手段を考える。要はx以上で最小の2のべき乗数を見つけるだけ。
    # fftsize = 2**nextpow2(first_center+sp.ceil(Nk_max/2)) # use smallest possible FFT size (increase sparsity)
    fftsize = 2**nextpow2(first_center+sp.ceil(Nk_max/2))

    if perf_rast:
        winlen = sp.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop) # number of temporal atoms per FFT Frame
        if winlen == 0:
            fftsize = fftsize * 2
            winlen = sp.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop)
    else:
        winlen = sp.floor((fftsize-sp.ceil(Nk_max/2)-first_center)/atom_hop)+1 # number of temporal atoms per FFT Frame

    last_center = first_center + (winlen-1)*atom_hop
    hopsize = (last_center + atom_hop) - first_center # hop size of FFT frames
    fft_overlap = (fftsize-hopsize/fftsize)*100 # overlap of FFT frames in percent ***AK:needed?

    # init variables
    temp_kernel = sp.zeros( fftsize, dtype=complex )
    spec_kernel = sp.zeros( (n_bins*winlen,fftsize), dtype=complex )

    # Compute kernel
    atom_ind = 0;
    idx = 0
    for k in xrange(n_bins):
        Nk = round( Q * fs / float(fmin*2**(k/float(n_bins))) ) # N[k] = (fs/fk)*Q. Rounding will be omitted in future versions
        
        win_func = sp.sqrt(sp.signal.get_window(window, Nk))
        # win_func = sp.signal.get_window(window, Nk)
        
        fk = fmin*2**(k/float(n_bins))
        temp_kernel_bin = (win_func/float(Nk)) * sp.exp(2*sp.pi*1j*fk*sp.arange(Nk)/float(fs))
        atom_offset = first_center - sp.ceil(Nk/2.0)

        for i in xrange(int(winlen)):
            shift = atom_offset + (i * atom_hop)
            temp_kernel[shift:Nk+shift] = temp_kernel_bin
            atom_ind = atom_ind+1
            _spec_kernel = sp.fftpack.fft(temp_kernel)
            _spec_kernel[sp.absolute(_spec_kernel)<=thr] = 0
            spec_kernel[idx] = _spec_kernel
            # spar_kernel = sparse(sp.r_[spar_kernel,spec_kernel])
            temp_kernel = sp.zeros( fftsize, dtype=complex ) # reset window
            idx += 1

    # spar_kernel = (spar_kernel.T)/fftsize
    spec_kernel = (spec_kernel.T)/fftsize

    # Normalize the magnitudes of the atoms
    wx1 = sp.argmax(sp.absolute(spec_kernel[:,0])) #matlabのmaxと挙動を合わせるためabsを取る
    wx2 = sp.argmax(sp.absolute(spec_kernel[:,-1])) #上に同じ
    wK = spec_kernel[wx1:wx2+1,:]
    wK = sp.diag(sp.dot(wK, wK.conjugate().T))
    wK = wK[round(1/q):(len(wK)-round(1/q)-2)]
    weight = 1./sp.mean(sp.absolute(wK))
    weight = weight *(hopsize/float(fftsize))
    weight = sp.sqrt(weight) # sqrt because the same weight is applied in icqt again
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
def cqt_oct(x, fmin, fmax, n_bins, fs, q=1.0, atom_hop_factor=0.25, thr=0.0005, window='blackmanharris'):
    """
    CQT計算(Schorkhuber,Klapuriによるオクターブごとに一気に計算する手法)
    CQT toolbox(http://www.eecs.qmul.ac.uk/~anssik/cqt/)のMATLABコードを移植したもの。

    """
    # define
    n_octaves = sp.ceil(sp.log2(fmax/fmin)).astype('int')
    fmin = (fmax/2**n_octaves) * 2**(1/n_bins) # set fmin to actual value
    xlen_init = len(x)

    # design lowpass filter
    LPorder = 6 # order of the anti-aliasing filter
    cutoff = 0.5
    B,A = sp.signal.butter(LPorder, cutoff, 'low') # design f_nyquist/2-lowpass filter

    # design kernel for one octave 
    cqt_kernel,params = make_oct_cqt_kernels(fmax, n_bins, fs, q=q, atom_hop_factor=atom_hop_factor, thr=thr, window=window)

    # calculate CQT
    listCQ = []
    max_block = params['fftsize'] * 2**(n_octaves-1) # largest FFT Block (virtual)
    prefix_zeros = max_block
    suffix_zeros = max_block
    x_zp = sp.r_[sp.zeros(prefix_zeros), x, sp.zeros(suffix_zeros)] # zeropadding
    overlap = params['fftsize'] - params['hopsize']
    K = cqt_kernel.conjugate().T # conjugate spectral kernel for cqt transformation

    for i in xrange(n_octaves):
        framed = makeFramedData(x_zp, params['fftsize'], params['hopsize'], 'boxcar')
        XX = sp.fftpack.fft(framed, params['fftsize']) # applying fft to each column (each FFT frame)
        CQ_oct = sp.dot(K, XX.T)
        listCQ.append(CQ_oct) # calculating cqt coefficients for all FFT frames for this octave

        if i != n_octaves-1:
            x_zp = sp.signal.filtfilt(B, A, x_zp) # anti aliasing filter
            x_zp = x_zp[0::2] # drop samplerate by 2

    # map to sparse matrix representation
    arrayCQ = _list2ndarray(listCQ, n_octaves, n_bins, params['firstcenter'], params['atom_hop'], params['winlen'])

    intCQ = interpolate_cqt(arrayCQ, n_octaves, n_bins)
    
    # discard prefix suffix zero spectrum
    empty_hops = params['firstcenter'] / params['atom_hop']
    max_drop = empty_hops*2**(n_octaves-1) - empty_hops
    dropped_samples = (max_drop-1)*params['atom_hop'] + params['firstcenter']
    output_time_vec = sp.arange(intCQ.shape[1]) * params['atom_hop'] - prefix_zeros + dropped_samples
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
    params['true_fftsize'] = params['fftsize'] / params['winlen']
    params['true_hopsize'] = params['hopsize'] / params['winlen']
    
    return arrayCQ, intCQ_dropped, cqt_kernel, params

def interpolate_cqt(CQ, n_octaves, n_bins):
    intCQ = sp.zeros( CQ.shape )
    
    for k in xrange(CQ.shape[0]):
        oct = int(n_octaves-sp.floor((k+1-0.1)/float(n_bins)))
        step_vec = sp.arange(0,CQ.shape[1],2**(oct-1))
        Xbin = CQ[k,step_vec]
        intCQ[k,:] = sp.interp(sp.arange(CQ.shape[1]), step_vec, sp.absolute(Xbin))
        
    return intCQ

def cqt_oct_intp(x, fmin, fmax, n_bins, fs, q=1.0, atom_hop_factor=0.25, thr=0.0005, window='blackmanharris'):
    """
    CQT(補間済みのCQT結果とfftsize,hopsizeのみを返す)
    """
    arrayCQ,CQ,K,prms = cqt_oct(x, fmin, fmax, n_bins, fs, q, atom_hop_factor, thr, window)
    fftsize = prms['true_fftsize']
    hopsize = prms['true_hopsize']
    return CQ,fftsize,hopsize

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
