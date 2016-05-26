# -*- coding: utf-8 -*-

"""
@file beta_nmf_cy.pyx
@brief Beta-divergence NMF (cython version)
@author ふぇいと (@stfate)

@description

"""

import cython
import numpy as np
cimport numpy as np


def beta_nmf_cy(X, n_basis, b=1.0, n_iter=100, show_div=True, Winit=None):
    """
        beta-divergenceによるNMF (全てのbで収束性を保証)
    
        Parameters:
            X, nBasis, nIter -- 他のnmf_**と同じ
            b -- beta-divergenceのbeta(0-2)
        Returns:
            他のnmf_**と同じ
    """
    cdef float SPARSENESS = 0.001
    cdef int nF = X.shape[0]
    cdef int nT = X.shape[1]
    cdef int _n_iter = n_iter
    cdef int _b = b
    cdef np.ndarray[double,ndim=2] _X = X.copy()
    cdef np.ndarray[double,ndim=2] W = None
    if Winit != None:
        W = Winit
    else:
        W = np.random.rand( nF, n_basis ) * _X.mean()
    cdef np.ndarray[double,ndim=2] H = np.random.rand( n_basis, nT ) * _X.mean()
    
    cdef double D = 1e99
    cdef double phiB = _calcPhi(_b)
    cdef int it
    cdef np.ndarray[double,ndim=2] WH = np.zeros( (nF,nT), dtype=np.double )
    
    for it from 0 <= it < _n_iter:
        # print it
        WH = np.dot(W, H)
        W *= (np.dot( _X * (WH**(_b-2)), H.T ) / np.dot(WH**(_b-1), H.T))**phiB
        # W /= (W**2).sum(0) + 0.0000001
        WH = np.dot(W, H)
        H *= (np.dot( W.T, _X * (WH**(_b-2))) / (SPARSENESS + np.dot(W.T, WH**(_b-1))))**phiB
        # H *= (np.dot( W.T, _X * (WH**(_b-2))) / (np.dot(W.T, WH**(_b-1))))**phiB
        H /= H.max()
        
        if show_div:
            WH = np.dot(W, H)
            D = _calcBetaDivergence(WH, _X, _b)
            print 'Beta-divergence: {0}'.format(D)
        
    return W,H
    
def beta_nmf_activation_cy(X, W, b=1.0, n_iter=100, show_div=True):
    """
        基底を固定しての\beta-NMF (全てのbで収束性を保証)
    """
    cdef float SPARSENESS = 0.001
    cdef int n_basis = W.shape[1]
    cdef int nF = X.shape[0]
    cdef int nT = X.shape[1]
    
    H = np.ones( (n_basis, nT), dtype=np.double )
    # cdef np.ndarray[double,ndim=2] H = np.random.rand( n_basis, nT ) * _X.mean()
    
    cdef double D = 1e99
    cdef double phiB = _calcPhi(b)
    cdef int it
    cdef np.ndarray[double,ndim=2] WH = np.zeros( (nF,nT), dtype=np.double )
    
    for it from 0 <= it < n_iter:
        WH = np.dot(W, H)
        H *= (np.dot( W.T, X * (WH**(b-2))) / (SPARSENESS + np.dot(W.T, WH**(b-1))))**phiB
        # H *= (np.dot( W.T, _X * (WH**(_b-2))) / (np.dot(W.T, WH**(_b-1))))**phiB
        H /= H.max()
        
        if show_div:
            WH = np.dot(W,H)
            D = _calcBetaDivergence(WH, X, b)
            print 'Beta-divergence: {0}'.format(D)
        
    return H
    
def beta_nmf_activation_other_cy(X, W, b=1.0, n_iter=100, n_basis_undesired=20, show_div=True):
    """
        基底を固定しての\beta-NMF (全てのbで収束性を保証)
        教師基底Wで表せない成分をWu*Huで表す．
        
        実装してみたが想定通りの動作をしていない．調べる．
        -> HALION ONEで作成したpiano+violin重畳音で試したところ
           2classの教師ありNMFとして正常に動いている模様．
           初期値が適切でなかったのかもしれない．

        b=1.0の場合(Iダイバージェンスのとき)、
    """
    cdef float SPARSENESS = 0.001
    cdef int n_basis = W.shape[1]
    cdef int nF = X.shape[0]
    cdef int nT = X.shape[1]
    
    # cdef np.ndarray[double,ndim=2] H = np.ones( (n_basis, nT), dtype=float )
    cdef np.ndarray[double,ndim=2] H = np.random.rand(n_basis, nT)
    cdef np.ndarray[double,ndim=2] Wu = np.random.rand( nF, n_basis_undesired )
    cdef np.ndarray[double,ndim=2] Hu = np.random.rand( n_basis_undesired, nT )
    
    cdef float D = 1e99
    cdef float mu = 1e6
    cdef float phiB = _calcPhi(b)
    
    cdef int it
    for it in xrange(n_iter):
        WH = np.dot(W, H) + np.dot(Wu, Hu)
        H *= (np.dot( W.T, X * (WH**(b-2))) / (SPARSENESS + np.dot(W.T, WH**(b-1))))**phiB
        H /= H.max()
        WH = np.dot(W,H) + np.dot(Wu,Hu)
        # Wu *= (np.dot( X * (WH**(b-2)), Hu.T) / np.dot(WH**(b-1), Hu.T))**phiB
        ortho_factor = mu * np.dot(Wu.T, np.sum(W**2,axis=1)) # 基底の直交化のための罰則項
        Wu *= (np.dot( X * (WH**(b-2)), Hu.T) / (np.dot(WH**(b-1), Hu.T) + ortho_factor))**phiB
        Wu /= (Wu**2).sum(0) + 0.0000001
        WH = np.dot(W, H) + np.dot(Wu, Hu)
        Hu *= (np.dot( Wu.T, X * (WH**(b-2))) / (SPARSENESS + np.dot(Wu.T, WH**(b-1))))**phiB
        Hu /= Hu.max()
        
        if show_div:
            WH = np.dot(W, H) + np.dot(Wu, Hu)
            D = _calcBetaDivergence(WH, X, b)
            print 'Beta-divergence: {0}'.format(D)
        
    # return H,Wu,Hu
    return H
    
def beta_nmf_activation_multiclass_cy(X, Wc, b=1.0, n_iter=500, show_div=True):
    """
        クラスごとに基底を学習した教師ありNMF
        
        Parameters:
            X: 入力スペクトログラム
            Wc: クラスごとの基底行列のlist
            b: beta-factor
            n_iter: NMFのiteration数
            
        Returns:
            H: クラス毎のアクティベーションのlist
    """
    cdef float SPARSENESS = 0.001
    cdef int n_class = len(Wc)
    cdef int nF = X.shape[0]
    cdef int nT = X.shape[1]
    cdef int _b = b
    cdef int _n_iter = n_iter
    cdef np.ndarray[double,ndim=2] _X = X.copy()
    
    n_basis_list = [_W.shape[1] for _W in Wc]
    # Hc = [np.ones( (n_basis, nT), dtype=np.double) for n_basis in n_basis_list]
    Hc = [np.random.rand(n_basis, nT) * _X.mean() for n_basis in n_basis_list]
    
    cdef double D = 1e99
    cdef double phiB = _calcPhi(_b)
    cdef int it,c
    cdef np.ndarray[double,ndim=2] WH = np.zeros( (nF,nT), dtype=np.double )
    
    for it from 0 <= it < _n_iter:
        for c from 0 <= c < n_class:
            WH = _computeWH(Wc, Hc)
            Hc[c] *= (np.dot( Wc[c].T, _X * (WH**(_b-2))) / (SPARSENESS + np.dot(Wc[c].T, WH**(_b-1))))**phiB
            Hc[c] /= Hc[c].max()
            
        if show_div:
            WH = _computeWH(Wc, Hc)
            D = _calcBetaDivergence(WH, _X, _b)
            print 'Beta-divergence: {0}'.format(D)
        
    return Hc

cdef _computeWH(Wc, Hc):
    cdef np.ndarray[double,ndim=2] WH = np.zeros( (Wc[0].shape[0],Hc[0].shape[1]) )
    cdef int N = len(Wc)
    cdef int c
    for c from 0 <= c < N:
        WH += np.dot(Wc[c], Hc[c])
    
    return WH


""" helper functions """

cdef double _calcBetaDivergence(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y, double b):
    """
        \beta-divergenceを計算する
    
        Parameters:
            X: NMFで推定したスペクトログラム
            Y: 真のスペクトログラム(=入力)
            b: beta-factor
    
        Returns:
            beta-divergenceの値
    """
    if b == 1:
        d = np.sum(Y * np.log(Y/X) + (X-Y))
    elif b == 0:
        d = np.sum(Y/X - np.log(Y/X) - 1)
    else:
        d = np.sum(Y**b / (b * (b-1)) + (X**b) / b - Y * (X**(b-1)) / (b-1))
    
    return d
    
cdef double _calcPhi(double b):
    """
        収束性を保証した\beta-NMFで用いる\phiを計算する
    """
    if b < 1:
        return 1.0 / (2.0 - b)
    elif b > 2:
        return 1.0 / (b - 1.0)
    else:
        return 1.0
        