# -*- coding: utf-8 -*-

"""
@file conv_nmf.py
@brief Convolutive NMF
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from multiprocessing import Pool


def conv_nmf(X, n_basis, T=8, n_iter=100):
    """
    convolutive NMF

    P.Smaragdis, ''Convolutive Speech Bases and their Application to Supervised Speech Separation''
    (IEEE TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, 2007)をそのまま実装したもの．
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    W = (sp.rand( T, nF, n_basis ) + 10) * X.mean()
    H = (sp.rand( n_basis, nT ) + 10) * X.mean()
    Hcand = (sp.rand(T, n_basis, nT) + 10) * X.mean()
    one_mat = sp.ones((nF,nT))

    for i in xrange(n_iter):
        # W[t]をすべて更新
        WH = computeWH(W, H)
        for t in xrange(T):
            W[t] *= sp.dot(X/WH, _rshift(H,t).T) / sp.dot(one_mat, _rshift(H,t).T)
        # すべてのtに対してそれぞれ更新されたHを求め平均する
        WH = computeWH(W, H)
        for t in xrange(T):
            Hcand[t] = H * sp.dot(W[t].T, _lshift(X/WH,t)) / (SPARSENESS + sp.dot(W[t].T, one_mat))
        H = Hcand.mean(0)
            
        WH = computeWH(W, H)
        D = _calcKLdivergence(X, WH)
        print D

    return W,H

def updateW(X, W, H, t, one_mat):
    WH = computeWH(W, H)
    W[t] *= sp.dot(X/WH, _matrix_rshift(H,t).T) / sp.dot(one_mat, _matrix_rshift(H,t).T)

    return W

def updateH(X, W, Hcand, H, t, one_mat):
    WH = computeWH(W, H)
    Hcand[t] = H * sp.dot(W[t].T, _matrix_lshift(X/WH,t)) / (SPARSENESS + sp.dot(W[t].T, one_mat))

    return Hcand

def conv_nmf_activation(X, W, n_iter=100):
    """
    事前学習した基底を用いてのconvolutive NMF
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    T = W.shape[0]
    n_basis = W.shape[2]
    H = (sp.rand( n_basis, nT ) + 10) * X.mean()
    Hcand = (sp.rand(T, n_basis, nT) + 10) * X.mean()
    one_mat = sp.ones((nF,nT))

    for i in xrange(n_iter):
        WH = computeWH(W, H)
        for t in xrange(T):
            Hcand[t] = H * sp.dot(W[t].T, _lshift(X/WH,t)) / (SPARSENESS + sp.dot(W[t].T, one_mat))
        H = Hcand.mean(0)
            
        WH = computeWH(W, H)
        D = _calcKLdivergence(X, WH)
        print D

    return H

def computeWH(W, H):
    """
    WとHからスペクトログラムを再合成する
    """
    T = W.shape[0]
    Y = sp.zeros( (W.shape[1],H.shape[1]) )
    for t in xrange(T):
        Y += sp.dot(W[t], _matrix_rshift(H, t))

    return Y

""" helper functions """

def _rshift(X, t):
    if t == 0:
        return X
    else:
        X2 = sp.c_[sp.zeros((X.shape[0],t)), X[:,:-t]]
        return X2

def _lshift(X, t):
    if t == 0:
        return X
    else:
        X2 = sp.c_[X[:,t:], sp.zeros((X.shape[0],t))]
        return X2

def _calcKLdivergence(X, WH):
    return (X * sp.log(X/WH) - X + WH).sum()
