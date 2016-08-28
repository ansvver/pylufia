# -*- coding: utf-8 -*-

"""
Convolutive NMF (beta-divergence)
"""

import scipy as sp

def conv_beta_nmf(X, n_basis, b=1.0, T=8, n_iter=100, show_div=True, Winit=None):
    """
    Beta-divergence規準によるConvolutive NMF
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    if Winit != None:
        W = Winit
    else:
        W = (sp.rand( T, nF, n_basis ) + 10) * X.mean()
    H = (sp.rand( n_basis, nT ) + 10) * X.mean()
    Hcand = (sp.rand(T, n_basis, nT) + 10) * X.mean()
    phiB = _calcPhi(b)

    for i in range(n_iter):
        # W[t]をすべて更新
        WH = computeWH(W, H)
        for t in range(T):
            W[t] *= (sp.dot(X/(WH**(2-b)), _rshift(H,t).T) / sp.dot(WH**(b-1), _rshift(H,t).T))**phiB
        # すべてのtに対してそれぞれ更新されたHを求め平均する
        WH = computeWH(W, H)
        for t in range(T):
            Hcand[t] = H * (sp.dot(W[t].T, _lshift(X / (WH**(2-b)),t)) / (SPARSENESS + sp.dot(W[t].T, _lshift(WH,t)**(b-1))))**phiB
        H = Hcand.mean(0)
        
        if show_div:
            WH = computeWH(W, H)
            D = _calcBetaDivergence(X, WH, b)
            print( 'Beta-divergence={0}'.format(D) )

    return W,H

def computeWH(W, H):
    """
    WとHからスペクトログラムを再合成する
    """
    T = W.shape[0]
    Y = sp.zeros( (W.shape[1],H.shape[1]) )
    for t in range(T):
        Y += sp.dot(W[t], _rshift(H, t))

    return Y

def conv_beta_nmf_activation(X, W, b=1.0, n_iter=100, show_div=True):
    """
    事前学習した基底を用いてのconvolutive NMF
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    T = W.shape[0]
    n_basis = W.shape[2]
    # H = (sp.rand( n_basis, nT ) + 10) * X.mean()
    # Hcand = (sp.rand(T, n_basis, nT) + 10) * X.mean()
    H = sp.ones( (n_basis, nT) )
    Hcand = sp.rand(T, n_basis, nT)
    phiB = _calcPhi(b)

    for i in range(n_iter):
        WH = computeWH(W, H)
        for t in range(T):
            Hcand[t] = H * (sp.dot(W[t].T, _lshift(X / (WH**(2-b)),t)) / (SPARSENESS + sp.dot(W[t].T, _lshift(WH,t)**(b-1))))**phiB
        H = Hcand.mean(0)
        
        if show_div:
            WH = computeWH(W, H)
            D = _calcBetaDivergence(X, WH, b)
            print( 'Beta-divergence={0}'.format(D) )

    return H

def conv_beta_nmf_activation_multiclass(X, Wc, b=1.0, n_iter=100, show_div=True):
    """
    事前学習した基底を用いてのconvolutive NMF
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    T = Wc[0].shape[0]
    n_basis = Wc[0].shape[2]
    # H = (sp.rand( n_basis, nT ) + 10) * X.mean()
    # Hcand = (sp.rand(T, n_basis, nT) + 10) * X.mean()
    # H = sp.ones( (n_basis, nT) )
    n_basis_list = [W.shape[2] for W in Wc]
    # Hc = [np.ones( (n_basis, nT), dtype=np.double) for n_basis in n_basis_list]
    Hc = [sp.rand(n_basis, nT) * X.mean() for n_basis in n_basis_list]
    Hcand = sp.rand(T, n_basis, nT)
    phiB = _calcPhi(b)

    for i in range(n_iter):
        for c in range(len(Wc)):
            # WH = computeWH(W[c], Hc[c])
            WH = computeMulticlassWH(Wc, Hc)
            for t in range(T):
                Hcand[t] = Hc[c] * (sp.dot(Wc[c][t].T, _lshift(X / (WH**(2-b)),t)) / (SPARSENESS + sp.dot(Wc[c][t].T, _lshift(WH,t)**(b-1))))**phiB
            Hc[c] = Hcand.mean(0)
            

        if show_div:
            WH = computeMulticlassWH(Wc, Hc)
            D = _calcBetaDivergence(X, WH, b)
            print( 'Beta-divergence={0}'.format(D) )

    return Hc

def computeMulticlassWH(Wc, Hc):
    WH = sp.zeros( (Wc[0].shape[1],Hc[0].shape[1]) )

    for c in range(len(Wc)):
        WH += computeWH(Wc[c], Hc[c])

    return WH


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

def _calcBetaDivergence(X, Y, b):
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
        d = (Y*(sp.log(Y)-sp.log(X)) + (X-Y)).sum()
    elif b == 0:
        d = (Y/X - sp.log(Y/X) - 1).sum()
    else:
        d = (Y**b/(b*(b-1)) + (X**b)/b - Y*(X**(b-1))/(b-1)).sum()
        
    return d
    
def _calcPhi(b):
    """
        収束性を保証した\beta-NMFで用いる\phiを計算する
    """
    if b < 1:
        return 1.0/(2.0-b)
    elif b > 2:
        return 1.0/(b-1.0)
    else:
        return 1.0