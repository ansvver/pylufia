# -*- coding: utf-8 -*-

import scipy as sp


def beta_nmf(X, b=1.0, n_basis=10, n_iter=100, show_div=True, Winit=None):
    """
        beta-divergenceによるNMF (全てのbで収束性を保証)
    
        Parameters:
            X, nBasis, nIter -- 他のnmf_**と同じ
            b -- beta-divergenceのbeta(0-2)
        Returns:
            他のnmf_**と同じ
    """
    SPARSENESS = 0.001
    nF,nT = X.shape
    # W = sp.rand( nF,nBasis ) + 1.0
    # H = sp.rand( nBasis,nT ) + 1.0
    
    # X += 1.0
    # X += 0.00001
    if Winit != None:
        W = Winit
    else:
        W = (sp.rand( nF, n_basis ) + 10)*X.mean()
    H = (sp.rand( n_basis, nT ) + 10)*X.mean()
    
    D = 1e99
    prevD = 1e100
    phiB = _calcPhi(b)
    
    for it in range(n_iter):
        # print(it)
        WH = sp.dot(W,H)
        W *= (sp.dot( X * (WH**(b-2)), H.T ) / sp.dot(WH**(b-1), H.T))**phiB
        # W /= (W**2).sum(0) + 0.0000001
        WH = sp.dot(W,H)
        H *= (sp.dot( W.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(W.T, WH**(b-1))))**phiB
        # H /= H.max()
        
        if show_div:
            WH = sp.dot(W,H)
            prevD = D
            D = _calcBetaDivergence(WH, X, b)
            print( 'Beta-divergence: {0}'.format(D) )
        
        # if it > 10 and D > prevD:
            # break
        
    return W,H
    
def beta_nmf_activation(X, W, b=1.0, n_iter=100, show_div=True):
    """
        基底を固定しての\beta-NMF (全てのbで収束性を保証)
    """
    SPARSENESS = 0.001
    n_basis = W.shape[1]
    nF,nT = X.shape
    
    H = sp.ones( (n_basis, nT) )
    
    D = 1e99
    prevD = 1e100
    phiB = _calcPhi(b)
    
    for it in range(n_iter):
        WH = sp.dot(W,H)
        H *= (sp.dot( W.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(W.T, WH**(b-1))))**phiB
        # H /= H.max()
        
        if show_div:
            WH = sp.dot(W,H)
            prevD = D
            D = _calcBetaDivergence(WH, X, b)
            print( 'Beta-divergence: {0}'.format(D) )
        # if it > 10 and D > prevD:
            # break
        
    return H
    
def beta_nmf_activation_other(X, W, b=1.0, n_iter=100, n_basis_undesired=20, show_div=True):
    """
        基底を固定しての\beta-NMF (全てのbで収束性を保証)
        教師基底Wで表せない成分をWu*Huで表す．
        
        実装してみたが想定通りの動作をしていない．調べる．
        -> HALION ONEで作成したpiano+violin重畳音で試したところ
           2classの教師ありNMFとして正常に動いている模様．
           初期値が適切でなかったのかもしれない．
    """
    SPARSENESS = 0.001
    n_basis = W.shape[1]
    nF,nT = X.shape
    
    H = sp.ones( (n_basis, nT) )
    # Wu = (sp.rand( nF, n_basis_undesired ) + 10)*X.mean()
    # Hu = (sp.rand( n_basis_undesired, nT ) + 10)*X.mean()
    # Wu = sp.rand( nF, n_basis_undesired ) + 10
    # Hu = sp.rand( n_basis_undesired, nT ) + 10
    Wu = sp.ones( (nF, n_basis_undesired) )
    Hu = sp.ones( (n_basis_undesired, nT) )
    
    D = 1e99
    prevD = 1e100
    mu = 1e6
    phiB = _calcPhi(b)
    
    for it in range(n_iter):
        WH = sp.dot(W,H) + sp.dot(Wu,Hu)
        H *= (sp.dot( W.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(W.T, WH**(b-1))))**phiB
        WH = sp.dot(W,H) + sp.dot(Wu,Hu)
        # Wu *= (sp.dot( X * (WH**(b-2)), Hu.T) / sp.dot(WH**(b-1), Hu.T))**phiB
        ortho_factor = mu*sp.dot(Wu.T,sp.sum(W**2,axis=1)) # 基底の直交化のための罰則項
        Wu *= (sp.dot( X * (WH**(b-2)), Hu.T) / (sp.dot(WH**(b-1), Hu.T) + ortho_factor))**phiB
        Wu /= (Wu**2).sum(0) + 0.0000001
        WH = sp.dot(W,H) + sp.dot(Wu,Hu)
        Hu *= (sp.dot( Wu.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(Wu.T, WH**(b-1))))**phiB
        
        if show_div:
            WH = sp.dot(W,H) + sp.dot(Wu,Hu)
            prevD = D
            D = _calcBetaDivergence(WH, X, b)
            print( 'Beta-divergence: {0}'.format(D) )
        
    return H,Wu,Hu
    
def beta_nmf_activation_hp(X, Wh, Wp, b=1.0, n_iter=100, show_div=True):
    """
        調波成分，非調波成分それぞれの基底を利用する教師ありNMF
        
        Parameters:
            X: 入力スペクトログラム
            Wh: 調波成分の基底行列
            Wp: 非調波成分の基底行列
            b: beta-factor
            n_iter: NMFのiteration数
            
        Returns:
            Hh: 調波成分アクティベーション
            Hp: 非調波成分アクティベーション
    """
    SPARSENESS = 0.001
    n_basis_h = Wh.shape[1]
    n_basis_p = Wp.shape[1]
    nF,nT = X.shape
    
    Hh = sp.ones( (n_basis_h, nT) )
    Hp = sp.ones( (n_basis_p, nT) )
    
    D = 1e99
    prevD = 1e100
    phiB = _calcPhi(b)
    
    for it in range(n_iter):
        WH = sp.dot(Wh,Hh) + sp.dot(Wp,Hp)
        Hh *= (sp.dot( Wh.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(Wh.T, WH**(b-1))))**phiB
        Hh /= Hh.max()
        WH = sp.dot(Wh,Hh) + sp.dot(Wp,Hp)
        Hp *= (sp.dot( Wp.T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(Wp.T, WH**(b-1))))**phiB
        Hp /= Hp.max()
        
        if show_div:
            WH = sp.dot(Wh,Hh) + sp.dot(Wp,Hp)
            prevD = D
            D = _calcBetaDivergence(WH, X, b)
        
    return Hh,Hp
    
def beta_nmf_activation_multiclass(X, Wc, b=1.0, n_iter=100, show_div=True):
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
    def computeWH(Wc,Hc):
        WH = sp.zeros( (Wc[0].shape[0],Hc[0].shape[1]) )
        for _W,_H in zip(Wc,Hc):
            WH += sp.dot(_W,_H)
            
        return WH
    
    SPARSENESS = 0.001
    n_class = len(Wc)
    nF,nT = X.shape
    n_basis_list = [_W.shape[1] for _W in Wc]
    Hc = [sp.ones( (n_basis, nT) ) for n_basis in n_basis_list]
    
    D = 1e99
    prevD = 1e100
    phiB = _calcPhi(b)
    
    for it in range(n_iter):
        for c in range(n_class):
            WH = computeWH(Wc,Hc)
            Hc[c] *= (sp.dot( Wc[c].T, X * (WH**(b-2))) / (SPARSENESS + sp.dot(Wc[c].T, WH**(b-1))))**phiB
            # Hc[c] *= (sp.dot( Wc[c].T, X * (WH**(b-2))) / sp.dot(Wc[c].T, WH**(b-1)))*phiB
            Hc[c] /= Hc[c].max()
            
        if show_div:
            WH = computeWH(Wc,Hc)
            prevD = D
            D = _calcBetaDivergence(WH, X, b)
            print( 'Beta-divergence: {0}'.format(D) )
        
    return Hc

def saveBasis(W, fname_out, show_plot=False):
    """
        基底Wのプロットを保存
    """
    # n_freq,n_basis = W.shape
    # Wmax = W.max()
    # for i,_w in enumerate(W.T):
    #     pp.subplots_adjust(wspace=0)
    #     ax = pp.subplot(1, n_basis, i+1)
    #     pp.plot(_w, sp.arange(n_freq))
    #     pp.axis([0,Wmax,0,n_freq])
    #     if i != 0:
    #         pp.setp(ax.get_yticklabels(), visible=False)
    # pp.savefig(fname_out)
    
    # if show_plot == True:
    #     pp.show()
    pass
    
""" helper functions """
    
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