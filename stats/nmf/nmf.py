# -*- coding: utf-8 -*-

"""
@file nmf.py
@brief NMF
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
# import matplotlib.pyplot as pp


def nmf_euclid(X, n_basis, conv_thres):
    """
    Calculate NMF using euclid distance
    (Implemented by Maezawa-san)
    
    Parameters:
      X: ndarray
        input matrix
      nBasis: int
        number of basis vectors
      convThres: float
        threshold for convergence
      dist: string
        type of distance function for cost calculation
        ('euclid' or 'KL' or 'IS')
    
    Results:
      W: ndarray
        Basis matrix
      H: ndarray
        Activation matrix
    
    """

    # 必要な変数の準備
    W = sp.rand( X.shape[0],n_basis ) + 1.0
    H = sp.rand( n_basis,X.shape[1] ) + 1.0
    ratio = 1e40
    cost = 1e40
    na = sp.newaxis

    # 本筋
    while ratio > conv_thres:
        W *= sp.dot(X, H.T) / sp.dot(W, sp.dot(H, H.T))
        H *= sp.dot(W.T, X) / sp.dot(sp.dot(W.T, W), H)

        cost_cur = ( (X - (W[:, :, na] * H[na, :, :]).sum(1))**2 ).sum()
        ratio, cost = (cost-cost_cur)/cost_cur , cost_cur

    return W, H
    
def nmf_kl(X, n_basis, conv_thres):
    """
    Calculate NMF using KL divergence
    (Implemented by Maezawa-san)
    
    Parameters:
      X: ndarray
        input matrix
      nBasis: int
        number of basis vectors
      convThres: float
        threshold for convergence
      dist: string
        type of distance function for cost calculation
        ('euclid' or 'KL' or 'IS')
    
    Results:
      W: ndarray
        Basis matrix
      H: ndarray
        Activation matrix
    
    """

    # 必要な変数の準備
    W = sp.rand( X.shape[0],n_basis ) + 1.0
    H = sp.rand( n_basis,X.shape[1] ) + 1.0
    ratio = 1e40
    cost = 1e40
    na = sp.newaxis

    # 本筋
    while ratio > conv_thres:
        H *= ( W[:,:,na]*X[:,na,:] / ( sp.dot(W,H)[:,na,:]) ).sum(0) / ( W.sum(0) )[:,na]
        W *= ( H[na,:,:]*X[:,na,:] / ( sp.dot(W,H)[:,na,:]) ).sum(2) / ( H.sum(1) )[na,:]
    
        cost_cur = ( X*sp.log(X/sp.dot(W,H))-X+sp.dot(W,H) ).sum()
        ratio, cost = (cost-cost_cur)/cost_cur , cost_cur

    return W, H

def nmf_is(X, n_basis, n_iter):
    '''
        Implemented by Maezawa-san
        
        板倉斎藤距離を最小化するような、
        基底とアクチベーションを推定する。
        
        [引数]
          X - 入力アレイ（２次元; F-by-T）
          nBasis - 基底の数
          nIter - 反復回数
        
        [戻り値]
          W - 基底ベクトルの配列 ( F-by-nBasis )
          H - アクチベーションの配列 (nBasis-by-T)
        
        scipy.dot(W,H)はXに板倉斎藤距離の意味で近い。
    '''
    SPARSENESS = 0.001 # IS-NMFにおける、L2正則化項
    nF,nT = X.shape
    X+=1.0
    H = (sp.rand( n_basis, nT ) + 10)*X.mean()
    W = (sp.rand( nF, n_basis ) + 10)*X.mean()
    # dot = scipy.dot
    pISD = 1e100
    for it in xrange(n_iter):
        idWH = 1.0/sp.dot(W,H)
        print it
        isd = X*idWH
        isd = (isd - sp.log(isd)-1).sum()
        
        print 'ISD={0}'.format(isd)
        #if (pISD-isd < 10) :
        #    print 'converged'
        #    break
        #pISD = isd
        
        H *= sp.dot( W.T, ( X * (idWH*idWH) ) ) / (SPARSENESS + sp.dot( W.T, idWH ) )
        H /= H.max()
        idWH = 1.0/sp.dot(W,H)
        W *= sp.dot(  ( X *(idWH*idWH) ) , H.T ) / sp.dot( idWH , H.T) 
        continue
        #print 'saving'
        dWH = sp.dot(W,H)
        # pp.clf()
        # pp.subplot(2,1,1)
        # pp.imshow(sp.log(1+X), aspect='auto', origin = 'lower')
        # pp.subplot(2,1,2)
        # pp.imshow(sp.log(1+dWH), aspect='auto', origin = 'lower')
        # pp.savefig('img/{0:0>2}.a.png'.format(it))
        # continue
        # pp.clf()
        # for n in xrange(nBasis):
            # pp.subplot(nBasis,2,1+2*n)
            # pp.plot(sp.log(1+ H[n,:] ))
            # pp.subplot(nBasis,2,2+2*n)
            # pp.plot( sp.log(1+W[:,n] ))
            # pp.savefig('img/{0:0>2}.b.png'.format(it))
    return W,H

def nmf_is_activation(X, W, n_iter = 20):
    '''
        Implemented by Maezawa-san
        
        基底を固定し、アクチベーションのみを推定する。
        [引数]
        X, W, nIter - nmf_isと同じ
        
        [戻り値]
        H - nmf_isと同じ
    '''
    SPARSENESS = 0.001 # IS-NMFにおける、L2正則化項
    n_basis = W.shape[1]
    assert X.shape[0] == W.shape[0]
    nF, nT = X.shape
    H = sp.ones( (n_basis,nT) )
    # dot = sp.dot
    for it in xrange(n_iter):
        idWH = 1.0/sp.dot(W,H)

        isd = X*idWH
        isd = (isd - sp.log(isd)-1).sum()
        # print isd,
        H *= sp.dot( W.T, ( X * idWH*idWH ) ) / (SPARSENESS + sp.dot( W.T, idWH ))
        H /= H.max()
    # print ''
    return H
    