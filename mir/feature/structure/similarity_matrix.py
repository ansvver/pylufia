# -*- coding: utf-8 -*-

"""
@file similarity_matrix.py
@brief similarity matrix extractor
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
from pylufia.stats.distance import *


def similarity_matrix(X, Y, metric='euclid'):
    """
    Calculate similarity matrix (forループで馬鹿正直に計算する旧Version)
    """
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    SM = sp.zeros((n_X, n_Y))
    
    if metric == 'cos':
        dist_func = cos_dist
    elif metric == 'euclid':
        dist_func = euclid_dist
    elif metric == 'mahal':
        dist_func = mahal_dist
    else:
        dist_func = cos_dist
    
    for i in xrange(n_X):
        cur_X = X[i]
        for j in xrange(i, n_Y): # 対称性を利用して計算量削減
            SM[i, j] = dist_func(cur_X, Y[j])
    
    # 対称位置の値を一気に代入(二重加算される対角成分の値は1/2する)
    # finalSM = SM + tr(SM)
    # diag(finalSM) /= 2
    tSM = SM.T
    SM = SM + tSM
    for i in xrange(SM.shape[0]):
        SM[i, i] /= 2.0
        
    return SM
    
def similarity_matrix_2d(X, Y, metric='cos'):
    """
    Calculate similarity matrix
    
    Parameters:
      X: ndarray
        input matrix 1
      Y: ndarray
        input matrix 2
      distFunc: function
        distance function
    
    Returns:
      result: ndarray
        similarity matrix
    """
    n_X = len(X)
    n_Y = len(Y)
    
    if metric == 'cos':
        dist_func = cos_dist_2d
    elif metric == 'euclid':
        dist_func = euclid_dist_2d
    elif metric == 'mahal':
        dist_func = mahal_dist_2d
    else:
        dist_func = cos_dist_2d
    
    #SM = sp.zeros((nX, nY))
    
    SM = [map(dist_func, n_X * [X[i]], Y) for i in xrange(n_X)]
    #for i in xrange(nX):
    #    SM.append(map(distFunc, nX * [X[i]], Y))
        
    SM = sp.array(SM)
        
    return SM
    
def similarity_matrix_scipy(X, Y, dist_func):
    """
    Calculate similarity matrix (scipy.spatial.distance使用)
    """
    return sp.spatial.distance.cdist(X, Y, dist_func)
    
def add_axis_to_similarity_matrix2d(feature, X, Y, SM, metric='cos'):
    """
    既にあるSimilarity Matrixに対し，一つ特徴量を追加したSimilarity Matrixを作る
    """
    n_obsX = SM.shape[0]
    n_obsY = SM.shape[1]
    n_frm = X[0].shape[0]
    n_dim = X[0].shape[1]
    SM2 = np.zeros( (SM.shape[0]+1,SM.shape[1]+1), dtype=np.double)
    
    if metric == 'cos':
        dist_func = cos_dist_2d
    elif metric == 'euclid':
        dist_func = euclid_dist_2d
    elif metric == 'mahal':
        dist_func = mahal_dist_2d
    else:
        dist_func = cos_dist_2d
        
    SM2[:n_obsX,:n_obsY] = SM
    for i in xrange(n_obsY):
        SM2[-1,i] = dist_func(feature, Y[i])
        # SM2[-1,i] = rhythmDist_euclid(feature, Y[i])
    SM2[-1,-1] = 0.0
    for j in xrange(n_obsX):
        SM2[j,-1] = rhythmDist_cos(feature, X[j])
        # SM2[j,-1] = rhythmDist_euclid(feature, X[j])
        
    return SM2
