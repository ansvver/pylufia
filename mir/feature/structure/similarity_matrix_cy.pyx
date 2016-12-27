# -*- coding: utf-8 -*-

"""
====================================================================
Cython implementation of structural feature extractors
====================================================================
"""

import cython
import numpy as np
cimport numpy as np
import numpy.linalg as linalg
from pylufia.stats.distance import *


def similarity_matrix_cy(np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=2] Y, metric='cos'):
    """
    Cython版similarity matrix
    """
    cdef np.ndarray[double, ndim=2] SM = np.zeros((X.shape[0], Y.shape[0]), dtype=np.double)
    cdef int i, j
    cdef int Xdim = X.shape[0]
    cdef int Ydim = Y.shape[0]
    
    if metric == 'cos':
        dist_func = cos_dist_cy
    elif metric == 'euclid':
        dist_func = euclid_dist_cy
    elif metric == 'kl':
        dist_func = KLDivergence_cy
    elif metric == 'kl2':
        dist_func = KL2Divergence_cy
    else:
        dist_func = cos_dist_cy
    
    for i from 0 <= i < Xdim:
        for j from i <= j < Ydim:
            SM[i, j] = dist_func(X[i], Y[j])
            SM[j, i] = SM[i, j]
            
    return SM
    
def similarity_matrix_2d_cy(X, Y, metric='cos'):
    """
    2次元特徴量のリストからsimilarity matrixを計算
    """
    cdef int n_obsX = len(X)
    cdef int n_obsY = len(Y)
    cdef int n_frm = X[0].shape[0]
    cdef int n_dim = X[0].shape[1]
    cdef np.ndarray[double, ndim=2] SM = np.zeros((n_obsX, n_obsY), dtype=np.double)
    cdef int i, j
    
    if metric == 'cos':
        dist_func = cos_dist_2d_cy
    elif metric == 'euclid':
        dist_func = euclid_dist_2d_cy
    elif metric == 'Idiv':
        dist_func = I_divergence_symmetry
    elif metric == 'ISdiv':
        dist_func = IS_divergence_symmetry
    elif metric == 'KLdiv':
        dist_func = KLDivergence_2d
    else:
        dist_func = cos_dist_2d_cy
    
    for i from 0 <= i < n_obsX:
        for j from i <= j < n_obsY:
            SM[i,j] = dist_func(X[i], Y[j])
            SM[j,i] = SM[i, j]
            
    return SM

def add_axis_to_similarity_matrix_2d_cy(feature, X, Y, SM, metric='cos'):
    """
    既にあるSimilarity Matrixに対し，一つ特徴量を追加したSimilarity Matrixを作る
    """
    cdef int n_obsX = SM.shape[0]
    cdef int n_obsY = SM.shape[1]
    cdef int n_frm = X[0].shape[0]
    cdef int n_dim = X[0].shape[1]
    cdef int i,j
    cdef np.ndarray[double, ndim=2] SM2 = np.zeros( (SM.shape[0]+1,SM.shape[1]+1), dtype=np.double)
    
    if metric == 'cos':
        dist_func = cos_dist_2d_cy
    elif metric == 'euclid':
        dist_func = euclid_dist_2d_cy
    elif metric == 'Idiv':
        dist_func = I_divergence_symmetry
    elif metric == 'ISdiv':
        dist_func = IS_divergence_symmetry
    elif metric == 'KLdiv':
        dist_func = KL_divergence_symmetry
    else:
        dist_func = cos_dist_2d_cy
        
    SM2[:n_obsX,:n_obsY] = SM
    for i from 0 <= i < n_obsY:
        SM2[-1,i] = dist_func(feature, Y[i])
    SM2[-1,-1] = 0.0
    for j from 0 <= j < n_obsX:
        SM2[j,-1] = dist_func(X[j], feature)
        
    return SM2
