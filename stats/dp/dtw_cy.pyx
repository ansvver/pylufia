#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dtw_cy.pyx

Dynamic Time WarpingのCython実装
"""

import cython
import numpy as np
cimport numpy as np
import scipy.spatial.distance as spdist
import ymh_mir.stats.distance as distance


def dtw_cy(seq1, seq2):
    """
    DTW
    
    縦/横ずれペナルティあり、不一致ペナルティが均一のオーソドックスなもの。
    コストに距離関数を入れるよりも挿入誤りには頑健なので
    PDF-MusicXMLマッチングにはこちらのほうが向いていると思われる
    """
    seq1 = np.array(seq1).astype(np.double)
    seq2 = np.array(seq2).astype(np.double)
    cdef long n_frames1 = len(seq1)
    cdef long n_frames2 = len(seq2)
    
    cdef np.ndarray[double,ndim=2] seq1_mat = np.resize( seq1, ( n_frames2,n_frames1 ) ).T
    cdef np.ndarray[double,ndim=2] seq2_mat = np.resize( seq2, ( n_frames1,n_frames2 ) )
    
    cdef np.ndarray[double,ndim=2] mismatch_mat = np.absolute(seq1_mat-seq2_mat)
    mismatch_idx = np.where(mismatch_mat != 0)
    mismatch_mat[ mismatch_idx[0],mismatch_idx[1] ] = 1
    
    cdef np.ndarray[double,ndim=2] cost_mat = np.zeros( ( n_frames1,n_frames2 ), dtype=float )
    cost_mat[0,0] = mismatch_mat[0,0]
    cdef int i,j
    #for i in range( 1, n_frames1 ):
    for i from 1 <= i < n_frames1:
        cost_mat[i,0] = cost_mat[i-1,0] + mismatch_mat[i,0]
    #for j in range( 1, n_frames2 ):
    for j from 1 <= j < n_frames2:
        cost_mat[0,j] = cost_mat[0,j-1] + mismatch_mat[0,j]
    #for i in range( 1, n_frames1 ):
    for i from 1 <= i < n_frames1:
        #for j in range( 1, n_frames2 ):
        for j from 1 <= j < n_frames2:
            cost_s = cost_mat[i-1,j-1] + mismatch_mat[i,j]
            cost_h = cost_mat[i,j-1] + mismatch_mat[i,j]
            cost_v = cost_mat[i-1,j] + mismatch_mat[i,j]
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)

    n_frames_mean = (n_frames1 + n_frames2)/2
    confidence = 1.0 / (cost_mat[-1,-1]/n_frames_mean + 1e-5)
    
    return cost_mat,path_mat,confidence
    
def dtw_with_distfunc_cy(feature1, feature2, metric="cos"):
    """
    連続値特徴ベクトル同士のDP (cos類似度などの類似尺度をコストとする)
    """
    if metric == "cos":
        distfunc = distance.cos_dist_cy
    elif metric == "euclid":
        distfunc = distance.euclid_dist_cy
    else:
        metric = distance.euclid_dist_cy
    
    cdef long n_frames1 = feature1.shape[0]
    cdef long n_frames2 = feature2.shape[0]
    cdef np.ndarray[double,ndim=2] cost_mat = np.zeros((n_frames1,n_frames2))
    cdef long i,j
    cdef double cost_s,cost_h,cost_v
    cost_mat[0,0] = distfunc(feature1[0],feature2[0])
    #for i in range(1, n_frames1):
    for i from 1 <= i < n_frames1:
        cost_mat[i,0] = cost_mat[i-1,0] + distfunc(feature1[i],feature2[0])
    #for j in range(1, n_frames2):
    for j from 1 <= j < n_frames2:
        cost_mat[0,j] = cost_mat[0,j-1] + distfunc(feature1[0],feature2[j])
    #for i in range(1, n_frames1):
    for i from 1 <= i < n_frames1:
        #for j in range(1, n_frames2):
        for j from 1 <= j < n_frames2:
            cost_s = cost_mat[i-1,j-1] + distfunc(feature1[i],feature2[j])
            cost_h = cost_mat[i,j-1] + distfunc(feature1[i],feature2[j])
            cost_v = cost_mat[i-1,j] + distfunc(feature1[i],feature2[j])
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    n_frames_mean = (n_frames1 + n_frames2)/2
    confidence = 1.0 / (cost_mat[-1,-1]/n_frames_mean + 1e-5)
    
    return cost_mat,path_mat,confidence
    
cdef _trace_optimal_path(np.ndarray[double,ndim=2] cost_mat):
    """
    cost matrixから最短経路の探索
    """
    is_terminal = False
    cdef long n_row = cost_mat.shape[0]
    cdef long n_col = cost_mat.shape[1]
    cdef np.ndarray[long,ndim=2] path_mat = np.zeros( (n_row,n_col), dtype=long )
    path_mat[-1,-1] = 1
    cdef long i = cost_mat.shape[0]-1
    cdef long j = cost_mat.shape[1]-1
    while is_terminal == False:
        if i > 0 and j > 0:
            idx = np.argmin( np.array([cost_mat[i-1,j],cost_mat[i,j-1],cost_mat[i-1,j-1]]) )
            if idx == 0:
                i -= 1
            elif idx == 1:
                j -= 1
            elif idx == 2:
                i -= 1
                j -= 1
        elif i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        
        path_mat[i,j] = 1
        
        if i == 0 and j == 0:
            is_terminal = True
            
    return path_mat
    