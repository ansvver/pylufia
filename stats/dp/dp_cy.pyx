#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
_dp.pyx

DP matchingのCython実装
"""

import cython
import numpy as np
cimport numpy as np
import scipy.spatial.distance as spdist
import ymh_mir.stats.distance as distance


def dp_match_cy(seq1, seq2):
    """
    DP matching
    
    縦/横ずれペナルティあり、不一致ペナルティが均一のオーソドックスなもの。
    コストに距離関数を入れるよりも挿入誤りには頑健なので
    PDF-MusicXMLマッチングにはこちらのほうが向いていると思われる
    """
    move_penalty = 1.0
    mismatch_penalty = 3.0
    
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    
    seq1_mat = np.resize(seq1, (len(seq2),len(seq1))).T
    seq2_mat = np.resize(seq2, (len(seq1),len(seq2)))
    
    mismatch_mat = np.absolute(seq1_mat-seq2_mat)
    mismatch_idx = np.where(mismatch_mat != 0)
    mismatch_mat[mismatch_idx[0],mismatch_idx[1]] = 1
    
    cost_mat = np.zeros((len(seq1),len(seq2)))
    for i in range(1, len(seq1)):
        cost_mat[i,0] = cost_mat[i-1,0] + move_penalty + mismatch_mat[i,0] * mismatch_penalty
    for j in range(1, len(seq2)):
        cost_mat[0,j] = cost_mat[0,j-1] + move_penalty + mismatch_mat[0,j] * mismatch_penalty
    for i in range(1, len(seq1)):
        for j in range(1, len(seq2)):
            cost_s = cost_mat[i-1,j-1] + mismatch_mat[i,j] * mismatch_penalty
            cost_h = cost_mat[i,j-1] + move_penalty + mismatch_mat[i,j] * mismatch_penalty
            cost_v = cost_mat[i-1,j] + move_penalty + mismatch_mat[i,j] * mismatch_penalty
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    return cost_mat,path_mat
    
def dp_match_with_dist_cy(seq1, seq2):
    """
    コストにシーケンス要素間の距離を導入したDP
    """
    cost_mat = np.zeros((len(seq1),len(seq2)))
    for i in range(1, len(seq1)):
        cost_mat[i,0] = cost_mat[i-1,0] + np.absolute(seq1[i]-seq2[0])
    for j in range(1, len(seq2)):
        cost_mat[0,j] = cost_mat[0,j-1] + np.absolute(seq1[0]-seq2[j])
    for i in range(1, len(seq1)):
        for j in range(1, len(seq2)):
            cost_mat[i,j] = min(cost_mat[i-1,j],cost_mat[i,j-1],cost_mat[i-1,j-1]) + np.absolute(seq1[i]-seq2[j])
            
    path_mat = _trace_optimal_path(cost_mat)
    
    return cost_mat,path_mat
    
def dp_match_semitone_power_cy(feature1, feature2):
    """
    半音パワー特徴量同士のDP
    """
    cdef double move_penalty = 1.0
    cdef double mismatch_penalty = 3.0
    
    distfunc = distance.cos_dist_cy
    # distfunc = distance.euclid_dist_cy
    
    cdef int n_frames1 = feature1.shape[0]
    cdef int n_frames2 = feature2.shape[0]
    cdef np.ndarray[double,ndim=2] cost_mat = np.zeros((n_frames1,n_frames2))
    cdef int i,j
    cdef double cost_s,cost_h,cost_v
    for i in range(1, n_frames1):
        cost_mat[i,0] = cost_mat[i-1,0] + move_penalty + distfunc(feature1[i],feature2[0]) * mismatch_penalty
    for j in range(1, n_frames2):
        cost_mat[0,j] = cost_mat[0,j-1] + move_penalty + distfunc(feature1[0],feature2[j]) * mismatch_penalty
    for i in range(1, n_frames1):
        for j in range(1, n_frames2):
            cost_s = cost_mat[i-1,j-1] + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_h = cost_mat[i,j-1] + move_penalty + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_v = cost_mat[i-1,j] + move_penalty + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    confidence = 1.0 / cost_mat[-1,-1]
    
    return cost_mat,path_mat,confidence
    
cdef _trace_optimal_path(np.ndarray[double,ndim=2] cost_mat):
    """
    cost matrixから最短経路の探索
    """
    isTerminal = False
    cdef int n_row = cost_mat.shape[0]
    cdef int n_col = cost_mat.shape[1]
    cdef np.ndarray[int,ndim=2] path_mat = np.zeros((n_row,n_col), dtype=int)
    path_mat[-1,-1] = 1
    cdef int i = cost_mat.shape[0]-1
    cdef int j = cost_mat.shape[1]-1
    while isTerminal == False:
        if (i,j) > (0,0):
            idx = np.argmin(np.array([cost_mat[i-1,j],cost_mat[i,j-1],cost_mat[i-1,j-1]]))
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
        
        if (i,j) == (0,0):
            isTerminal = True
            
    return path_mat
    
cdef _comp_semitone_pow(feature1, feature2):
    """
    あるフレームの半音パワー同士の一致を比較する
    """
    pass
