#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy as sp
import scipy.spatial.distance as spdist
import pylufia.stats.distance as distance


def dtw(seq1, seq2):
    """
    DTW
    
    縦/横ずれペナルティあり、不一致ペナルティが均一のオーソドックスなもの。
    コストに距離関数を入れるよりも挿入誤りには頑健なので
    PDF-MusicXMLマッチングにはこちらのほうが向いていると思われる
    """
    seq1 = sp.array(seq1)
    seq2 = sp.array(seq2)
    
    seq1_mat = sp.resize(seq1, ( len(seq2),len(seq1) ) ).T
    seq2_mat = sp.resize(seq2, ( len(seq1),len(seq2) ) )
    
    mismatch_mat = sp.absolute(seq1_mat-seq2_mat)
    mismatch_idx = sp.where(mismatch_mat != 0)
    mismatch_mat[ mismatch_idx[0],mismatch_idx[1] ] = 1
    
    cost_mat = sp.zeros( ( len(seq1),len(seq2) ) )
    cost_mat[0,0] = mismatch_mat[0,0]
    for i in range( 1, len(seq1) ):
        cost_mat[i,0] = cost_mat[i-1,0] + mismatch_mat[i,0]
    for j in range( 1, len(seq2) ):
        cost_mat[0,j] = cost_mat[0,j-1] + mismatch_mat[0,j]
    for i in range( 1, len(seq1) ):
        for j in range( 1, len(seq2) ):
            cost_s = cost_mat[i-1,j-1] + mismatch_mat[i,j]
            cost_h = cost_mat[i,j-1] + mismatch_mat[i,j]
            cost_v = cost_mat[i-1,j] + mismatch_mat[i,j]
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)

    confidence = 1.0 / (cost_mat[-1,-1] + 1e-5)
    
    return cost_mat,path_mat,confidence
    
def dtw_with_distfunc(feature1, feature2, metric="cos"):
    """
    連続値特徴ベクトル同士のDP (cos類似度などの類似尺度をコストとする)
    """
    if metric == "cos":
        distfunc = distance.cos_dist
    elif metric == "euclid":
        distfunc = distance.euclid_dist
    else:
        metric = distance.euclid_dist
    
    n_frames1 = feature1.shape[0]
    n_frames2 = feature2.shape[0]
    cost_mat = np.zeros((n_frames1,n_frames2))
    cost_mat[0,0] = distfunc(feature1[0],feature2[0])
    for i in range(1, n_frames1):
        cost_mat[i,0] = cost_mat[i-1,0] + distfunc(feature1[i],feature2[0])
    for j in range(1, n_frames2):
        cost_mat[0,j] = cost_mat[0,j-1] + distfunc(feature1[0],feature2[j])
    for i in range(1, n_frames1):
        for j in range(1, n_frames2):
            cost_s = cost_mat[i-1,j-1] + distfunc(feature1[i],feature2[j])
            cost_h = cost_mat[i,j-1] + distfunc(feature1[i],feature2[j])
            cost_v = cost_mat[i-1,j] + distfunc(feature1[i],feature2[j])
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    confidence = 1.0 / (cost_mat[-1,-1] + 1e-5)
    
    return cost_mat,path_mat,confidence
    
def _trace_optimal_path(cost_mat):
    """
    cost matrixから最短経路の探索
    """
    is_terminal = False
    n_row = cost_mat.shape[0]
    n_col = cost_mat.shape[1]
    path_mat = np.zeros( (n_row,n_col), dtype=int )
    path_mat[-1,-1] = 1
    i = cost_mat.shape[0]-1
    j = cost_mat.shape[1]-1
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
