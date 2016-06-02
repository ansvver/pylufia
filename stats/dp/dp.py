# -*- coding: utf-8 -*-

"""
@file dp.py
@brief DP matching functions
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.spatial.distance as spdist
import pylufia.stats.distance as distance


def dp_match(seq1, seq2):
    """
    DP matching
    
    縦/横ずれペナルティあり、不一致ペナルティが均一のオーソドックスなもの。
    コストに距離関数を入れるよりも挿入誤りには頑健なので
    PDF-MusicXMLマッチングにはこちらのほうが向いていると思われる
    """
    move_penalty = 1.0
    mismatch_penalty = 3.0
    
    seq1 = sp.array(seq1)
    seq2 = sp.array(seq2)
    
    seq1_mat = sp.resize(seq1, (len(seq2),len(seq1))).T
    seq2_mat = sp.resize(seq2, (len(seq1),len(seq2)))
    
    mismatch_mat = sp.absolute(seq1_mat-seq2_mat)
    mismatch_idx = sp.where(mismatch_mat != 0)
    mismatch_mat[mismatch_idx[0],mismatch_idx[1]] = 1
    
    cost_mat = sp.zeros((len(seq1),len(seq2)))
    for i in xrange(1, len(seq1)):
        cost_mat[i,0] = cost_mat[i-1,0] + move_penalty + mismatch_mat[i,0] * mismatch_penalty
    for j in xrange(1, len(seq2)):
        cost_mat[0,j] = cost_mat[0,j-1] + move_penalty + mismatch_mat[0,j] * mismatch_penalty
    for i in xrange(1, len(seq1)):
        for j in xrange(1, len(seq2)):
            cost_s = cost_mat[i-1,j-1] + mismatch_mat[i,j] * mismatch_penalty
            cost_h = cost_mat[i,j-1] + move_penalty + mismatch_mat[i,j] * mismatch_penalty
            cost_v = cost_mat[i-1,j] + move_penalty + mismatch_mat[i,j] * mismatch_penalty
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    return cost_mat,path_mat
    
def dp_match_with_dist(seq1, seq2):
    """
    コストにシーケンス要素間の距離を導入したDP
    """
    cost_mat = sp.zeros((len(seq1),len(seq2)))
    for i in xrange(1, len(seq1)):
        cost_mat[i,0] = cost_mat[i-1,0] + sp.absolute(seq1[i]-seq2[0])
    for j in xrange(1, len(seq2)):
        cost_mat[0,j] = cost_mat[0,j-1] + sp.absolute(seq1[0]-seq2[j])
    for i in xrange(1, len(seq1)):
        for j in xrange(1, len(seq2)):
            cost_mat[i,j] = min(cost_mat[i-1,j],cost_mat[i,j-1],cost_mat[i-1,j-1]) + sp.absolute(seq1[i]-seq2[j])
            
    path_mat = _trace_optimal_path(cost_mat)
    
    return cost_mat,path_mat
    
def dp_match_semitone_power(feature1, feature2):
    """
    半音パワー特徴量同士のDP
    """
    move_penalty = 1.0
    mismatch_penalty = 3.0
    
    distfunc = distance.cos_dist_cy
    # distfunc = distance.euclid_dist_cy
    
    n_frames1 = feature1.shape[0]
    n_frames2 = feature2.shape[0]
    cost_mat = sp.zeros((n_frames1,n_frames2))
    for i in xrange(1, n_frames1):
        cost_mat[i,0] = cost_mat[i-1,0] + move_penalty + distfunc(feature1[i],feature2[0]) * mismatch_penalty
    for j in xrange(1, n_frames2):
        cost_mat[0,j] = cost_mat[0,j-1] + move_penalty + distfunc(feature1[0],feature2[j]) * mismatch_penalty
    for i in xrange(1, n_frames1):
        for j in xrange(1, n_frames2):
            cost_s = cost_mat[i-1,j-1] + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_h = cost_mat[i,j-1] + move_penalty + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_v = cost_mat[i-1,j] + move_penalty + distfunc(feature1[i],feature2[j]) * mismatch_penalty
            cost_mat[i,j] = min(cost_s,cost_h,cost_v)
            
    path_mat = _trace_optimal_path(cost_mat)
    
    confidence = 1.0 / cost_mat[-1,-1]
    
    return cost_mat,path_mat,confidence
    
def _trace_optimal_path(cost_mat):
    """
    cost matrixから最短経路の探索
    """
    isTerminal = False
    path_mat = sp.zeros(cost_mat.shape)
    path_mat[-1,-1] = 1
    i = cost_mat.shape[0]-1
    j = cost_mat.shape[1]-1
    while isTerminal == False:
        if (i,j) > (0,0):
            idx = sp.argmin(sp.array([cost_mat[i-1,j],cost_mat[i,j-1],cost_mat[i-1,j-1]]))
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
    
def _comp_semitone_pow(feature1, feature2):
    """
    あるフレームの半音パワー同士の一致を比較する
    """
    pass

    
# def main():
#     import matplotlib.pyplot as pp
#     x = [1,2,3,4,5]
#     y = [1,1,2,1,3,3,4,5]
    
#     cost_mat,path_mat = dp_match(x,y)
    
#     idx = sp.where(path_mat == 1)
#     path_coord_x = [_x for _x in idx[1]]
#     path_coord_y = [_y for _y in idx[0]]
    
#     print cost_mat
#     print path_mat
    
#     print path_coord_x
#     print path_coord_y
    
#     pp.imshow(cost_mat, aspect='auto', origin='lower', cmap='gist_yarg')
#     pp.xticks(sp.arange(cost_mat.shape[1]), y)
#     pp.yticks(sp.arange(cost_mat.shape[0]), x)
#     pp.axis([0,cost_mat.shape[1]-1,0,cost_mat.shape[0]-1])
#     pp.colorbar()
#     pp.grid(True)
#     pp.hold(True)
#     pp.plot(path_coord_x, path_coord_y, 'r')
#     pp.show()
    
# if __name__ == '__main__':
#     main()
