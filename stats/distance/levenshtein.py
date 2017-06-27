# -*- coding: utf-8 -*-

"""
@package
@brief
@author Dan SASAI (YCJ,RDD)
"""

import scipy as sp


def levenshtein(seq1, seq2):
    """ levenshtein distance
    @param seq1 a list of sequence 1
    @param seq2 a list of sequence 2
    @return distance value
    """
    n_seq1 = len(seq1)
    n_seq2 = len(seq2)
    cost_mat = sp.zeros( (n_seq1+1, n_seq2+1) )
    for i in range(1, n_seq1+1):
        cost_mat[i,0] = i
    for j in range(1, n_seq2+1):
        cost_mat[0,j] = j

    for j in range(1, n_seq2+1):
        for i in range(1, n_seq1+1):
            if seq1[i-1] == seq2[j-1]:
                subst_cost = 0
            else:
                subst_cost = 1
            cost_mat[i,j] = min(cost_mat[i-1,j] + 1, cost_mat[i,j-1] + 1, cost_mat[i-1,j-1] + subst_cost)

    return cost_mat[n_seq1,n_seq2]
