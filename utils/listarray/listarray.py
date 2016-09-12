# -*- coding: utf-8 -*-

"""
listarray.py

list関連の便利関数群
"""

def makeListArray2d(n_row, n_col):
    """
    指定したサイズの2次元リストを生成する
    """
    list_array_2d = [[] for i in xrange(n_row)]
    for i,cur_list in enumerate(list_array_2d):
        list_array_2d[i] = [[] for j in xrange(n_col)]

    return list_array_2d

def makeListArray2d_copy(orig_list):
    """
    入力リストと同じ形状の2次元リストを生成する
    """
    n_row = len(orig_list)
    list2d = [[] for i in range(n_row)]
    for i,cur_list in enumerate(orig_list):
        n_col = len(cur_list)
        list2d[i] = [[] for j in range(n_col)]

    return list2d
