# -*- coding: utf-8 -*-

"""
@file mds.py
@brief MDS(MultiDimensional Scaling)
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.linalg as linalg


def mds(SM):
    """
    MDS (Multi Dimensional Scaling)
    
    Parameters:
      SM: ndarray
        Input similarity matrix
    
    Returns:
      V1: ndarray
        Dimension 1 of MDS
      V2: ndarray
        Dimension 2 of MDS
    """
    N = SM.shape[0]
    
    # 距離の2乗行列を作成
    D = SM * SM
    print "D={}".format(D)

    # 中心化行列
    one = sp.eye(N) - sp.ones((N, N)) / N
    print "one={}".format(one)

    # ヤング・ハウスホルダー変換
    P = -0.5 * one * D * one # これだと要素積になってしまうのでは？
    # P = -0.5 * sp.dot( sp.dot(one, D), one )
    print "P={}".format(P)

    # 固有値分解
    W, V = sp.linalg.eig(P)
    ind = sp.argsort(W)
    print "W={}".format(W)
    print "V={}".format(V)
    print "ind={}".format(ind)
    x1 = ind[-1]
    x2 = ind[-2]
    
    # 標準偏差を掛けたデータにする
    # s = P.std(axis=0)
    # w1 = s[x1]
    # w2 = s[x2]
    
    # V1 = w1 * V[:, x1]
    # V2 = w2 * V[:, x2]
    V1 = V[:,x1]
    V2 = V[:,x2]

    # 実数値に変換
    V1 = V1.astype('float')
    V2 = V2.astype('float')

    return V1, V2
