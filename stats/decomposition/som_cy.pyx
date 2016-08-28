# -*- coding: utf-8 -*-

"""
_som.pyx

som.py のCython実装．
"""

import cython
import numpy as np
cimport numpy as np

def som_cy( np.ndarray[double, ndim=2] data, int nx, int ny, plotLabels = None ):
    '''
        Implemented by Maezawa-san
        
        自己組織化マップで、高次元データを２次元に圧縮する
        ２次元のデータとは、XとY軸から構成されるデータであり、
        XとYはグリッド上で表現される。
        
        [引数]
         data - N x D 次元のデータ。Nはデータの総数、Dはデータの次元
         nx - X軸の分割数
         ny - Y軸の分割数
         plotLabels - None以外の場合、データをプロットする、この場合、plotLabelsはN次元の文字列配列である必要がある。
        [戻り値]
         X, Y, IDX のTuple
         X - X軸のリスト
         Y - Y軸のリスト
         IDX - N次元の配列。
        　　　　　　　各入力データ点が、X軸とY軸のどこに対応するかが保存されている。 
        　　　　　　　data[n,:]に対応するデータは X[ IDX[n] ], Y[IDX[n]] に射影されている。
        
    '''
    _X,_Y = np.meshgrid( np.arange(nx), np.arange(ny) )
    cdef np.ndarray[int, ndim=1] X = np.zeros(_X.shape[0]*_X.shape[1], dtype=np.int)
    cdef np.ndarray[int, ndim=1] Y = np.zeros(_Y.shape[0]*_Y.shape[1], dtype=np.int)
    X = _X.flatten()
    Y = _Y.flatten()
    cdef int nData = np.size(data, 0)
    cdef int nDim = np.size(data, 1)
    cdef int nX = len(X)
    cdef int nY = len(Y)
    cdef np.ndarray[int, ndim=1] IDX = np.zeros( nData, dtype=np.int )
    cdef np.ndarray[double, ndim=2] weightVec = np.zeros( (nX,nDim), dtype=np.double )
    cdef np.ndarray[double, ndim=2] diff = np.zeros( (nX,nDim), dtype=np.double )
    # cdef np.ndarray[double, ndim=1] distance = np.zeros(nX, dtype=np.double)
    cdef int bmu = 0
    cdef np.ndarray[double, ndim=1] T = np.zeros( nX, dtype=np.double )
    cdef int i
    
    for i from 0 <= i < nX:
        weightVec[i,:] = data[ i%nData, : ]
    weightVec += np.random.randn(nX, nDim) * data.std(0)[np.newaxis,:] * 0.01 #+ data.mean(0)[sp.newaxis,:]
    
    # compute the distace
    cdef double alpha = 1.0 #0.99999
    cdef double r = float( nx*nx+ny*ny  )
    # iPrev=sp.zeros_like(IDX)
    cdef int it = 0
    for it from 0 <= it < 100:
        print(it)
        for i from 0 <= i < nData:
            diff = data[i,:] - weightVec[:,:]
            # distance = (diff*diff).sum(1)
            # bmu = distance.argmin()
            bmu = (diff*diff).sum(1).argmin()
            IDX[i] = bmu
            T = np.exp( - 0.5 * ( (X-X[bmu])**2 + (Y-Y[bmu])**2 ) / r )
            
            #print T
            weightVec += T[:,np.newaxis]*alpha*diff[np.newaxis,bmu,:] #+ sp.randn( weightVec.shape[0], weightVec.shape[1] )*0.1
            # print weightVec
        
        #exit()
        #if ( (iPrev==IDX).all() ):break
        #iPrev[:] = IDX[:]
        alpha *= 0.98
        r *= 0.91
        
    for i from 0 <= i < nData:
        d =  ((weightVec-data[i,:])**2).sum(1) ** 0.5
        IDX[i] = d.argmin()
        # print IDX[i]
    # print IDX
    
    return X,Y,IDX

""" helper functions """

cdef int _myargmin(np.ndarray[double, ndim=1] X):
    cdef int i
    cdef int minIdx = 0
    cdef double minVal = X[0]
    cdef int nX = len(X)
    
    for i from 1 <= i < nX:
        if X[i] < minVal:
            minVal = X[i]
            minIdx = i
            
    return minIdx
    
cdef np.ndarray[double,ndim=1] _mysum_1(np.ndarray[double, ndim=2] X):
    cdef int r
    cdef int c
    cdef int nRow = np.size(X, 0)
    cdef int nCol = np.size(X, 1)
    cdef np.ndarray[double,ndim=1] sum_array = np.zeros( nRow, dtype=np.double)
    
    for r from 0 <= r < nRow:
        for c from 0 <= c < nCol:
            sum_array[r] += X[r,c]
            
    return sum_array
