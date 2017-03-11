# -*- coding: utf-8 -*-

"""
====================================================================
Cython implementation of distance.py
====================================================================
"""

import cython
import numpy as np
cimport numpy as np
import numpy.linalg as linalg
#from libcpp.vector cimport vector
#cdef extern from <vector> namespace std


def euclid_dist_cy(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y):
    """
    Calculate euclid distance
    
    Parameters:
      x: ndarray
        input vector1
      y: ndarray
        input vector2
        
    Returns:
      result: double
        euclid distance for (x,y)
    """
    cdef double sumval = 0.0
    cdef int i
    cdef int N = len(x)
    
    for i from 0 <= i < N:
        sumval += (x[i] - y[i]) * (x[i] - y[i]) # 2乗
    
    sumval **= 0.5
    
    return sumval
    
def euclid_dist_2d_cy(np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=2] Y):
    """
    euclid距離の計算(2次元ベクトル同士)
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = np.mean([euclid_dist_cy(x,y) for x, y in zip(RX,RY)])
    
    return d
    
def cos_dist_cy(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y):
    """
    Calculate cosine distance
    
    Parameters:
      x: ndarray
        input vector1
      y: ndarray
        input vector2
        
    Returns:
      result: double
        cosine distance for (x,y)
    """
    cdef double sumval = 0.0
    cdef double norm_x = 0.0
    cdef double norm_y = 0.0
    cdef int i
    cdef int N = len(x)
    cdef double d = 0.0
    
    for i from 0 <= i < N:
        norm_x += x[i] * x[i]
        norm_y += y[i] * y[i]
    norm_x **= 0.5
    norm_y **= 0.5
    
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
    
    for i from 0 <= i < N:
        sumval += x[i] * y[i]
        
    d = 1.0 - sumval / (norm_x * norm_y)
    
    return d

def inv_cos_dist_cy(np.ndarray[double,ndim=1] x, np.ndarray[double,ndim=1] y):
    xinv = 1.0 / (x+1e-50)
    yinv = 1.0 / (y+1e-50)
    return 0.5 * ( cos_dist_cy(x,y) + cos_dist_cy(xinv,yinv) )
    
def cos_dist_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    """
    cos距離の計算(2次元ベクトル同士)
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = np.mean([cos_dist_cy(x, y) for x,y in zip(RX,RY)])
    # d = np.amax([cos_dist_cy(x, y) for x,y in zip(RX,RY)])
    # d = np.median([cos_dist_cy(x, y) for x,y in zip(RX,RY)])
    
    return d

def cos_dist_2d_T_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    RX,RY = _adjust_vector_dimensions(X, Y)
    RX = RX.T
    RY = RY.T
    d = np.mean( [cos_dist_cy(x,y) for x,y in zip(RX,RY)] )
    return d

def inv_cos_dist_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    RX,RY = _adjust_vector_dimensions(X, Y)

    RXinv = 1.0 / (RX+1e-50)
    RYinv = 1.0 / (RY+1e-50)

    d = np.mean( [cos_dist_cy(x,y) + cos_dist_cy(xi, yi) for x,y,xi,yi in zip(RX,RY,RXinv,RYinv)] )

    return d

def inv_cos_dist_2d_T_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    RX,RY = _adjust_vector_dimensions(X, Y)
    RX = RX.T
    RY = RY.T
    d = np.mean( [inv_cos_dist_cy(x,y) for x,y in zip(RX,RY)] )
    return d

def weighted_cos_dist_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    RX,RY = _adjust_vector_dimensions(X, Y)
    weight = (RX+RY).sum(1)
    weight /= weight.sum()
    d = np.mean([cos_dist_cy(x, y)*weight[i] for i,(x,y) in enumerate(zip(RX,RY))])

    return d

def mahal_dist_cy(np.ndarray[double,ndim=1] x, np.ndarray[double,ndim=1] y, np.ndarray[double,ndim=2] V):
    """
    Mahalanobis distance
    """
    cdef int i
    cdef int N = len(x)
    
    Vi = linalg.inv(V)
    
    d = np.sqrt(np.dot(np.dot((x-y).T, Vi), (x-y)))
    
    return d
    
def mahal_dist_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y, np.ndarray[double,ndim=2] V):
    pass

def corrcoef_cy(np.ndarray[double,ndim=1] x, np.ndarray[double,ndim=1] y):
    """
    相関係数
    """
    cdef int i
    cdef int N = len(x)
    cdef double mean_x = 0.0
    cdef double mean_y = 0.0
    for i from 0 <= i < N:
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= N
    mean_y /= N

    cdef double numer = 0.0
    for i from 0 <= i < N:
        numer += (x[i] - mean_x) * (y[i] - mean_y)

    cdef double norm_x = 0.0
    cdef double norm_y = 0.0
    for i from 0 <= i < N:
        norm_x += (x[i] - mean_x)**2
        norm_y += (y[i] - mean_y)**2
    norm_x = np.sqrt(norm_x)
    norm_y = np.sqrt(norm_y)

    score = numer / (norm_x * norm_y + 1e-10)

    return score

def corrcoef_inv_cy(np.ndarray[double,ndim=1] x, np.ndarray[double,ndim=1] y):
    return 1.0 / corrcoef_cy(x, y)

def inv_corrcoef_cy(np.ndarray[double,ndim=1] x, np.ndarray[double,ndim=1] y):
    xinv = 1.0 / (x+1e-50)
    yinv = 1.0 / (y+1e-50)
    return 0.5 * ( corrcoef_cy(x,y) + corrcoef_cy(xinv,yinv) )

def corrcoef_2d_rowwise_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    """
    相関係数(2次元arrayに対する)
    """
    #RX,RY = _adjust_vector_dimensions(X, Y)
    d = np.sum( [corrcoef_cy(x, y) for x,y in zip(X,Y)] ) / X.shape[0]
    return d

def corrcoef_2d_colwise_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    #RX = RX.T
    #RY = RY.T
    XT = X.T
    YT = Y.T
    d = np.sum( [corrcoef_cy(x, y) for x,y in zip(XT,YT)] ) / XT.shape[0]
    return d

def inv_corrcoef_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    d = np.mean( [inv_corrcoef_cy(x,y) for x,y in zip(X,Y)] )
    return d

def inv_corrcoef_2d_T_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    #RX = RX.T
    #RY = RY.T
    XT = X.T
    YT = Y.T
    d = np.mean( [inv_corrcoef_cy(x,y) for x,y in zip(XT,YT)] )
    return d

def weighted_corrcoef_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y, np.ndarray[double,ndim=1] weight):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    #if len(weight) < RX.shape[0]:
    #    weight = np.r_[weight, np.zeros(RX.shape[0]-len(weight))]
    X = weight[:,np.newaxis] * X
    Y = weight[:,np.newaxis] * Y
    d = np.mean( [corrcoef_cy(x, y) for x,y in zip(X,Y)] )
    return d

def weighted_corrcoef_2d_T_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y, np.ndarray[double,ndim=1] weight):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    X = X.T * weight[:,np.newaxis]
    Y = Y.T * weight[:,np.newaxis]
    d = np.mean( [corrcoef_cy(x,y) for x,y in zip(X,Y)] )
    return d

def ccf_2d_cy(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    #RX,RY = _adjust_vector_dimensions(X, Y)
    XT = X.T
    YT = Y.T
    d = np.sum( [np.correlate(x,y,mode='valid') for x,y in zip(XT,YT)] ) / (XT.shape[0]*XT.shape[1])
    return d

def dtw_dist_type1_cy(np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=2] Y, metric='cos'):
    """
    Dynamic Time Warping (Type I)
    
    Parameters:
      X: ndarray
        input matrix 1
      Y: ndarray
        input matrix 2
        
    Returns:
      result: double
        DTW distance for (X,Y)
    """
    cdef int i, j, k
    cdef int n_rowX = np.size(X, 0)
    cdef int n_colX = np.size(X, 1)
    cdef int n_rowY = np.size(Y, 0)
    cdef int n_colY = np.size(Y, 1)

    cdef np.ndarray[double, ndim=2] D = np.zeros((n_rowX, n_rowY), dtype=np.double)
    cdef np.ndarray[double, ndim=1] aryX = np.zeros(n_colX, dtype=np.double)
    cdef np.ndarray[double, ndim=1] aryY = np.zeros(n_colY, dtype=np.double)
    
    if metric == 'cos':
        dist_func = cos_dist_cy
    elif metric == 'corrcoef':
        dist_func = corrcoef_inv_cy
    elif metric == 'euclid':
        dist_func = euclid_dist_cy
    elif metric == 'mahal':
        dist_func = mahal_dist_cy
    else:
        dist_func = cos_dist_cy

    ## D[0, 0]
    for k from 0 <= k < n_colX: # X[0, :]
        aryX[k] = X[0, k]
        
    for k from 0 <= k < n_colY: # Y[0, :]
        aryY[k] = Y[0, k]
        
    D[0, 0] = dist_func(aryX, aryY)
    
    ## D[i, 0]
    #aryY = Y[0, :]
    for i from 1 <= i < n_rowX:
        #aryX = X[i, :]
        for k from 0 <= k < n_colX:
            aryX[k] = X[i, k]
        D[i, 0] = dist_func(aryX, aryY) + D[i-1, 0]

    ## D[0, i]
    #aryX = X[0, :]
    for k from 0 <= k < n_colX:
        aryX[k] = X[0, k]
    
    for i from 1 <= i < n_rowY:
        #aryY = Y[i, :]
        for k from 0 <= k < n_colY:
            aryY[k] = Y[i, k]
        D[0, i] = dist_func(aryX, aryY) + D[0, i-1]

    ## D[i, j]
    for i from 1 <= i < n_rowX:
        #aryX = X[i, :]
        for k from 0 <= k < n_colX:
            aryX[k] = X[i, k]

        for j from 1 <= j < n_rowY:
            #aryY = Y[j, :]
            for k from 0 <= k < n_colY:
                aryY[k] = Y[j, k]
            D[i, j] = dist_func(aryX, aryY) + _min3(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[-1, -1] / np.sqrt(D.shape[0]**2 + D.shape[1]**2)

def dtw_dist_type3_cy(np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=2] Y, metric='cos'):
    """
    Dynamic Time Warping (Type III)
    
    Parameters:
      X: ndarray
        input matrix 1
      Y: ndarray
        input matrix 2
        
    Returns:
      result: double
        DTW distance for (X,Y)
    """
    cdef int i, j, k
    cdef int n_rowX = np.size(X, 0)
    cdef int n_colX = np.size(X, 1)
    cdef int n_rowY = np.size(Y, 0)
    cdef int n_colY = np.size(Y, 1)

    cdef np.ndarray[double, ndim=2] D = np.zeros((n_rowX, n_rowY), dtype=np.double)
    cdef np.ndarray[double, ndim=1] aryX = np.zeros(n_colX, dtype=np.double)
    cdef np.ndarray[double, ndim=1] aryY = np.zeros(n_colY, dtype=np.double)
    
    if metric == 'cos':
        dist_func = cos_dist_cy
    elif metric == 'euclid':
        dist_func = euclid_dist_cy
    elif metric == 'mahal':
        dist_func = mahal_dist_cy
    else:
        dist_func = cos_dist_cy

    ## D[0, 0]
    for k from 0 <= k < n_colX: # X[0, :]
        aryX[k] = X[0, k]
        
    for k from 0 <= k < n_colY: # Y[0, :]
        aryY[k] = Y[0, k]
    
    D[0, 0] = dist_func(aryX, aryY)
    
    ## D[:, 0]
    #aryY = Y[0, :]
    for i from 1 <= i < n_rowX:
        #aryX = X[i, :]
        for k from 0 <= k < n_colX:
            aryX[k] = X[i, k]
        D[i, 0] = dist_func(aryX, aryY)

    ## D[0, :]
    #aryX = X[0, :]
    for k from 0 <= k < n_colX:
        aryX[k] = X[0, k]
    
    for i from 1 <= i < n_rowY:
        #aryY = Y[i, :]
        for k from 0 <= k < n_colY:
            aryY[k] = Y[i, k]
        D[0, i] = dist_func(aryX, aryY)
        
    ## D[:, 1]
    for k from 0 <= k < n_colY:
        aryY[k] = Y[1, k]
        
    for i from 1 <= i < n_rowX:
        #aryX = X[i, :]
        for k from 0 <= k < n_colX:
            aryX[k] = X[i, k]
        D[i, 1] = dist_func(aryX, aryY) + D[i-1, 0]
    
    ## D[1, :]
    for k from 0 <= k < n_colX:
        aryX[k] = X[1, k]
        
    for i from 1 <= i < n_rowY:
        #aryY = Y[i, :]
        for k from 0 <= k < n_colY:
            aryY[k] = Y[i, k]
        D[1, i] = dist_func(aryX, aryY) + D[0, i-1]
        
    ## D[i, j]
    for i from 2 <= i < n_rowX:
        #aryX = X[i, :]
        for k from 0 <= k < n_colX:
            aryX[k] = X[i, k]

        for j from 2 <= j < n_rowY:
            #aryY = Y[j, :]
            for k from 0 <= k < n_colY:
                aryY[k] = Y[j, k]
            D[i, j] = dist_func(aryX, aryY) + _min3(D[i-1, j-1], D[i-1, j-2], D[i-2, j-1])

    return D[-1, -1]

def KL_divergence_cy(X, Y):
    d =(Y*(np.log(Y+1e-10)-np.log(X+1e-10))).sum()
    return d

def KL2_divergence_cy(X, Y):
    d = 0.5 * KL_divergence_cy(X, Y) + 0.5 * KL_divergence_cy(Y, X)
    return d

def KL_divergence_2d_cy(X, Y):
    """
    KL-divergence
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = ( (RY*(np.log(RY+1e-10)-np.log(RX+1e-10))).sum(1) ).mean()
    
    return d
    
def KL2_divergence_2d_cy(X, Y):
    """
    symmetric KL divergence (KL2)

    対称化したKLダイバージェンス
    """
    d = 0.5*KL_divergence_2d_cy(X, Y) + 0.5*KL_divergence_2d_cy(Y, X)
    
    return d
    
""" helper functions """

cdef _adjust_vector_dimensions(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Y):
    """
    距離計算する2つの2次元ベクトルの要素数を合わせる
    長い方に合わせ、短い方のベクトルは0詰めする
    """
    if (X.shape[0] == Y.shape[0]):
        RX = X
        RY = Y
    elif (X.shape[0] > Y.shape[0]):
        RX = X
        RY = np.r_[Y, np.zeros( (X.shape[0]-Y.shape[0], Y.shape[1]) )]
    else:
        RX = np.r_[X, np.zeros( (Y.shape[0]-X.shape[0], X.shape[1]) )]
        RY = Y
        
    return RX,RY

cdef double _min(double x, double y):
    """
    Min functions for two variable
    
    Parameters:
      x: double
        variable 1
      y: double
        variable 2
        
    Returns:
      result: minimum value
    """
    if x <= y:
        return x
    else:
        return y

cdef double _min3(double x, double y, double z):
    """
    Min functions for three variable
    
    Parameters:
      x: double
        variable 1
      y: double
        variable 2
      z: double
        variable 3
        
    Returns:
      result: double
        minimum value
    """
    return _min(x, _min(y, z))

# DTW transition cost
cdef double _dtw_T(double x, double y):
    """
    Calculate DTW transition cost
    
    Parameters:
      x: double
      y: double
      
    Returns:
      result: float
        sqrt(x**2+y**2)
    """
    return (x*x + y*y) ** 0.5
