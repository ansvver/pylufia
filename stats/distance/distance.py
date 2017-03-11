# -*- coding: utf-8 -*-

"""
====================================================================
distance calculation functions
====================================================================
"""

import scipy as sp
import scipy.linalg as linalg

def euclid_dist(x, y):
    """
    Compute euclid distance (1d vector)
    
    Parameters:
      x: ndarray
        input vector1
      y: ndarray
        input vector2
    
    Returns:
      result: double
        euclid distance for (x,y)
    """
    return sp.sqrt((x-y).sum())
    
def euclid_dist_2d(X, Y):
    """
    Compute euclid distance (2d vector)
    
    Parameters:
      X: ndarray
        input vector1
      Y: ndarray
        input vector2
    
    Returns:
      result: double
        euclid distance for (X,Y)
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = sp.mean([euclid_dist(x, y) for x, y in zip(RX, RY)])
    
    return d

def cos_dist(x, y):
    """
    Compute cosine distance (1d vector)
    
    Parameters:
      x: ndarray
        input vector1
      y: ndarray
        input vector2
    
    Returns:
      result: double
        cosine distance for (x,y)
    
    Notes:
      norm=0の場合の0除算対策を入れるため自前実装している
    """
    norm_x = sp.sqrt(sp.sum(x * x))
    norm_y = sp.sqrt(sp.sum(y * y))
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
        
    score = 1.0 - sp.dot(x, y) / (norm_x * norm_y)
    
    return score
    
def cos_dist_2d(X, Y):
    """
    Compute cosine distance (2d vector)
    
    Parameters:
      X: ndarray
        input vector1
      Y: ndarray
        input vector2
    
    Returns:
      result: double
        cosine distance for (X,Y)
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = sp.mean([cos_dist(x, y) for x, y in zip(RX, RY)])

    return d
    
def mahal_dist(x, y, V):
    """
    Compute mahalanobis distance
    
    Parameters:
      x: ndarray
        input matrix 1
      y: ndarray
        input matrix 2
      V: ndarray
        covariance matrix
    
    Returns:
      result: float
        mahalanobis distance
        
    Vは観測全体の共分散行列か？ -> そう。
    """
    # return sp.spatial.distance.mahalanobis(x, y, V)
    inv_V = linalg.inv(V)
    d = sp.sqrt(sp.dot(sp.dot((x-y), inv_V), (x-y).T))
    
    return d

def corrcoef(x, y):
    """
    相関係数
    """
    numer = ( (x-x.mean()) * (y-y.mean()) ).sum()
    denom = sp.sqrt( ( ( x-x.mean() )**2 ).sum() ) * sp.sqrt( ( ( y-y.mean() )**2 ).sum() ) + 1e-50
    score = numer / denom
    return score

def corrcoef_2d(X, Y):
    """
    相関係数(2次元arrayに対する)
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    d = sp.mean( [corrcoef(x, y) for x,y in zip(RX,RY)] )
    return d

def corrcoef_2d_T(X, Y):
    RX,RY = _adjust_vector_dimensions(X, Y)
    RX = RX.T
    RY = RY.T
    d = sp.mean( [corrcoef(x, y) for x,y in zip(RX,RY)] )
    return d

def weighted_corrcoef_2d(X, Y):
    RX,RY = _adjust_vector_dimensions(X, Y)
    weight = (RX+RY).sum(1)
    weight /= weight.sum()
    d = sp.mean( [corrcoef(x, y)*weight[i] for i,(x,y) in enumerate(zip(RX,RY))] )
    return d

def dtw_dist_2d(X, Y):
    """
    Dynamic Time Warping (Python version)
    
    Parameters:
      X: ndarray
        input matrix 1
      Y: ndarray
        input matrix 2
    
    Returns:
      DTW distance
    
    Notes:
      いちおうPythonで実装してみたものの非常に重く，大量のデータに適用する場合は使いものにならない．
      基本的にはdtw_cy_type*()の方を用いること．
    
    """
    D = np.zeros((X.shape[0], Y.shape[0]))
    
    dist_func = sp.spatial.distance.cosine

    D[0, 0] = dist_func(X[0, :], Y[0, :])
    for i in range(1, X.shape[0]):
        D[i, 0] = dist_func(X[i, :], Y[0, :]) + D[i-1, 0]

    for j in range(1, Y.shape[0]):
        D[0, j] = dist_func(X[0, :], Y[j, :]) + D[0, j-1]

    for i in range(1, X.shape[0]):
        for j in range(1, Y.shape[0]):
            D[i, j] = dist_func(X[i, :], Y[j, :]) + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[-1, -1]
    
def i_divergence(X, Y):
    """
    I-divergence
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = (RY*(sp.log(RY+1e-10)-sp.log(RX+1e-10)) + (RX-RY)).sum()
    
    return d
    
def i_divergence_symmetry(X, Y):
    d = (I_divergence(X, Y) + I_divergence(Y, X))/2.0
    
    return d
    
def kl_divergence(X, Y):
    d = ( Y * ( sp.log(Y) - sp.log(X) ) ).sum()
    return d

def kl2_divergence(X, Y):
    d = 0.5 * KL_divergence(X, Y) + 0.5 * KL_divergence(Y, X)
    return d

def kl_divergence_2d(X, Y):
    """
    KL-divergence
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = ( (RY*(sp.log(RY)-sp.log(RX))).sum(1) ).mean()
    
    return d
    
def kl2_divergence_2d(X, Y):
    """
    symmetric KL divergence (KL2)

    対称化したKLダイバージェンス
    """
    d = 0.5*KL_divergence(X, Y) + 0.5*KL_divergence(Y, X)
    
    return d

def js_divergence(X, Y):
    RX,RY = _adjust_vector_dimensions(X, Y)

    M = 0.5 * (RX+RY)
    d = 0.5 * ( KL_divergence(RX, M) + KL_divergence(RY, M) )
    return d
    
def is_divergence(X, Y):
    """
    板倉斎藤距離
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    d = (RY/RX - sp.log(RY/RX + 0.00001) - 1).sum()
    
    return d
    
def is_divergence_symmetry(X, Y):
    d = (IS_divergence(X, Y) + IS_divergence(Y, X))/2.0
    
    return d
    
def beta_divergence(X, Y, b):
    """
    \beta-divergence
    
    Parameters:
      X: NMFで推定したスペクトログラム
      Y: 真のスペクトログラム(=入力)
      b: beta-factor
    
    Returns:
      beta-divergenceの値
    """
    RX,RY = _adjust_vector_dimensions(X, Y)
    
    if b == 1:
        d = (RY*(sp.log(RY+0.00001)-sp.log(RX+0.00001)) + (RX-RY)).sum()
    elif b == 0:
        d = (RY/RX - sp.log(RY/RX) - 1).sum()
    else:
        d = (RY**b/(b*(b-1)) + (RX**b)/b - RY*(RX**(b-1))/(b-1)).sum()
        
    return d
    

def kl_divergence_gauss(gmm_prm1, gmm_prm2):
    """
    Calculate KL divergence (Gauss distribution vs Gauss distribution)
    
    Parameters:
      gmmPrm1: dict
        Parameters of gauss distribution 1
      gmmPrm2: dict
        Parameters of gauss distribution 2
    
    Returns:
      KL divergence
    """
    mean_vec1 = sp.array(gmm_prm1['means'])
    cov_mat1 = sp.matrix(gmm_prm1['covs'])
    mean_vec2 = sp.array(gmm_prm2['means'])
    cov_mat2 = sp.matrix(gmm_prm2['covs'])
    
    fact1 = sp.log10(sp.linalg.det(cov_mat2) / sp.linalg.det(cov_mat1))
    fact2 = sp.trace(sp.dot(cov_mat2.I, cov_mat1))
    fact3 = float(sp.dot(sp.dot((mean_vec1 - mean_vec2), cov_mat2.I), (mean_vec1 - mean_vec2).T))
    d = len(mean_vec1)
    
    score = 0.5 * (fact1 + fact2 + fact3 - d)
    
    return score

def kl2_divergence_gauss(gmm_prm1, gmm_prm2):
    """
    Calculate KL divergence (Gauss distribution vs Gauss distribution)
    
    Parameters:
      gmmPrm1: dict
        Parameters of gauss distribution 1
      gmmPrm2: dict
        Parameters of gauss distribution 2
    
    Returns:
      KL divergence
      
    Notes:
      対称性を保証するため，KL(P, Q)とKL(Q, P)の双方を求めて平均する．
    """
    dist1 = KL_divergence_gauss(gmm_prm1, gmm_prm2)
    dist2 = KL_divergence_gauss(gmm_prm2, gmm_prm1)
    
    return (dist1 + dist2) / 2.0


""" helper functions """

def _adjust_vector_dimensions(X, Y):
    """
    距離計算する2つの2次元ベクトルの要素数を合わせる
    長い方に合わせ、短い方のベクトルは0詰めする
    """
    if (X.shape[0] == Y.shape[0]):
        RX = X
        RY = Y
    elif (X.shape[0] > Y.shape[0]):
        RX = X
        RY = sp.r_[Y, sp.zeros( (X.shape[0]-Y.shape[0], Y.shape[1]) )]
    else:
        RX = sp.r_[X, sp.zeros( (Y.shape[0]-X.shape[0], X.shape[1]) )]
        RY = Y
        
    return RX,RY

def _joint_stddev(xa, xb):
    """
    Calculate joint standard deviation (for 1D array)
    
    Parameters:
      xa: ndarray
        input vector 1
      xb: ndarray
        input vector 2
        
    Returns:
      result: float
        joint standard deviation value
    """
    mean_xab = (sum(xa) + sum(xb)) / float(len(xa) + len(xb))
        
    jstd = (sum((xa - mean_xab)**2) + sum((xb - mean_xab)**2)) / float(len(xa) + len(xb) - 1)
    jstd = sp.sqrt(jstd)
        
    return jstd

def _joint_stddev_2d(Xa, Xb):
    """
    Calculate joint standard deviation (for 2D array)
    
    Parameters:
      Xa: ndarray
        input matrix 1
      Xb: ndarray
        input matrix 2
    
    Returns:
      result: float
        joint standard deviation value
    
    Notes:
      全フレームにわたって平均してしまっているが，これでよいのか？
    """
    mean_Xab = (sum(sum(Xa)) + sum(sum(Xb))) / float(sp.size(Xa) + sp.size(Xb))
        
    jstd = (sum(sum((Xa - mean_Xab)**2)) + sum(sum((Xb - mean_Xab)**2))) / float(sp.size(Xa) + sp.size(Xb) - 1)
        
    return jstd
