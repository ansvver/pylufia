# -*- coding: utf-8 -*-

"""
@file kmeans_cy.pyx
@brief kmeans clustering (cython version)
@author ふぇいと (@stfate)

@description

"""

import cython
import numpy as np
cimport numpy as np
import plifia.stats.distance as distance


def kmeans_cy(features, n_clusters=2, max_iter=100, init="pp"):
    """
    k-means clustering
    
    Parameters:
      features: ndarray
        set of features [obs * dims]
      K: int
        number of centroids
      maxIter: int
        max times of iteration
      nRep: int
        max times of replicates

    Returns:
      label: ndarray
        cluster label of each observation
      center: ndarray
        centroids

    Notes:
      自前実装してみたがscipy.cluster.vqにkmeans2があるのでそちらを使ったほうが良い．
      -> kmeans2には線形代数演算上のバグがある模様．
    
    """
    cdef int n_obs = features.shape[0]
    cdef int n_dim = features.shape[1]
    cdef np.ndarray[int,ndim=1] labels = np.zeros(n_obs, dtype=np.int32)
    cdef np.ndarray[int,ndim=1] old_labels = np.zeros(n_obs, dtype=np.int32)
    cdef np.ndarray[double,ndim=2] centers = np.zeros( (n_clusters,n_dim), dtype=np.double )
    cdef int k,it
    cdef double error=0.0,E=0.0

    # 初期クラスタの割り当て
    labels,centers = _init(features, n_clusters, method=init)
    
    old_labels = np.zeros(n_obs, dtype=np.int32)
    
    # k-meansクラスタリングiteration
    for it in xrange(max_iter):
        print "iterates: {}".format(it)

        # 直前のクラスタ分割を保存
        old_labels = labels.copy()
        
        # E-step: 各観測をセントロイドとの距離が最小になるクラスタへ再配置
        labels,error = _update_cluster_label(features, centers)
        E += error
            
        # M-step: セントロイドの更新
        centers = _update_centroid(features, labels, n_clusters)
        
        if np.absolute(labels-old_labels).sum() == 0:
            break
    
    return labels, centers
    

""" helper functions """

cdef double _sum(np.ndarray[double, ndim=1] vec):
    cdef int i
    cdef int N = np.size(vec)
    cdef double sumval = 0.0
    for i from 0 <= i < N:
        sumval += vec[i]
    return sumval
    
cdef int _argmin(np.ndarray[double, ndim=1] vec):
    cdef int min_idx = 0
    cdef double min_val = vec[0]
    cdef int i
    cdef int N = np.size(vec)
    for i from 1 <= i < N:
        if vec[i] < min_val:
            min_val = vec[i]
            min_idx = i
    return min_idx

cdef _init(features, n_clusters, method="random"):
    n_obs = features.shape[0]
    n_dim = features.shape[1]
    labels = np.zeros(n_obs, dtype=np.int32)
    centers = np.zeros( (n_clusters,n_dim), dtype=np.double )

    # random
    if method == "random":
        # 初期クラスタの割り当て
        labels = (np.random.rand(n_obs) * n_clusters).astype(np.int32)
        
        # 初期セントロイド 
        for k in xrange(n_clusters):
            subset = features[np.where(labels == k)]
            centers[k] = np.mean(subset, axis=0)
    # kmeans++
    elif method == "pp":
        centers[0] = features[np.random.randint(n_obs)]
        n_clusters_choosed = 1
        D = np.zeros(n_obs, dtype=float)
        while (n_clusters_choosed < n_clusters):
            for obs in xrange(n_obs):
                cur_center,cur_dist = _find_nearest_center(features[obs], centers[:n_clusters_choosed])
                D[obs] = cur_dist
            P = D**2 / (D**2).sum()
            next_center_idx = np.random.choice(n_obs, 1, p=P)
            centers[n_clusters_choosed,:] = features[next_center_idx,:]
            n_clusters_choosed += 1

    return labels,centers

cdef _find_nearest_center(feature, centers):
    n_clusters = centers.shape[0]
    D = np.zeros(n_clusters, dtype=float)
    for i,center in enumerate(centers):
        D[i] = distance.euclid_dist_cy(feature, center)
    minidx = np.argmin(D)
    return centers[minidx],D[minidx]

cdef _update_cluster_label(np.ndarray[double, ndim=2] features, np.ndarray[double, ndim=2] centers):
    """
    現在のcentroidからクラスタ割当ラベルlabelを更新 (E-step)
    """
    cdef int i
    cdef int k
    cdef int n_obs = np.size(features, 0)
    cdef int n_dim = np.size(features, 1)
    cdef int n_clusters = np.size(centers, 0)
    cdef np.ndarray[double, ndim=1] D = np.zeros( n_clusters, dtype=np.double )
    cdef np.ndarray[double, ndim=1] D2 = np.zeros( n_clusters, dtype=np.double )
    cdef double error = 0.0
    cdef int min_idx = 0
    cdef np.ndarray[int, ndim=1] label = np.zeros(n_obs, dtype=np.int32)
    
    for i from 0 <= i < n_obs:
        for k from 0 <= k < n_clusters:
            # D[k] = np.sqrt(np.sum((features[i]-centers[k])**2))
            D[k] = _sum( (features[i]-centers[k])**2 )**(0.5)
            D[k] = D[k]**2
            #D[k] = distance.KL2Divergence_cy(features[i], centers[k])
            
        error += _sum(D)
        min_idx = _argmin(D)
        label[i] = min_idx
        
    return label, error
    
cdef _update_centroid(np.ndarray[double,ndim=2] features, np.ndarray[int,ndim=1] label, int n_clusters):
    """
    更新されたラベルからセントロイドを更新 (M-step)
    """
    cdef int n_dim = features.shape[1]
    cdef np.ndarray[double,ndim=2] center = np.zeros( (n_clusters,n_dim), dtype=np.double )

    for k in xrange(n_clusters):
        subset = features[np.where(label == k)]
        if subset.size > 0:
            center[k] = subset.mean(0)
        
    return center
