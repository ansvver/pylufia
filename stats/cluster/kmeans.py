# -*- coding: utf-8 -*-

"""
@file kmeans.py
@brief kmeans clustering
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import random

import time


def kmeans(features, n_clusters=2, max_iter=50):
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
    n_obs = features.shape[0]
    n_dims = features.shape[1]
    labels = sp.zeros(n_obs)
    old_labels = sp.zeros(n_obs)
    centers = sp.zeros( (n_clusters,n_dims) )
    
    error = 0.0
    
    # 初期クラスタの割り当て
    labels = (sp.rand(n_obs) * n_clusters).astype('int')
        
    # 初期セントロイド
    for k in xrange(n_clusters):
        subset = features[sp.where(labels == k)]
        centers[k] = subset.mean(0)
    
    old_labels = sp.zeros(n_obs)
    
    # k-meansクラスタリングiteration
    it = 0
    # while (numIter < maxIter and sp.sum(abs(label-old_label)) != 0):
    while (it < max_iter):
        print it
        # 直前のクラスタ分割を保存
        old_labels = labels.copy()
        
        # E-step: 各観測をセントロイドとの距離が最小になるクラスタへ再配置
        for i in xrange(n_obs):
            D = sp.array([sp.sqrt( ( (features[i]-centers[k])**2 ).sum() ) for k in xrange(n_clusters)])
            error += (D**2).sum()
            min_idx = sp.argmin(D)
            labels[i] = min_idx
            
        # M-step: セントロイドの更新
        for k in xrange(n_clusters):
            subset = features[sp.where(labels == k)]
            centers[k] = subset.mean(0)
            
        # if sp.sum(abs(label-old_label)) == 0:
        #     break
        it += 1
        
    return labels, centers
