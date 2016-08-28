# -*- coding: utf-8 -*-

"""
============================================================
@file   gmm.py
@date   2014/05/07
@author sasai

@brief
Gaussian Mixture Model

============================================================
"""

from ..prob_dists import MultiGauss
import scipy as sp
import scipy.misc as spmisc
from pylufia.stats.cluster import *

import scipy.linalg as linalg

class GMM():
    """ GMM implementation """
    def __init__(self, X, K=50, n_iter=100):
        self.X = X
        self.N = self.X.shape[0]
        self.K = K
        self.n_iter = n_iter
        self.rz = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def init_parameter(self, method='random'):
        if method == 'random':
            ## 乱数で初期化
            self.pi = sp.ones(self.K) / self.K
            self.mu = sp.rand( self.K, self.X.shape[1] ) * 2.0 - 1
            self.sigma = sp.zeros( (self.K, self.X.shape[1], self.X.shape[1]), dtype=sp.float64 )
            for k in range(self.K):
                self.sigma[k] = sp.eye(self.X.shape[1])
                # self.sigma[k] = sp.ones( (self.X.shape[1],self.X.shape[1]) )

        elif method == 'kmeans':
            ## k-means法で初期化
            whitenX = whiten_cy(self.X)
            label,center = kmeans_cy(whitenX, K=self.K, max_iter=50, n_rep=10)

            self.mu = center
            for k in range(self.K):
                cur_k_idx = sp.where(label == k)[0]
                self.pi[k] = len(cur_k_idx)
                cur_k_obs = self.X[cur_k_idx]
                self.sigma[k] = sp.cov(cur_k_obs)
            self.pi /= self.pi.sum()
            
        self.rz = sp.zeros( (self.N,self.K), dtype=sp.float64 )
        
    def train(self):
        """
        パラメータ学習(EMアルゴリズム)
        """

        ## initialize pi,mu,sigma
        self.init_parameter(method='random')

        for it in range(self.n_iter):
            print( 'iterates {0}'.format(it) )

            ## e-step: update rz
            for n in range(self.N):
                # 普通に計算する
                # gmm_pdfs = sp.zeros(self.K)
                # for k in xrange(self.K):
                #     gmm_pdfs[k] = self.pi[k] * MultiGauss(self.mu[k],self.sigma[k]).prob(self.X[n])

                # 対数領域にもっていってから計算する
                gmm_logpdfs = sp.zeros(self.K)
                for k in range(self.K):
                    gmm_logpdfs[k] = sp.log(self.pi[k]) + MultiGauss(self.mu[k],self.sigma[k]).log_prob(self.X[n])
                gmm_pdfs = sp.exp(gmm_logpdfs - spmisc.logsumexp(gmm_logpdfs))

                for k in range(self.K):
                    self.rz[n,k] = gmm_pdfs[k] / gmm_pdfs.sum(0)

            # self.rz = sp.maximum(self.rz, 1e-10)
            self.rz /= self.rz.sum(1)[:,sp.newaxis]

            print('rz:')
            print(self.rz)
            print( self.rz.min() )

            ## m-step: update mu,sigma,pi
            Nk = self.rz.sum(0)
            for k in range(self.K):
                cur_rz = self.rz[:,k]
                self.mu[k] = 1.0/Nk[k] * (cur_rz[:,sp.newaxis] * self.X ).sum(0)

            for k in range(self.K):
                cov_mat_sum = sp.zeros( (self.X.shape[1],self.X.shape[1]), dtype=sp.float64 )
                for n in range(self.N):
                    cov_mat_sum += self.rz[n,k] * sp.dot( (self.X[n]-self.mu[k])[:,sp.newaxis], (self.X[n]-self.mu[k])[sp.newaxis,:])
                # for d1 in xrange(self.X.shape[1]):
                #     for d2 in xrange(self.X.shape[1]):
                #         self.sigma[k,d1,d2] = ( self.rz[:,k] * (self.X[:,d1]-self.mu[k,d1]) * (self.X[:,d2]-self.mu[k,d2]) ).sum()
                self.sigma[k] = cov_mat_sum / Nk[k]

                # if linalg.det(self.sigma[k]) <= 0.0:
                #     print 'det(Sigma)=0'
                #     print self.sigma[k]

            for k in range(self.K):
                self.pi[k] = Nk[k] / float(self.N)

            print('pi:')
            print(self.pi)
            print('mu:')
            print(self.mu)
            print('sigma:')
            print(self.sigma)

            # L = self.likelihood(self.X)
            # print 'likelihood={0}'.format(L)

    def likelihood(self, X):
        """
        データXに対するモデルの尤度を計算する
        """
        L = 0.0
        if X.ndim == 2:
            N = X.shape[0]

            for n in range(N):
                gauss_sum = 0.0
                for k in range(self.K):
                    gauss_sum += sp.exp( sp.log(self.pi[k]) + MultiGauss(self.mu[k],self.sigma[k]).log_prob(X[n]) )
                L += gauss_sum

        elif X.ndim == 1:
            for k in range(self.K):
                L += sp.exp( sp.log(self.pi[k]) + MultiGauss(self.mu[k],self.sigma[k]).log_prob(X) )

    def likelihood_1d(self, x):
        L = 0.0
        for k in range(self.K):
            L += sp.exp( sp.log(self.pi[k]) + MultiGauss(self.mu[k],self.sigma[k]).log_prob(x) )

        return L
