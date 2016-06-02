# -*- coding: utf-8 -*-

"""
@file gmm_cy.pyx
@brief GMM
@author ふぇいと (@stfate)

@description

"""

import numpy as np
cimport numpy as np
import scipy.linalg as linalg
import scipy.misc as spmisc
from pylufia.stats.cluster import *


EPS = np.finfo(float).eps

class GMM_cy():
    """ GMM implementation """
    def __init__(self, X, n_mixtures=50, n_iter=100, min_covar=1e-3, init='random'):
        self.X = X
        self.Xmean = self.X.mean(0)
        self.Xvar = self.X.var(0)
        self.n_obs = self.X.shape[0]
        self.n_mixtures = n_mixtures
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.init = init
        self.responsibilities = None
        self.weights = None
        self.means = None
        self.covars = None

    def init_parameter(self, method='random'):
        if method == 'random':
            ## 乱数で初期化
            self.weights = np.ones(self.n_mixtures, dtype=np.double) / self.n_mixtures
            self.means = np.random.rand( self.n_mixtures, self.X.shape[1] ) * 2.0 - 1
            self.covars = np.zeros( (self.n_mixtures, self.X.shape[1], self.X.shape[1]), dtype=np.double )
            for k in xrange(self.n_mixtures):
                self.covars[k] = np.eye(self.X.shape[1])

        elif method == 'kmeans':
            ## k-means法で初期化
            label,centers = kmeans_cy(self.X, n_clusters=self.n_mixtures, max_iter=100, init="pp")

            self.weights = np.zeros(self.n_mixtures, dtype=np.double)
            self.means = np.zeros( (self.n_mixtures,self.X.shape[1]) )
            self.covars = np.zeros( (self.n_mixtures,self.X.shape[1],self.X.shape[1]), dtype=np.double )

            # self.means = centers
            # for k in xrange(self.n_mixtures):
            #     Xk_idx = np.where(label == k)[0]
            #     self.weights[k] = len(Xk_idx)
            #     Xk = self.X[Xk_idx]
            #     if Xk.size > 0:
            #         self.covars[k] = np.cov(Xk.T) + self.min_covar * np.eye(self.X.shape[1])
            #     else:
            #         self.covars[k] = self.min_covar * np.eye(self.X.shape[1])
            # self.weights /= self.weights.sum()

            self.weights[:] = 1.0 / self.n_mixtures
            self.means = centers
            for k in xrange(self.n_mixtures):
                self.covars[k] = np.cov(self.X.T) + self.min_covar * np.eye(self.X.shape[1])
            
        self.responsibilities = np.zeros( (self.n_obs,self.n_mixtures), dtype=np.double )
        
    def train(self):
        """
        パラメータ学習(EMアルゴリズム)
        """

        ## initialize parameters
        self.init_parameter(method=self.init)

        cdef int it,n,k
        cdef np.ndarray[double,ndim=2] gmm_log_prob = np.zeros( (self.n_obs,self.n_mixtures), dtype=np.double)
        cdef np.ndarray[double,ndim=2] gmm_prob = np.zeros( (self.n_obs,self.n_mixtures), dtype=np.double)
        cdef np.ndarray[double,ndim=1] Nk = np.zeros(self.n_mixtures, dtype=np.double)
        cdef np.ndarray[double,ndim=1] cur_rz = np.zeros(self.n_obs, dtype=np.double)
        cdef np.ndarray[double,ndim=2] cov_mat_sum = np.zeros((self.X.shape[1],self.X.shape[1]), dtype=np.double)
        cdef int n_obs = self.n_obs
        cdef int K = self.n_mixtures

        # debug: sklearn版と同じ初期値を与える
        # import scipy.io
        # m_dict = scipy.io.loadmat('init_params.mat')
        # self.means = m_dict['means']
        # self.covars = m_dict['covars']
        # self.weights = m_dict['weights'][0]
        
        ## EM algorithm
        for it from 0 <= it < self.n_iter:
            print 'iterates {0}'.format(it)

            ## e-step: update responsibility
            log_prob,self.responsibilities = self._compute_responsibility(self.X, self.means, self.covars, self.weights, self.min_covar)

            # L = log_prob.sum()
            # print 'log likelihood={0}'.format(L)

            # print 'responsibilities:'
            # print self.responsibilities
            # print self.responsibilities.min()

            ## m-step: update means,covars,weights
            Nk = self.responsibilities.sum(0)

            # weights
            self.weights = Nk / Nk.sum()
            
            # means
            self.means = np.dot(self.responsibilities.T, self.X) / Nk[:,np.newaxis]

            # covars
            n_dims = self.X.shape[1]
            for k from 0 <= k < K:
                post = self.responsibilities[:,k]
                avg_cov = np.dot(post * self.X.T, self.X) / (post.sum() + 10 * EPS)
                mu = self.means[k][np.newaxis]
                self.covars[k] = (avg_cov - np.dot(mu.T, mu)) + self.min_covar * np.eye(n_dims)

            # print 'weights:'
            # print self.weights
            # print 'means:'
            # print self.means
            # print 'covars:'
            # print self.covars

            L = self.log_likelihood(self.X)
            print 'log_likelihood={0}'.format(L)

    def log_likelihood(self, X):
        """
        データXに対するモデルの尤度を計算する
        """
        log_prob,responsibilities = self._compute_responsibility(X, self.means, self.covars, self.weights, self.min_covar)
        L = log_prob.sum()

        return L

    def predict_cluster(self, X):
        """
        クラスタ尤度を計算する
        """
        cdef int n_obs = X.shape[0]
        cdef int n_dim = X.shape[1]
        cdef int n_mixtures = self.means.shape[0]
        log_prob = np.zeros((n_obs,n_mixtures), dtype=np.double)
        cdef int k,n
        cdef double covars_det = 0.0
        cdef np.ndarray[double,ndim=2] covars_inv = np.zeros((X.shape[1],X.shape[1]), dtype=np.double)
        cdef double term1,term2,lpdf

        for k from 0 <= k < n_mixtures:
            try:
                cv_chol = linalg.cholesky(self.covars[k], lower=True)
            except linalg.LinAlgError:
                cv_chol = linalg.cholesky(self.covars[k] + self.min_covar * np.eye(n_dim), lower=True)
            cv_log_det = 2 * np.log( np.diagonal(cv_chol) ).sum()
            cv_sol = linalg.solve_triangular(cv_chol, (X-self.means[k]).T, lower=True ).T
            log_prob[:,k] = -0.5 * ( (cv_sol**2).sum(1) + n_dim * np.log(2*np.pi) + cv_log_det )
            log_prob[:,k] = np.log(self.weights[k]) + log_prob[:,k]

        return log_prob


    def _compute_responsibility(
        self,
        np.ndarray[double,ndim=2] X,
        np.ndarray[double,ndim=2] means,
        np.ndarray[double,ndim=3] covars,
        np.ndarray[double,ndim=1] weights,
        double min_covar=1e-7):
        cdef int n_obs = X.shape[0]
        cdef int n_dim = X.shape[1]
        cdef int n_mixtures = means.shape[0]
        log_prob = np.zeros((n_obs,n_mixtures), dtype=np.double)
        cdef int k,n
        cdef double covars_det = 0.0
        cdef np.ndarray[double,ndim=2] covars_inv = np.zeros((X.shape[1],X.shape[1]), dtype=np.double)
        cdef double term1,term2,lpdf

        for k from 0 <= k < n_mixtures:
            try:
                cv_chol = linalg.cholesky(covars[k], lower=True)
            except linalg.LinAlgError:
                cv_chol = linalg.cholesky(covars[k] + min_covar * np.eye(n_dim), lower=True)
            cv_log_det = 2 * np.log( np.diagonal(cv_chol) ).sum()
            cv_sol = linalg.solve_triangular(cv_chol, (X-means[k]).T, lower=True ).T
            log_prob[:,k] = -0.5 * ( (cv_sol**2).sum(1) + n_dim * np.log(2*np.pi) + cv_log_det )
            log_prob[:,k] = np.log(weights[k]) + log_prob[:,k]

        rtn_log_prob = spmisc.logsumexp(log_prob, axis=1)
        responsibilities = np.exp(log_prob - rtn_log_prob[:,np.newaxis])

        return rtn_log_prob,responsibilities
