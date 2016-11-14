# -*- coding: utf-8 -*-

"""
============================================================
@file   gmm_cy.pyx
@date   2014/05/07
@author sasai

@brief Gaussian Mixture Model

============================================================
"""

import numpy as np
cimport numpy as np
import scipy.linalg as linalg
import scipy.misc as spmisc
from ymh_mir.stats.cluster import *


class GMM_cy():
    EPS = np.finfo(float).eps

    """ GMM implementation """
    def __init__(self, n_components=50, n_iter=100, min_covar=1e-3, init="random"):
        self.n_obs = 0
        self.n_dims = 0
        self.n_components = n_components
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.init = init
        self.responsibilities = None
        self.weights = None
        self.means = None
        self.covars = None

    def init_parameter(self, X, method="random"):
        n_obs = X.shape[0]
        n_dims = X.shape[1]

        if method == "random":
            ## 乱数で初期化
            self.weights = np.ones(self.n_components, dtype=np.double) / self.n_components
            self.means = np.random.rand( self.n_components, n_dims ) * 2.0 - 1
            self.covars = np.zeros( (self.n_components, n_dims, n_dims), dtype=np.double )
            for k in range(self.n_components):
                self.covars[k] = np.eye(n_dims)

        elif method == "kmeans":
            ## k-means法で初期化
            label,centers = kmeans_cy(X, n_clusters=self.n_components, max_iter=50)

            self.weights = np.zeros(self.n_components, dtype=np.double)
            self.means = np.zeros( (self.n_components,n_dims) )
            self.covars = np.zeros( (self.n_components,n_dims,n_dims), dtype=np.double )

            # self.means = centers
            # for k in xrange(self.K):
            #     Xk_idx = np.where(label == k)[0]
            #     self.weights[k] = len(Xk_idx)
            #     Xk = self.X[Xk_idx]
            #     if Xk.size > 0:
            #         self.covars[k] = np.cov(Xk.T) + self.min_covar * np.eye(self.X.shape[1])
            #     else:
            #         self.covars[k] = self.min_covar * np.eye(self.X.shape[1])
            # self.weights /= self.weights.sum()

            self.weights[:] = 1.0 / self.n_components
            self.means = centers
            for k in range(self.n_components):
                self.covars[k] = np.cov(X.T) + self.min_covar * np.eye(n_dims)
            
        self.responsibilities = np.zeros( (n_obs,self.n_components), dtype=np.double )
        
    def train(self, features):
        """
        パラメータ学習(EMアルゴリズム)
        """

        ## initialize parameters
        self.n_obs,self.n_dims = features.shape
        self.init_parameter(features, method=self.init)

        cdef int it,n,c
        cdef np.ndarray[double,ndim=2] gmm_log_prob = np.zeros( (self.n_obs,self.n_components), dtype=np.double)
        cdef np.ndarray[double,ndim=2] gmm_prob = np.zeros( (self.n_obs,self.n_components), dtype=np.double)
        cdef np.ndarray[double,ndim=1] Nk = np.zeros(self.n_components, dtype=np.double)
        cdef np.ndarray[double,ndim=1] cur_rz = np.zeros(self.n_obs, dtype=np.double)
        cdef np.ndarray[double,ndim=2] cov_mat_sum = np.zeros( (self.n_dims,self.n_dims), dtype=np.double)
        cdef int n_obs = self.n_obs
        cdef int n_dims = self.n_dims
        cdef int n_components = self.n_components

        # debug: sklearn版と同じ初期値を与える
        # import scipy.io
        # m_dict = scipy.io.loadmat('init_params.mat')
        # self.means = m_dict['means']
        # self.covars = m_dict['covars']
        # self.weights = m_dict['weights'][0]
        
        ## EM algorithm
        for it from 0 <= it < self.n_iter:
            print( "iterates {0}".format(it) )

            ## e-step: update responsibility
            log_prob,self.responsibilities = self._compute_responsibility(features, self.means, self.covars, self.weights, self.min_covar)

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
            self.means = np.dot(self.responsibilities.T, features) / Nk[:,np.newaxis]

            # covars
            for c from 0 <= c < n_components:
                post = self.responsibilities[:,c]
                avg_cov = np.dot(post * features.T, features) / (post.sum() + 10 * cls.EPS)
                mu = self.means[c][np.newaxis]
                self.covars[c] = (avg_cov - np.dot(mu.T, mu)) + self.min_covar * np.eye(n_dims)

            # print 'weights:'
            # print self.weights
            # print 'means:'
            # print self.means
            # print 'covars:'
            # print self.covars

            L = self.log_likelihood(features)
            print( "log_likelihood={0}".format(L) )

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
        cdef int n_dims = X.shape[1]
        cdef int n_components = self.means.shape[0]
        log_prob = np.zeros( (n_obs,n_components), dtype=np.double )
        cdef int c,n
        cdef double covars_det = 0.0
        cdef np.ndarray[double,ndim=2] covars_inv = np.zeros( (n_dims,n_dims), dtype=np.double)
        cdef double term1,term2,lpdf

        for c from 0 <= c < n_components:
            try:
                cv_chol = linalg.cholesky(self.covars[c], lower=True)
            except linalg.LinAlgError:
                cv_chol = linalg.cholesky(self.covars[c] + self.min_covar * np.eye(n_dim), lower=True)
            cv_log_det = 2 * np.log( np.diagonal(cv_chol) ).sum()
            cv_sol = linalg.solve_triangular(cv_chol, (X-self.means[c]).T, lower=True ).T
            log_prob[:,c] = -0.5 * ( (cv_sol**2).sum(1) + n_dim * np.log(2*np.pi) + cv_log_det )
            log_prob[:,c] = np.log(self.weights[c]) + log_prob[:,c]

        return log_prob

    def save(self, dirname):
        import json
        import os
        params = {
            "n_obs": self.n_obs,
            "n_dims": self.n_dims,
            "n_components": self.n_components,
            "n_iter": self.n_iter,
            "min_covar": self.min_covar,
            "init": self.init,
            "responsibilities": self.responsibilities,
            "weights": self.weights,
            "means": self.means,
            "covars": self.covars
        }
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        fn_out = os.path.join(dirname, "params.json")
        json.dump( params, open(fn_out, "w") )

    @classmethod
    def load(cls, dirname):
        import json
        import os
        params = json.load( open( os.path.join(dirname, "params.json") ) )
        model = cls()
        model.n_obs = params["n_obs"]
        model.n_dims = params["n_dims"]
        model.n_components = params["n_components"]
        model.n_iter = params["n_iter"]
        model.min_covar = params["min_covar"]
        model.init = params["init"]
        model.responsibilities = params["responsibitilies"]
        model.weights = params["weights"]
        model.means = params["means"]
        model.covars = params["covars"]
        return model

    def _compute_responsibility(self, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] means,
                                np.ndarray[double,ndim=3] covars, np.ndarray[double,ndim=1] weights, double min_covar=1e-7):
        cdef int n_obs = X.shape[0]
        cdef int n_dims = X.shape[1]
        cdef int n_components = means.shape[0]
        log_prob = np.zeros( (n_obs,n_components), dtype=np.double )
        cdef int c,n
        cdef double covars_det = 0.0
        cdef np.ndarray[double,ndim=2] covars_inv = np.zeros( (n_dims,n_dims), dtype=np.double )
        cdef double term1,term2,lpdf

        for c from 0 <= c < n_components:
            try:
                cv_chol = linalg.cholesky(covars[c], lower=True)
            except linalg.LinAlgError:
                cv_chol = linalg.cholesky(covars[c] + min_covar * np.eye(n_dims), lower=True)
            cv_log_det = 2 * np.log( np.diagonal(cv_chol) ).sum()
            cv_sol = linalg.solve_triangular(cv_chol, (X-means[c]).T, lower=True ).T
            log_prob[:,c] = -0.5 * ( (cv_sol**2).sum(1) + n_dims * np.log(2*np.pi) + cv_log_det )
            log_prob[:,c] = np.log(weights[c]) + log_prob[:,c]

        rtn_log_prob = spmisc.logsumexp(log_prob, axis=1)
        responsibilities = np.exp(log_prob - rtn_log_prob[:,np.newaxis])

        return rtn_log_prob,responsibilities
