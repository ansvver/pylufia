# -*- coding: utf-8 -*-

"""
vb_lda_cy.pyx

変分ベイズLDAのCython実装
"""

import cython
import numpy as np
cimport numpy as np
import scipy.special
from scipy.special import polygamma

import itertools
import time


cdef digamma(x):
    return polygamma(0, x)

cdef trigamma(x):
    return polygamma(1, x)

cdef sum_2d(np.ndarray[double,ndim=2] X):
    cdef int M = X.shape[0]
    cdef int N = X.shape[1]
    cdef int m,n
    cdef double sumval = 0.0

    for m from 0 <= m < M:
        for n from 0 <= n < N:
            sumval += X[m,n]

    return sumval

cdef newton_alpha_cy(gam, n_iter=20, init_alpha=None):
    """
    Newton-Raphson法によるalphaの逐次更新

    http://satomacoto.blogspot.jp/2009/12/pythonlda.html
    """
    D,K = gam.shape
    g = np.zeros( (1,K) )
    pg = digamma(gam).sum(0) - digamma(gam.sum(1)).sum()
    alpha = init_alpha.copy()

    cdef int it
    cdef double alpha0
    for it in xrange(n_iter):
        alpha0 = alpha.sum()
        g = D * (digamma(alpha0) - digamma(alpha)) + pg
        h = -1.0 / trigamma(alpha)
        hgz = np.dot(h,g) / (1.0 / trigamma(alpha0) + h.sum())

        # for i in xrange(K):
        #     alpha[i] = alpha[i] - h[i] * (g[i] - hgz) / D
        alpha = alpha - h * (g - hgz) / D

    return alpha


class VBLDA_cy():
    """
    Latent Dirichlet Allocation class

    推論はVariational Bayesで行う(Bleiの原論文ほぼそのまま)
    """
    def __init__(self, documents, K, alpha=None, beta=None, n_iter=100):
        self.documents = documents
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter

        self.gam = None
        self.phi = None

    def infer(self):
        """
        LDAのパラメータ推論を行う
        """
        ## initialize
        self.D,self.N = self.documents.shape
        self.alpha = np.zeros(self.K, dtype=np.double)
        self.alpha[:] = np.random.rand(self.K)
        self.alpha /= self.alpha.sum()
        self.beta = np.ones( (self.K,self.N), dtype=np.double ) / self.N
        self.phi = np.zeros( (self.D,self.N,self.K), dtype=np.double )
        for d,n,k in itertools.product(xrange(self.D),xrange(self.N),xrange(self.K)):
            self.phi[d,n,k] = 1/float(self.K)
        
        self.gam = np.zeros( (self.D,self.K), dtype=np.double )
        for d,k in itertools.product(xrange(self.D),xrange(self.K)):
            self.gam[d,k] = self.alpha[k] + self.N/float(self.K)

        cdef int it
        for it from 0 <= it < self.n_iter:
            print( 'iterates: {0}'.format(it) )

            ## E-step: phiとgammaを更新
            t1 = time.clock()

            # for d in xrange(self.D):
            #     for n in xrange(self.N):
            #         for k in xrange(self.K):
            #             beta = self.beta[k,n] if self.documents[d,n] > 0 else 0.0
            #             self.vb_phi[d,n,k] = beta * sp.exp(digamma(self.vb_gamma[d,k])-digamma(self.vb_gamma[d].sum()))
            #         if self.vb_phi[d,n].sum() > 0:
            #             self.vb_phi[d,n] /= self.vb_phi[d,n].sum()
            #     self.vb_gamma[d] = self.alpha + self.vb_phi[d].sum(0)
            
            self.e_step()
            t2 = time.clock()
            print( 'time for e-step: {0}[s]'.format(t2-t1) )

            # print 'phi:{0}\ngamma:{0}\n'.format(self.phi, self.gam)

            ## M-step: alphaとbetaを更新
            t1 = time.clock()
            # beta
            # for i,j in itertools.product(xrange(self.K),xrange(self.N)):
            # for i in xrange(self.K):
            #     for j in xrange(self.N):
            #         wj_arr = sp.zeros( (self.D,self.N) )
            #         wj_arr[:,j] = self.documents[:,j]
            #         self.beta[i,j] = (self.vb_phi[:,:,i] * wj_arr).sum()
            #     self.beta[i] /= self.beta[i].sum()

            # # alpha
            # self.alpha = self.newton_alpha(self.vb_gamma, 20, self.alpha)
            self.m_step()
            t2 = time.clock()
            print( 'time for m-step: {0}[s]'.format(t2-t1) )

            print( 'alpha:{0}\nbeta:{1}\n'.format(self.alpha,self.beta) )
            L = self.likelihood(self.documents)
            # L = _lda.likelihood_cy(self.documents, self.vb_gamma, self.vb_phi, self.alpha, self.beta, self.K)
            print( 'log likelihood={}'.format(L) )

    def e_step(self):
        cdef int D = self.documents.shape[0]
        cdef int N = self.documents.shape[1]
        cdef int d,n,k
        cdef double _b, gam_sum
        cdef np.ndarray[double,ndim=1] phi_sum = np.zeros(D, dtype=np.double)

        # for d from 0 <= d < D:
        #     gam_sum = gam[d].sum()
        #     for n from 0 <= n < N:
        #         wn = documents[d,n]
        #         for k from 0 <= k < K:
        #             phi[d,n,k] = wn * beta[k,n] * np.exp(digamma(gam[d,k])-digamma(gam_sum))
        #         if phi[d,n].sum() > 0:
        #             phi[d,n] /= phi[d,n].sum()
        #     gamma[d] = alpha + phi[d].sum(0)

        # gam_sums = self.gam.sum(1)
        # for n from 0 <= n < N:
        #     wns = self.documents[:,n]
        #     phi_sum[:] = 0.0
        #     for k from 0 <= k < self.K:
        #         # print 'Wn:'
        #         # print wns
        #         # print 'Beta:'
        #         # print beta[k,n]
        #         # print 'Gamma:'
        #         # print np.exp(digamma(gam[:,k]))

        #         self.phi[:,n,k] = wns * self.beta[k,n] * np.exp(digamma(self.gam[:,k])-digamma(gam_sums))
        #         # phi[:,n,k] = beta[k,n] * np.exp(digamma(gam[:,k])-digamma(gam_sums))
        #         # phi[:,n,k] = wns * beta[k,n] * np.exp(digamma(gam[:,k]))
        #         phi_sum += self.phi[:,n,k]
        #     phi_sum_ary = np.tile(phi_sum, (self.K,1)).T
        #     self.phi[:,n] /= phi_sum_ary + 0.0000001

        # self.gam = self.alpha + self.phi.sum(1)

        wn = np.zeros(N)
        for d from 0 <= d < D:
            wn[:] = 0
            v = np.where(self.documents[d] > 0)[0]
            wn[v] = 1
            cur_gam = self.gam[d]
            self.phi[d] = np.dot((wn[np.newaxis,:]*self.beta).T, np.diag( np.exp( digamma(cur_gam) - digamma(cur_gam.sum() ) ) ) )
            self.phi[d] /= self.phi[d].sum(1)[:,np.newaxis] + 1e-7
            
            self.gam[d] = self.alpha + self.phi[d].sum(0)

    def m_step(self):
        cdef int D = self.documents.shape[0]
        cdef int N = self.documents.shape[1]
        cdef int i,j
        cdef np.ndarray[double,ndim=2] cur_phi = np.zeros((D,N), dtype=np.double)
        cdef np.ndarray[double,ndim=2] wj_arr = np.zeros((D,N), dtype=np.double)

        # for i from 0 <= i < self.K:
        #     cur_phi = self.phi[:,:,i]
        #     for j from 0 <= j < N:
        #         wj_arr[:] = 0.0
        #         wj_arr[:,j] = self.documents[:,j]
        #         self.beta[i,j] = sum_2d(cur_phi * wj_arr)
        #     self.beta[i] /= self.beta[i].sum()

        # for j from 0 <= j < N:
        #     wj_arr[:] = 0.0
        #     wj_arr[:,j] = self.documents[:,j]
        #     for i from 0 <= i < self.K:
        #         cur_phi = self.phi[:,:,i]
        #         # self.beta[i,j] = sum_2d(cur_phi * wj_arr)
        #         self.beta[i,j] = (cur_phi * wj_arr).sum()
        # self.beta /= self.beta.sum(1)[:,np.newaxis]

        # for j from 0 <= j < N:
        #     wj_arr[:] = 0.0
        #     wj_arr[:,j] = self.documents[:,j]
        #     self.beta[:,j] = (self.phi * wj_arr[:,:,np.newaxis]).sum((0,1))
        # self.beta /= self.beta.sum(1)[:,np.newaxis]

        cdef int d
        self.beta[:] = 0.0
        for d from 0 <= d < D:
            self.beta += np.dot( np.diag(self.documents[d]), self.phi[d] ).T
        self.beta /= self.beta.sum(1)[:,np.newaxis]

        # alpha
        self.alpha = newton_alpha_cy(self.gam, 20, self.alpha)

    def likelihood(self, documents):
        """
        モデル尤度を計算

        愚直に実装してみたが重すぎる．簡略化できないか？
        """
        egamma = self.gam / self.gam.sum(1)[:,np.newaxis]
        L = 0
        D = len(documents)
        for d from 0 <= d < D:
            # lik += (matrix(t[1]) * log(matrix(beta[t[0],:]) * egamma[i,:].T))[0,0]
            L += np.dot( np.dot( np.diag(documents[d]), np.log(self.beta).T ), egamma[d,:].T )
        return L

    def mnormalize(self, m, d=0):
        """
        x = mnormalize(m, d)
        normalizes a 2-D matrix m along the dimension d.
        m : matrix
        d : dimension to normalize (default 0)
        """
        v = m.sum(d)
        if d == 0:
            return np.dot(m, np.diag(1.0 / v))
        else:
            return np.dot(np.diag(1.0 / v), m)
