# -*- coding: utf-8 -*-

"""
@file vb_lda.py
@brief Variational Bayes LDA
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.special
from pylufia.stats.common import digamma,trigamma
import itertools

import time


class VBLDA():
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
        self.alpha = sp.rand(self.K)
        self.alpha /= self.alpha.sum()
        self.beta = sp.ones( (self.K,self.N) ) / self.N
        self.phi = sp.zeros( (self.D,self.N,self.K) )
        for d,n,k in itertools.product(xrange(self.D),xrange(self.N),xrange(self.K)):
            self.phi[d,n,k] = 1/float(self.K)
        
        self.gam = sp.zeros( (self.D,self.K) )
        for d,k in itertools.product(xrange(self.D),xrange(self.K)):
            self.gam[d,k] = self.alpha[k] + self.N/float(self.K)

        for it in xrange(self.n_iter):
            print 'iterates: {0}'.format(it)

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
            print 'time for e-step: {0}[s]'.format(t2-t1)

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
            print 'time for m-step: {0}[s]'.format(t2-t1)

            print 'alpha:{0}\nbeta:{1}\n'.format(self.alpha,self.beta)
            # L = self.likelihood()
            # L = _lda.likelihood_cy(self.documents, self.vb_gamma, self.vb_phi, self.alpha, self.beta, self.K)
            # print 'log likelihood={}'.format(L)

    def e_step(self):
        D,N = self.documents.shape
        phi_sum = sp.zeros(D, dtype=sp.double)

        # for d from 0 <= d < D:
        #     gam_sum = gam[d].sum()
        #     for n from 0 <= n < N:
        #         wn = documents[d,n]
        #         for k from 0 <= k < K:
        #             phi[d,n,k] = wn * beta[k,n] * np.exp(digamma(gam[d,k])-digamma(gam_sum))
        #         if phi[d,n].sum() > 0:
        #             phi[d,n] /= phi[d,n].sum()
        #     gamma[d] = alpha + phi[d].sum(0)

        gam_sums = self.gam.sum(1)
        for n in xrange(N):
            wns = self.documents[:,n]
            phi_sum[:] = 0.0
            for k in xrange(self.K):
                # print 'Wn:'
                # print wns
                # print 'Beta:'
                # print beta[k,n]
                # print 'Gamma:'
                # print np.exp(digamma(gam[:,k]))

                self.phi[:,n,k] = wns * self.beta[k,n] * sp.exp(digamma(self.gam[:,k])-digamma(gam_sums))
                # phi[:,n,k] = beta[k,n] * np.exp(digamma(gam[:,k])-digamma(gam_sums))
                # phi[:,n,k] = wns * beta[k,n] * np.exp(digamma(gam[:,k]))
                phi_sum += self.phi[:,n,k]
            phi_sum_ary = sp.tile(phi_sum, (self.K,1)).T
            self.phi[:,n] /= phi_sum_ary + 0.0000001

        self.gam = self.alpha + self.phi.sum(1)

    def m_step(self):
        D,N = self.documents.shape
        cur_phi = sp.zeros((D,N), dtype=sp.double)
        wj_arr = sp.zeros((D,N), dtype=sp.double)

        for i in xrange(self.K):
            cur_phi = self.phi[:,:,i]
            for j in xrange(N):
                wj_arr[:,:] = 0.0
                wj_arr[:,j] = self.documents[:,j]
                self.beta[i,j] = (cur_phi * wj_arr).sum()
            self.beta[i] /= self.beta[i].sum()

        # for j from 0 <= j < N:
        #     wj_arr[:,:] = 0.0
        #     wj_arr[:,j] = documents[:,j]
        #     beta[:,j] = (phi * wj_arr[:,:,np.newaxis]).sum((0,1))
        # beta /= beta.sum(1)[:,np.newaxis]

        # alpha
        self.alpha = self.newton_alpha(self.gam, 20, self.alpha)

    def likelihood(self):
        """
        モデル尤度を計算

        愚直に実装してみたが重すぎる．簡略化できないか？
        """
        gammafunc = scipy.special.gamma

        D = self.documents.shape[0]
        L = 0.0
        for d in xrange(D):
            print d
            cur_doc = self.documents[d]
            cur_phi = self.phi[d]
            cur_gam = self.gam[d]

            # E(log(p(theta|alpha)))
            E_logp_theta_alpha = sp.log(gammafunc(self.alpha.sum())) - sp.log(gammafunc(self.alpha)).sum() + (self.alpha-1)+(digamma(cur_gam)-digamma(cur_gam.sum())).sum()
        
            # E(log(p(z|theta)))
            E_logp_z_theta = 0.0
            N = cur_phi.shape[0]
            K = cur_phi.shape[1]
            for n in xrange(N):
                for k in xrange(K):
                    E_logp_z_theta += cur_phi[n,k] * (digamma(cur_gam[k])-digamma(cur_gam.sum()))
        
            # E(log(p(w|z,beta)))
            E_logp_w_z_beta = 0.0
            for n in xrange(N):
                for i in xrange(K):
                    for j in xrange(N):
                        if n == j:
                            wn = cur_doc[j]
                        else:
                            wn = 0
                        E_logp_w_z_beta += cur_phi[n,i]*wn*sp.log(self.beta[i,j]+0.000000001)

            # E(log(q(theta)))
            E_logq_theta = -sp.log(gammafunc(cur_gam.sum())) + sp.log(gammafunc(cur_gam)).sum() - ((cur_gam-1)*(digamma(cur_gam)-digamma(cur_gam.sum()))).sum()

            # E(log(q(z|phi)))
            E_logq_z_phi = -(cur_phi * sp.log(cur_phi+0.000000001)).sum()

            L += E_logp_theta_alpha + E_logp_z_theta + E_logp_w_z_beta + E_logq_theta + E_logq_z_phi

        return L

    def newton_alpha(self, gam, n_iter=20, init_alpha=None):
        """
        Newton-Raphson法によるalphaの逐次更新

        http://satomacoto.blogspot.jp/2009/12/pythonlda.html
        """
        D,K = gam.shape
        g = sp.zeros( (1,K) )
        pg = digamma(gam).sum(0) - digamma(gam.sum(1)).sum()
        # alpha = sp.rand(K)
        # alpha /= alpha.sum()
        alpha = init_alpha.copy()

        for it in xrange(n_iter):
            alpha0 = alpha.sum()
            g = D * (digamma(alpha0) - digamma(alpha)) + pg
            h = -1.0 / trigamma(alpha)
            hgz = sp.dot(h,g) / (1.0 / trigamma(alpha0) + h.sum())

            # for i in xrange(K):
            #     alpha[i] = alpha[i] - h[i] * (g[i] - hgz) / D
            alpha = alpha - h * (g - hgz) / D

        return alpha

    def mnormalize(self, m, d=0):
        """
        x = mnormalize(m, d)
        normalizes a 2-D matrix m along the dimension d.
        m : matrix
        d : dimension to normalize (default 0)
        """
        v = m.sum(d)
        if d == 0:
            return sp.dot(m, sp.diag(1.0 / v))
        else:
            return sp.dot(sp.diag(1.0 / v), m)
