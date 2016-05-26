# -*- coding: utf-8 -*-

"""
@file plsi.py
@brief pLSI(probabilistic Latent Semantic Indexing) implementation
@author ふぇいと (@stfate)

@description
PLSI(Probabilistic Latent Semantic Indexing)の実装
"""

import scipy as sp


class PLSI():
    def __init__(self):
        self.p_z = None
        self.p_dz = None
        self.p_wz = None
        self.p_z_dw = None

    def train(self, feature, n_z=10, n_iter=100):
        n_d = feature.shape[0]
        n_w = feature.shape[1]

        # initialize
        self.p_z = sp.rand(n_z)
        self.p_z /= self.p_z.sum()
        self.p_d_z = sp.rand(n_d,n_z)
        self.p_d_z /= self.p_d_z.sum(0)
        self.p_w_z = sp.rand(n_w,n_z)
        self.p_w_z /= self.p_w_z.sum(0)
        self.p_z_dw = sp.zeros((n_z,n_d,n_w))

        # update parameter
        for it in xrange(n_iter):
            print it
            L = 0.0

            # E-step: 各文書dの各単語wについて，トピック分布p(z|d,w)を計算
            for d in xrange(n_d):
                for w in xrange(n_w):
                    n_dw = feature[d,w] / float(feature[d].sum())
                    p_dw = 0.0
                    for z in xrange(n_z):
                        self.p_z_dw[z,d,w] = self.p_z[z] * self.p_d_z[d,z] * self.p_w_z[w,z]
                        p_dw += self.p_z[z] * self.p_w_z[w,z] * self.p_d_z[d,z]
                    L += n_dw * sp.log(p_dw)
                    self.p_z_dw[:,d,w] = self._normalize(self.p_z_dw[:,d,w])

            print 'log-prob={0}'.format(L)

            # M-step: p(z|d,w)を用いて，p(z),p(d|z),p(w|z)を計算
            self.p_z = sp.zeros(n_z)
            self.p_d_z = sp.zeros((n_d,n_z))
            self.p_w_z = sp.zeros((n_w,n_z))

            for z in xrange(n_z):
                for d in xrange(n_d):
                    for w in xrange(n_w):
                        n_dw = feature[d,w] / float(feature[d].sum())
                        score = self.p_z_dw[z,d,w] * n_dw
                        self.p_z[z] += score
                        self.p_d_z[d,z] += score
                        self.p_w_z[w,z] += score
                self.p_d_z[:,z] = self._normalize(self.p_d_z[:,z])
                self.p_w_z[:,z] = self._normalize(self.p_w_z[:,z])
            self.p_z = self._normalize(self.p_z)

    def _normalize(self, p_arr):
        if p_arr.sum() > 0:
            norm_p_arr = p_arr / p_arr.sum()
        else:
            norm_p_arr = p_arr

        return norm_p_arr

