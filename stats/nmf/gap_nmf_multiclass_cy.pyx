# -*- coding: utf-8 -*-

"""
GaPNMF for multiclass data cython implementation
"""

cimport cython
import numpy as np
cimport numpy as np
from ymh_mir.stats import GIG_cy

import time

class GaPNMF_multiclass_cy():
    def __init__(self, X, n_class=4, aw=0.1, bw=0.1, ah=0.1, bh=0.1, alpha=1.0, c=1.0, K=100, smoothness=100):
        self.X = X / X.mean()
        self.nF,self.nT = self.X.shape
        self.n_class = n_class
        self.aw = aw
        self.bw = bw
        self.ah = ah
        self.bh = bh
        self.alpha = alpha
        self.c = c
        self.K = K
        self.smoothness = smoothness

        # np.random.seed(98765)

        # initialize parameters
        self.rhow = np.zeros((self.n_class,self.nF,self.K), dtype=np.double)
        self.tauw = np.zeros((self.n_class,self.nF,self.K), dtype=np.double)
        self.rhoh = np.zeros((self.n_class,self.K,self.nT), dtype=np.double)
        self.tauh = np.zeros((self.n_class,self.K,self.nT), dtype=np.double)
        self.rhot = np.zeros((self.n_class,self.K), dtype=np.double)
        self.taut = np.zeros((self.n_class,self.K), dtype=np.double)
        self.Ew = np.zeros((self.n_class,self.nF,self.K), dtype=np.double)
        self.Ew_inv = np.zeros((self.n_class,self.nF,self.K), dtype=np.double)
        self.Ew_inv_inv = np.zeros((self.n_class,self.nF,self.K), dtype=np.double)
        self.Eh = np.zeros((self.n_class,self.K,self.nT), dtype=np.double)
        self.Eh_inv = np.zeros((self.n_class,self.K,self.nT), dtype=np.double)
        self.Eh_inv_inv = np.zeros((self.n_class,self.K,self.nT), dtype=np.double)
        self.Et = np.zeros((self.n_class,self.K), dtype=np.double)
        self.Et_inv = np.zeros((self.n_class,self.K), dtype=np.double)
        self.Et_inv_inv = np.zeros((self.n_class,self.K), dtype=np.double)
        self.gigW = None
        self.gigH = None
        self.gigT = None

    def _init_parameters(self, supervised=False):
        if not supervised:
            self.rhow = 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.nF,self.K))
            self.tauw = 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.nF,self.K))
        self.rhoh = 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.K,self.nT))
        self.tauh = 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.K,self.nT))
        self.rhot = self.K * 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.K))
        self.taut = 1./self.K * 10000 * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.n_class,self.K))

        # if not supervised:
        #     self.rhow = np.random.gamma(100, 1/1000., size=(self.n_class,self.nF,self.K))
        #     self.tauw[:] = 0.1
        # self.rhoh = np.random.gamma(100, 1/1000., size=(self.n_class,self.K,self.nT))
        # self.tauh[:] = 0.1
        # self.rhot = np.random.gamma(100, 1/1000., size=(self.n_class,self.K))
        # self.taut[:] = 0.1

        self.gigW = GIG_cy(self.aw, self.rhow[0], self.tauw[0])
        self.gigH = GIG_cy(self.ah, self.rhoh[0], self.tauh[0])
        self.gigT = GIG_cy(self.alpha/float(self.K), self.rhot[0], self.taut[0])

        self._compute_expectations(supervised=supervised)

    def _compute_expectations(self, supervised=False):
        """
        W,H,thetaの期待値E[y]とE[1/y]を計算する
        """
        for c in xrange(self.n_class):
            if not supervised:
                # gig_W = GIG_cy(self.a, self.rhow, self.tauw)
                self.gigW.update(self.aw, self.rhow[c], self.tauw[c])
                self.Ew[c] = self.gigW.expectation()
                self.Ew_inv[c] = self.gigW.inv_expectation()
                self.Ew_inv_inv[c] = 1.0 / self.Ew_inv[c]

            # gig_theta = GIG_cy(self.alpha/float(self.K), self.rhot, self.taut)
            self.gigT.update(self.alpha/float(self.K), self.rhot[c], self.taut[c])
            self.Et[c] = self.gigT.expectation()
            self.Et_inv[c] = self.gigT.inv_expectation()
            self.Et_inv_inv[c] = 1.0 / self.Et_inv[c]

            # gig_H = GIG_cy(self.b, self.rhoh, self.tauh)
            self.gigH.update(self.ah, self.rhoh[c], self.tauh[c])
            self.Eh[c] = self.gigH.expectation()
            self.Eh_inv[c] = self.gigH.inv_expectation()
            self.Eh_inv_inv[c] = 1.0 / self.Eh_inv[c]

    def infer(self, n_iter=100, show_prompt=False):
        """
        GaP-NMFパラメータの推論

        推論は変分ベイズ．
        """
        self._init_parameters(supervised=False)

        cdef int it = 0
        cdef double D = 0.0
        for it from 0 <= it < n_iter:
            if show_prompt:
                print 'iterates: {0}'.format(it)

            self._updateH()
            self._updateW()
            self._updateTheta()
            # self._clearBadK()

            if show_prompt:
                D = self.KLdivergence()
                print 'KLD={0}'.format(D)

        self._clearBadK()

        W = self.Ew
        H = self.Eh
        theta = self.Et

        return W,H,theta

    def infer_activation(self, Ew, Ew_inv, n_iter=100, update_w=False, show_prompt=False):
        """
        W(とtheta)は与えられたものを使い，Hのみを推論する
        """

        for k in xrange(len(self.Ew)):
            cur_K = Ew[k].shape[1]
            self.Ew[k,:,:cur_K] = Ew[k]
            self.Ew_inv[k,:,:cur_K] = Ew_inv[k]
            self.Ew_inv_inv[k,:,:cur_K] = 1.0 / self.Ew_inv[k,:,:cur_K]

        self._init_parameters(supervised=True)

        cdef int it
        cdef double D = 0.0
        # cdef np.ndarray[int,ndim=1] goodk = np.arange(self.K, dtype=np.int32)
        for it from 0 <= it < n_iter:
            if show_prompt:
                print 'iterates: {0}'.format(it)

            self._updateH()
            self._updateTheta()
            if update_w:
                self._updateW()

            if show_prompt:
                D = self.KLdivergence()
                print 'KLD={0}'.format(D)

        # self._clearBadK(supervised=True)

        H = self.Eh

        return H

    def infer_activation_with_free_basis(self, Ew, Ew_inv, Et=None, Et_inv=None, ahx=10, ahu=0.01, n_iter=100, n_free_basis=10, update_w=False, show_prompt=False):
        """
        Wは与えられたものを使い，Activation(HとTheta)のみを推論する
        ただしフリーな基底をいくつか用意し，どの教師基底にも属さない成分を吸収する
        """
        self.orig_K = Ew.shape[1]
        self.K = Ew.shape[1] + n_free_basis
        self.Ew = np.zeros( (Ew.shape[0],Ew.shape[1]+n_free_basis) )
        self.EW_inv = np.zeros( (Ew_inv.shape[0],Ew_inv.shape[1]+n_free_basis) )
        self.Ew[:,:Ew.shape[1]] = Ew
        self.Ew_inv[:,:Ew_inv.shape[1]] = Ew_inv
        self.Ew_inv_inv = 1.0 / self.Ew_inv
        self.ah = np.zeros( (self.K,self.Eh.shape[1]),dtype=np.double )
        self.ah[:self.Eh.shape[0],:] = ahx
        self.ah[self.Eh.shape[0]:,:] = ahu

        self._init_parameters(supervised=True)

        cdef int it
        cdef double D = 0.0
        cdef np.ndarray[int,ndim=1] goodk = np.arange(self.K, dtype=np.int32)
        for it from 0 <= it < n_iter:
            print 'iterates: {0}'.format(it)

            self._updateH(goodk=goodk)
            self._updateTheta(goodk=goodk)
            self._updateW_withFreeBasis(Ew, Ew_inv, goodk=goodk)

            if show_prompt:
                D = self.KLdivergence()
                print 'KLD={0}'.format(D)

        # self._clearBadK(supervised=True)

        H = self.Eh[:self.orig_K,:]

        return H

    def _E_tWH(self):
        E_tWH = np.zeros_like(self.X)
        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew = self.Ew[c]
            _Et = self.Et[c]
            _Eh = self.Eh[c]
            E_tWH += np.dot(_Ew[:,goodk], _Et[goodk,np.newaxis]*_Eh[goodk,:])
        return E_tWH

    def _E_tWH_inv(self, goodk=None):
        E_tWH_inv = np.zeros_like(self.X)
        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew_inv_inv = self.Ew_inv_inv[c]
            _Et_inv_inv = self.Et_inv_inv[c]
            _Eh_inv_inv = self.Eh_inv_inv[c]
            E_tWH_inv += np.dot(_Ew_inv_inv[:,goodk], _Et_inv_inv[goodk,np.newaxis]*_Eh_inv_inv[goodk,:])
        return E_tWH_inv

    def _updateW(self, goodk=None):
        """
        Update parameters of W [rhow,tauw]
        """
        # if goodk is None:
        #     goodk = self._goodK()

        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew = self.Ew[c]
            _Eh = self.Eh[c]
            _Et = self.Et[c]
            _Ew_inv_inv = self.Ew_inv_inv[c]
            _Eh_inv_inv = self.Eh_inv_inv[c]
            _Et_inv_inv = self.Et_inv_inv[c]

            E_tWH = self._E_tWH()
            E_tWH_inv = self._E_tWH_inv()
            # E_tWH = np.dot(_Ew[:,goodk], _Et[goodk,np.newaxis]*_Eh[goodk,:])
            # E_tWH_inv = np.dot(_Ew_inv_inv[:,goodk], _Et_inv_inv[goodk,np.newaxis]*_Eh_inv_inv[goodk,:])
            XX = self.X * E_tWH_inv**(-2)
            
            _rhow = np.zeros_like(self.rhow[c])
            _tauw = np.zeros_like(self.tauw[c])
            _rhow[:,goodk] = self.bw + np.dot(E_tWH**(-1), _Et[goodk] * _Eh[goodk,:].T)
            _tauw[:,goodk] = _Ew_inv_inv[:,goodk]**2 * np.dot(XX, _Et_inv_inv[goodk] * _Eh_inv_inv[goodk,:].T)
            _tauw[_tauw < 1e-100] = 0.0
            self.rhow[c] = _rhow
            self.tauw[c] = _tauw
            # self.rhow[c,:,goodk] = self.bw + np.dot(E_tWH**(-1), _Et[goodk] * _Eh[goodk,:].T)
            # self.tauw[c,:,goodk] = _Ew_inv_inv[:,goodk]**2 * np.dot(XX, _Et_inv_inv[goodk] * _Eh_inv_inv[goodk,:].T)
            # self.tauw[self.tauw < 1e-100] = 0

            # gig_W = GIG_cy(self.a, self.rhow[:,goodk], self.tauw[:,goodk])
            self.gigW.update(self.aw, _rhow[:,goodk], _tauw[:,goodk])
            _Ew = self.gigW.expectation()
            self.Ew[c] = _Ew[:,goodk]
            _Ew_inv = self.gigW.inv_expectation()
            self.Ew_inv[c] = _Ew_inv[:,goodk]
            _Ew_inv_inv = 1.0 / _Ew_inv
            self.Ew_inv_inv[c] = _Ew_inv_inv[:,goodk]

    def _updateH(self, goodk=None):
        """
        Update parameters of H [rhoh,tauh]
        """
        # if goodk is None:
        #     goodk = self._goodK()

        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew = self.Ew[c]
            _Eh = self.Eh[c]
            _Et = self.Et[c]
            _Ew_inv_inv = self.Ew_inv_inv[c]
            _Eh_inv_inv = self.Eh_inv_inv[c]
            _Et_inv_inv = self.Et_inv_inv[c]

            E_tWH = self._E_tWH()
            E_tWH_inv = self._E_tWH_inv()
            # E_tWH = np.dot(_Ew[:,goodk], _Et[goodk,np.newaxis]*_Eh[goodk,:])
            # E_tWH_inv = np.dot(_Ew_inv_inv[:,goodk], _Et_inv_inv[goodk,np.newaxis]*_Eh_inv_inv[goodk,:])
            XX = self.X * E_tWH_inv**(-2)

            _rhoh = np.zeros_like(self.rhoh[c])
            _tauh = np.zeros_like(self.tauh[c])
            _rhoh[goodk,:] = self.bh + np.dot(_Et[goodk,np.newaxis]*_Ew[:,goodk].T, E_tWH**(-1))
            _tauh[goodk,:] = _Eh_inv_inv[goodk,:]**2 * np.dot(_Et_inv_inv[goodk,np.newaxis]*_Ew_inv_inv[:,goodk].T, XX)
            _tauh[_tauh < 1e-100] = 0.0
            self.rhoh[c] = _rhoh
            self.tauh[c] = _tauh

            # self.rhoh[c,goodk,:] = self.bh + np.dot(_Et[goodk,np.newaxis]*_Ew[:,goodk].T, E_tWH**(-1))
            # self.tauh[c,goodk,:] = _Eh_inv_inv[goodk,:]**2 * np.dot(_Et_inv_inv[goodk,np.newaxis]*_Ew_inv_inv[:,goodk].T, XX)
            # self.tauh[self.tauh < 1e-100] = 0

            # gig_H = GIG_cy(self.b, self.rhoh[goodk,:], self.tauh[goodk,:])
            self.gigH.update(self.ah, self.rhoh[c,goodk,:], self.tauh[c,goodk,:])
            self.Eh[c,goodk,:] = self.gigH.expectation()
            self.Eh_inv[c,goodk,:] = self.gigH.inv_expectation()
            self.Eh_inv_inv[c,goodk,:] = 1.0 / self.Eh_inv[c,goodk,:]
        
    def _updateTheta(self, goodk=None):
        """
        Update parameters of theta [rhot,taut]
        """
        # if goodk is None:
        #     goodk = self._goodK()
        
        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew = self.Ew[c]
            _Eh = self.Eh[c]
            _Et = self.Et[c]
            _Ew_inv_inv = self.Ew_inv_inv[c]
            _Eh_inv_inv = self.Eh_inv_inv[c]
            _Et_inv_inv = self.Et_inv_inv[c]

            E_tWH = self._E_tWH()
            E_tWH_inv = self._E_tWH_inv()
            # E_tWH = np.dot(_Ew[:,goodk], _Et[goodk,np.newaxis]*_Eh[goodk,:])
            # E_tWH_inv = np.dot(_Ew_inv_inv[:,goodk], _Et_inv_inv[goodk,np.newaxis]*_Eh_inv_inv[goodk,:])
            XX = self.X * E_tWH_inv**(-2)

            self.rhot[c,goodk] = self.alpha*self.c + (np.dot(_Ew[:,goodk].T, E_tWH**(-1)) * _Eh[goodk,:]).sum(1)
            self.taut[c,goodk] = _Et_inv_inv[goodk]**2 * (np.dot(_Ew_inv_inv[:,goodk].T, XX) * _Eh_inv_inv[goodk,:]).sum(1)
            self.taut[self.taut < 1e-100] = 0

            # gig_theta = GIG_cy(self.alpha/float(self.K), self.rhot[goodk], self.taut[goodk])
            self.gigT.update(self.alpha/float(self.K), self.rhot[c,goodk], self.taut[c,goodk])
            self.Et[c,goodk] = self.gigT.expectation()
            self.Et_inv[c,goodk] = self.gigT.inv_expectation()
            self.Et_inv_inv[c,goodk] = 1.0 / self.Et_inv[c,goodk]

    def getW(self, c, active_only=True):
        if active_only:
            goodk = self._goodK(c)
        else:
            goodk = np.arange(self.K)
        return self.Ew[c,:,goodk],self.Ew_inv[c,:,goodk]

    def getH(self, c, active_only=True):
        if active_only:
            goodk = self._goodK(c)
        else:
            goodk = np.arange(self.K)
        return self.Eh[c,goodk,:],self.Eh_inv[c,goodk,:]

    def getTheta(self, c, active_only=True):
        if active_only:
            goodk = self._goodK(c)
        else:
            goodk = np.arange(self.K)
        return self.Et[c,goodk],self.Et_inv[c,goodk]

    def reconstruct(self, Ew=None, Eh=None, Et=None):
        if Ew is None:
            Ew = self.Ew
        if Eh is None:
            Eh = self.Eh
        if Et is None:
            Et = self.Et

        XX = np.zeros_like(self.X)
        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            _Ew = Ew[c]
            _Et = Et[c]
            _Eh = Eh[c]
            XX += np.dot(_Ew[:,goodk], _Et[goodk,np.newaxis]*_Eh[goodk,:])

        return XX
        
    def KLdivergence(self):
        """
        入力スペクトログラムXと推論されたスペクトログラムtheta*W*HのKLダイバージェンス
        """
        # goodk = self._goodK()
        X = self.X
        # Y = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        Y = self.reconstruct()
        
        Xnorm = X / X.sum()
        Ynorm = Y / Y.sum()

        return ( Xnorm * ( np.log(Xnorm) - np.log(Ynorm) ) ).sum()

        # goodk = self._goodK()
        # E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        
        # XX = self.X * E_tWH_inv**(-2)
        # score = (XX * sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis] * self.Eh_inv_inv[goodk,:])).sum() - sp.log(sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:]).sum())

        # return score

    def _goodK(self, c, cutoff=None):
        if cutoff is None:
            cutoff = 1e-10 * self.X.max()

        cdef np.ndarray[double,ndim=1] powers = self.Et[c] * self.Ew[c].max(0) * self.Eh[c].max(1)
        sorted_powers = np.flipud(np.argsort(powers))
        idx = np.where(powers[sorted_powers] > cutoff * powers.max())[0]
        goodk = sorted_powers[:(idx[-1]+1)]
        if powers[goodk[-1]] < cutoff:
            goodk = np.delete(goodk,-1)

        goodk = np.sort(goodk)

        # Etが1を超えているindexは削除するようにする
        # too_large_k = np.where(self.Et > 1.0)[0]
        # goodk = np.array(list(set(goodk) - set(too_large_k)))

        return goodk

    def _clearBadK(self, supervised=False):
        for c in xrange(self.n_class):
            goodk = self._goodK(c)
            badk = np.setdiff1d(np.arange(self.K), goodk)
            if not supervised:
                self.rhow[c,:,badk] = self.bw
                self.tauw[c,:,badk] = 0.0
            self.rhoh[c,badk,:] = self.bh
            self.tauh[c,badk,:] = 0.0
            self._compute_expectations(supervised=supervised)
            self.Et[c,badk] = 0.0
