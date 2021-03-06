# -*- coding: utf-8 -*-

"""
GaPNMF cython implementation
"""

cimport cython
import numpy as np
cimport numpy as np
from pylufia.stats import GIG_cy


class GapNmf_cy():
    def __init__(self, X, aw=0.1, bw=0.1, ah=0.1, bh=0.1, alpha=1.0, c=1.0, K=100, smoothness=100, criterion=0.0001):
        self.X = X / X.mean()
        self.nF,self.nT = self.X.shape

        self.K = K
        self.smoothness = smoothness

        self.aw_val = aw
        self.bw_val = bw
        self.ah_val = ah
        self.bh_val = bh
        self.alpha_val = alpha
        self.c_val = c
        self.criterion = criterion
        self._convert_hparam_to_matrix(aw, bw, ah, bh, alpha, c, None, None)

        self.scaling = 10000

        #np.random.seed(98765) #debug

        # initialize parameters
        self.rhow = np.zeros((self.nF,self.K), dtype=np.double)
        self.tauw = np.zeros((self.nF,self.K), dtype=np.double)
        self.rhoh = np.zeros((self.K,self.nT), dtype=np.double)
        self.tauh = np.zeros((self.K,self.nT), dtype=np.double)
        self.rhot = np.zeros(self.K, dtype=np.double)
        self.taut = np.zeros(self.K, dtype=np.double)
        self.Ew = np.zeros((self.nF,self.K), dtype=np.double)
        self.Ew_inv = np.zeros((self.nF,self.K), dtype=np.double)
        self.Ew_inv_inv = np.zeros((self.nF,self.K), dtype=np.double)
        self.Eh = np.zeros((self.K,self.nT), dtype=np.double)
        self.Eh_inv = np.zeros((self.K,self.nT), dtype=np.double)
        self.Eh_inv_inv = np.zeros((self.K,self.nT), dtype=np.double)
        self.Et = np.zeros(self.K, dtype=np.double)
        self.Et_inv = np.zeros(self.K, dtype=np.double)
        self.Et_inv_inv = np.zeros(self.K, dtype=np.double)
        self.gigW = None
        self.gigH = None
        self.gigT = None

    def _init_parameters(self, supervised=False):
        """
        潜在変数の初期化
        """
        self.rhow = self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.nF,self.K))
        self.tauw = self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.nF,self.K))
        self.rhoh = self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,self.nT))
        self.tauh = self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,self.nT))
        self.rhot = self.K * self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,))
        self.taut = 1./self.K * self.scaling * np.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,))

         #if not supervised:
         #    self.rhow = np.random.gamma(100, 1/1000., size=(self.nF,self.K))
         #    self.tauw[:] = 0.1
         #self.rhoh = np.random.gamma(100, 1/1000., size=(self.K,self.nT))
         #self.tauh[:] = 0.1
         #self.rhot = np.random.gamma(100, 1/1000., size=(self.K,))
         #self.taut[:] = 0.1

        self.gigW = GIG_cy(self.aw, self.rhow, self.tauw)
        self.gigH = GIG_cy(self.ah, self.rhoh, self.tauh)
        self.gigT = GIG_cy(self.alpha/float(self.K), self.rhot, self.taut)

        self._compute_expectations(supervised=supervised)

    def _init_parameters_fixed(self, supervised=False):
        """
        潜在変数の初期値をすべて1*scalingに固定するバージョン
        """
        if not supervised:
            self.rhow = self.scaling * np.ones( (self.nF,self.K) )
            self.tauw = self.scaling * np.ones( (self.nF,self.K) )
        self.rhoh = self.scaling * np.ones( (self.K,self.nT) )
        self.tauh = self.scaling * np.ones( (self.K,self.nT) )
        self.rhot = self.K * self.scaling * np.ones(self.K)
        self.taut = 1./self.K * self.scaling * np.ones(self.K)

        self.gigW = GIG_cy(self.aw, self.rhow, self.tauw)
        self.gigH = GIG_cy(self.ah, self.rhoh, self.tauh)
        self.gigT = GIG_cy(self.alpha/float(self.K), self.rhot, self.taut)

        self._compute_expectations(supervised=supervised)

    def load_init_parameters(self, rhow, tauw, rhoh, tauh, rhot, taut):
        self.rhow = rhow
        self.tauw = tauw
        self.rhoh = rhoh
        self.tauh = tauh
        self.rhot = rhot
        self.taut = taut

    def _compute_expectations(self, supervised=False):
        """
        W,H,thetaの期待値E[y]とE[1/y]を計算する
        """
        if not supervised:
            # gig_W = GIG_cy(self.a, self.rhow, self.tauw)
            self.gigW.update(self.aw, self.rhow, self.tauw)
            #self.Ew = self.gigW.expectation()
            #self.Ew_inv = self.gigW.inv_expectation()
            self.Ew,self.Ew_inv = self.gigW.expectation_both()
            self.Ew_inv_inv = 1.0 / self.Ew_inv

        # gig_theta = GIG_cy(self.alpha/float(self.K), self.rhot, self.taut)
        self.gigT.update(self.alpha/float(self.K), self.rhot, self.taut)
        #self.Et = self.gigT.expectation()
        #self.Et_inv = self.gigT.inv_expectation()
        self.Et,self.Et_inv = self.gigT.expectation_both()
        self.Et_inv_inv = 1.0 / self.Et_inv

        # gig_H = GIG_cy(self.b, self.rhoh, self.tauh)
        self.gigH.update(self.ah, self.rhoh, self.tauh)
        #self.Eh = self.gigH.expectation()
        #self.Eh_inv = self.gigH.inv_expectation()
        self.Eh,self.Eh_inv = self.gigH.expectation_both()
        self.Eh_inv_inv = 1.0 / self.Eh_inv

    def _convert_hparam_to_matrix(self, aw=0.1, bw=0.1, ah=0.1, bh=0.1, alpha=1.0, c=1.0, ahx=None, emp_index=None):
        """
        数値で与えられたhyperparamaterを行列表現に変換する．
        行列表現にするのは一部の基底のみ事前分布の値を変えてその基底のアクティベーションを立ちやすくする，
        といった調整を可能にするため．
        """
        self.aw = np.zeros( (self.nF,self.K) )
        self.aw[:] = aw
        self.bw = np.zeros( (self.nF,self.K) )
        self.bw[:] = bw
        self.ah = np.zeros( (self.K,self.nT) )
        if ahx is not None:
            self.ah[:] = ahx
            self.ah[emp_index,:] = ah
        else:
            self.ah[:] = ah
        self.bh = np.zeros( (self.K,self.nT) )
        self.bh[:] = bh
        self.alpha = np.zeros(self.K)
        self.alpha[:] = alpha
        self.c = np.zeros(self.K)
        self.c[:] = c

    def infer(self, n_iter=100, show_prompt=False):
        """
        GaP-NMFパラメータの推論

        推論は変分ベイズ．
        """
        self._convert_hparam_to_matrix(self.aw, self.bw, self.ah, self.bh, self.alpha, self.c, None, None)
        self._init_parameters(supervised=False)

        cdef int it = 0
        cdef double score = -np.inf
        cdef double last_score = -np.inf
        for it from 0 <= it < n_iter:
            self._updateH()
            self._updateW()
            self._updateTheta()
            self._clearBadK()

            last_score = score
            score = self.log_likelihood()
            improvement = (score - last_score) / np.abs(last_score)
            if show_prompt:
                print( 'iterates: {} log likelihood={:.2f} ({:.5f} improvement)'.format(it, score, improvement) )

            if it > 20 and improvement < self.criterion:
                break

        W = self.Ew
        H = self.Eh
        theta = self.Et

        return W,H,theta

    def infer_activation(self, Ew, Ew_inv, n_iter=100, update_w=False, show_prompt=False, hparams=None):
        """
        W(とtheta)は与えられたものを使い，Hのみを推論する
        """
        self.Ew = Ew
        self.Ew_inv = Ew_inv
        self.Ew_inv_inv = 1.0 / self.Ew_inv
        self.K = self.Ew.shape[1]

        if hparams:
            ah = hparams['ah']
            ahx = hparams['ahx']
            emp_index = hparams['emp_index']
        else:
            ah = self.ah
            ahx = None
            emp_index = None

        self._convert_hparam_to_matrix(self.aw, self.bw, ah, self.bh, self.alpha, self.c, ahx, emp_index)
        # self._init_parameters(supervised=True)
        self._init_parameters_fixed(supervised=True)

        cdef int it
        cdef np.ndarray[int,ndim=1] goodk = np.arange(self.K, dtype=np.int32)
        cdef double score = -np.inf
        cdef double last_score = -np.inf
        for it from 0 <= it < n_iter:
            self._updateH(goodk=goodk)
            self._updateTheta(goodk=goodk)
            if update_w:
                self._updateW(goodk=goodk)

            last_score = score
            score = self.log_likelihood()
            improvement = (score - last_score) / np.abs(last_score)
            if show_prompt:
                print( 'iterates: {} log likelihood={:.2f} ({:.5f} improvement)'.format(it, score, improvement) )

            if it > 20 and improvement < self.criterion:
                break

        # self._clearBadK(supervised=True)

        return self.Eh,self.Et

    def infer_activation_with_free_basis(self, Ew, Ew_inv, Et=None, Et_inv=None, ahx=10, ahu=0.01, n_iter=100, n_free_basis=10, update_w=False, show_prompt=False):
        """
        Wは与えられたものを使い，Activation(HとTheta)のみを推論する
        ただしフリーな基底をいくつか用意し，どの教師基底にも属さない成分を吸収する
        """
        self.orig_K = Ew.shape[1]
        self.K = self.orig_K + n_free_basis
        self._convert_hparam_to_matrix(self.aw_val, self.bw_val, self.ah_val, self.bh_val, self.alpha_val, self.c_val, None, None)
        self.ah = np.zeros( (self.K,self.Eh.shape[1]), dtype=np.double )
        self.ah[:Ew.shape[1],:] = ahx
        self.ah[Ew.shape[1]:,:] = ahu

        #self._init_parameters(supervised=False)
        self._init_parameters_fixed(supervised=False)
        self.Ew[:,:Ew.shape[1]] = Ew
        self.Ew_inv[:,:Ew_inv.shape[1]] = Ew_inv
        self.Ew_inv_inv = 1.0 / self.Ew_inv

        cdef int it
        cdef np.ndarray[int,ndim=1] goodk = np.arange(self.K, dtype=np.int32)
        cdef double score = -np.inf
        cdef double last_score = -np.inf
        for it from 0 <= it < n_iter:
            self._updateH(goodk=goodk)
            self._updateTheta(goodk=goodk)
            self._updateW_withFreeBasis(Ew, Ew_inv, goodk=goodk)

            last_score = score
            score = self.log_likelihood()
            improvement = (score - last_score) / np.abs(last_score)
            if show_prompt:
                print( 'iterates: {} log likelihood={:.2f} ({:.5f} improvement)'.format(it, score, improvement) )

            if it > 20 and improvement < self.criterion:
                break

        # self._clearBadK(supervised=True)

        Eh = self.Eh[:self.orig_K,:]
        Et = self.Et[:self.orig_K]

        return Eh,Et

    def _updateW(self, goodk=None):
        """
        Update parameters of W [rhow,tauw]
        """
        if goodk is None:
            goodk = self._goodK()
        cdef np.ndarray[double,ndim=2] E_tWH = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        cdef np.ndarray[double,ndim=2] E_tWH_inv = np.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,np.newaxis]*self.Eh_inv_inv[goodk,:])
        cdef np.ndarray[double,ndim=2] XX = self.X * (E_tWH_inv+1e-10)**(-2)
        
        self.rhow[:,goodk] = self.bw[:,goodk] + np.dot( (E_tWH+1e-10)**(-1), self.Et[goodk] * self.Eh[goodk,:].T)
        self.tauw[:,goodk] = self.Ew_inv_inv[:,goodk]**2 * np.dot(XX, self.Et_inv_inv[goodk] * self.Eh_inv_inv[goodk,:].T)
        self.tauw[self.tauw < 1e-100] = 0

        # gig_W = GIG_cy(self.a, self.rhow[:,goodk], self.tauw[:,goodk])
        self.gigW.update(self.aw[:,goodk], self.rhow[:,goodk], self.tauw[:,goodk])
        #self.Ew[:,goodk] = self.gigW.expectation()
        #self.Ew_inv[:,goodk] = self.gigW.inv_expectation()
        self.Ew[:,goodk],self.Ew_inv[:,goodk] = self.gigW.expectation_both()
        self.Ew_inv_inv[:,goodk] = 1.0 / self.Ew_inv[:,goodk]

    def _updateW_withFreeBasis(self, Ew, Ew_inv, goodk=None):
        #self._updateW(goodk=goodk)
        #self.Ew[:,:Ew.shape[1]] = Ew
        #self.Ew_inv[:,:Ew_inv.shape[1]] = Ew_inv
        #self.Ew_inv_inv = 1.0 / self.Ew_inv

        if goodk is None:
            goodk = self._goodK()
        cdef np.ndarray[double,ndim=2] E_tWH = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        cdef np.ndarray[double,ndim=2] E_tWH_inv = np.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,np.newaxis]*self.Eh_inv_inv[goodk,:])
        cdef np.ndarray[double,ndim=2] XX = self.X * (E_tWH_inv+1e-10)**(-2)
        
        self.rhow[:,goodk] = self.bw[:,goodk] + np.dot( (E_tWH+1e-10)**(-1), self.Et[goodk] * self.Eh[goodk,:].T)
        self.tauw[:,goodk] = self.Ew_inv_inv[:,goodk]**2 * np.dot(XX, self.Et_inv_inv[goodk] * self.Eh_inv_inv[goodk,:].T)
        self.tauw[self.tauw < 1e-100] = 0

        update_K = np.arange(Ew.shape[1]+1, self.Ew.shape[1])
        self.gigW.update(self.aw[:,update_K], self.rhow[:,update_K], self.tauw[:,update_K])
        #self.Ew[:,update_K] = self.gigW.expectation()
        #self.Ew_inv[:,update_K] = self.gigW.inv_expectation()
        self.Ew[:,update_K],self.Ew_inv[:,update_K] = self.gigW.expectation_both()
        self.Ew_inv_inv[:,update_K] = 1.0 / self.Ew_inv[:,update_K]

    def _updateH(self, goodk=None):
        """
        Update parameters of H [rhoh,tauh]
        """
        if goodk is None:
            goodk = self._goodK()
        cdef np.ndarray[double,ndim=2] E_tWH = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        cdef np.ndarray[double,ndim=2] E_tWH_inv = np.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,np.newaxis]*self.Eh_inv_inv[goodk,:])
        cdef np.ndarray[double,ndim=2] XX = self.X * (E_tWH_inv + 1e-10)**(-2)

        self.rhoh[goodk,:] = self.bh[goodk,:] + np.dot(self.Et[goodk,np.newaxis]*self.Ew[:,goodk].T, E_tWH**(-1))
        self.tauh[goodk,:] = self.Eh_inv_inv[goodk,:]**2 * np.dot(self.Et_inv_inv[goodk,np.newaxis]*self.Ew_inv_inv[:,goodk].T, XX)
        self.tauh[self.tauh < 1e-100] = 0

        # gig_H = GIG_cy(self.b, self.rhoh[goodk,:], self.tauh[goodk,:])
        self.gigH.update(self.ah[goodk,:], self.rhoh[goodk,:], self.tauh[goodk,:])
        #self.Eh[goodk,:] = self.gigH.expectation()
        #self.Eh_inv[goodk,:] = self.gigH.inv_expectation()
        self.Eh[goodk,:],self.Eh_inv[goodk,:] = self.gigH.expectation_both()
        self.Eh_inv_inv[goodk,:] = 1.0 / self.Eh_inv[goodk,:]
        
    def _updateTheta(self, goodk=None):
        """
        Update parameters of theta [rhot,taut]
        """
        if goodk is None:
            goodk = self._goodK()
        cdef np.ndarray[double,ndim=2] E_tWH = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        cdef np.ndarray[double,ndim=2] E_tWH_inv = np.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,np.newaxis]*self.Eh_inv_inv[goodk,:])
        cdef np.ndarray[double,ndim=2] XX = self.X * (E_tWH_inv + 1e-10)**(-2)
        # E_tWH_invに0が含まれるとzero division errorになる

        self.rhot[goodk] = self.alpha[goodk]*self.c[goodk] + (np.dot(self.Ew[:,goodk].T, (E_tWH+1e-10)**(-1)) * self.Eh[goodk,:]).sum(1)
        self.taut[goodk] = self.Et_inv_inv[goodk]**2 * (np.dot(self.Ew_inv_inv[:,goodk].T, XX) * self.Eh_inv_inv[goodk,:]).sum(1)
        self.taut[self.taut < 1e-100] = 0

        # gig_theta = GIG_cy(self.alpha/float(self.K), self.rhot[goodk], self.taut[goodk])
        self.gigT.update(self.alpha[goodk]/float(self.K), self.rhot[goodk], self.taut[goodk])
        #self.Et[goodk] = self.gigT.expectation()
        #self.Et_inv[goodk] = self.gigT.inv_expectation()
        self.Et[goodk],self.Et_inv[goodk] = self.gigT.expectation_both()
        self.Et_inv_inv[goodk] = 1.0 / self.Et_inv[goodk]

    def getW(self, active_only=True):
        if active_only:
            goodk = self._goodK()
        else:
            goodk = np.arange(self.K)
        return self.Ew[:,goodk],self.Ew_inv[:,goodk]

    def getH(self, active_only=True):
        if active_only:
            goodk = self._goodK()
        else:
            goodk = np.arange(self.K)
        return self.Eh[goodk,:],self.Eh_inv[goodk,:]

    def getTheta(self, active_only=True):
        if active_only:
            goodk = self._goodK()
        else:
            goodk = np.arange(self.K)
        return self.Et[goodk],self.Et_inv[goodk]

    def reconstruct(self, Ew=None, Eh=None, Et=None):
        goodk = self._goodK()
        if Ew is None:
            Ew = self.Ew
        if Eh is None:
            Eh = self.Eh
        if Et is None:
            Et = self.Et

        return np.dot(Ew[:,goodk], Et[goodk,np.newaxis]*Eh[goodk,:])
        
    def KLdivergence(self):
        """
        入力スペクトログラムXと推論されたスペクトログラムtheta*W*HのKLダイバージェンス
        """
        goodk = self._goodK()
        X = self.X
        Y = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])

        # return ( Y * ( np.log(Y) - np.log(X) ) + (X-Y) ).sum()
        return ( Y * ( np.log(Y+1e-10) - np.log(X+1e-10) ) ).sum()

    def log_likelihood(self):
        score = 0.0
        #goodk = self._goodK()
        goodk = np.arange(self.K, dtype=np.int32)
        E_tWH = np.dot(self.Ew[:,goodk], self.Et[goodk,np.newaxis]*self.Eh[goodk,:])
        E_tWH_inv = np.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,np.newaxis]*self.Eh_inv_inv[goodk,:])

        score -= ( self.X / E_tWH_inv + np.log(E_tWH) ).sum()
        gig = GIG_cy(self.aw, self.rhow, self.tauw)
        score += gig.gamma_term(self.Ew, self.Ew_inv, self.rhow, self.tauw, self.aw_val, self.bw_val)
        score += gig.gamma_term(self.Eh, self.Eh_inv, self.rhoh, self.tauh, self.ah_val, self.bh_val)
        score += gig.gamma_term(self.Et, self.Et_inv, self.rhot, self.taut, self.alpha_val/self.K, self.alpha_val)

        return score
        
    def _goodK(self, cutoff=None):
        if cutoff is None:
            cutoff = 1e-10 * self.X.max()

        cdef np.ndarray[double,ndim=1] powers = self.Et * self.Ew.max(0) * self.Eh.max(1)
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
        goodk = self._goodK()
        badk = np.setdiff1d(np.arange(self.K), goodk)
        if not supervised:
            self.rhow[:,badk] = self.bw[:,badk]
            self.tauw[:,badk] = 0.0
        self.rhoh[badk,:] = self.bh[badk,:]
        self.tauh[badk,:] = 0.0
        self._compute_expectations(supervised=supervised)
        self.Et[badk] = 0.0
