# -*- coding: utf-8 -*-

"""
gap_nmf.py

GaP-NMF(Gamma Process NMF)の実装．

M.D.Hoffman, ''Bayesian Nonparametric Matrix Factorization for Recorded Music''
をほぼそのまま実装
"""

import scipy as sp
from pylufia.stats import GIG_cy


class GaPNMF():
    def __init__(self, X, aw=0.1, bw=0.1, ah=0.1, bh=0.1, alpha=1.0, c=1.0, K=100, smoothness=100):
        self.X = X / X.mean()
        self.nF,self.nT = self.X.shape
        self.aw = aw
        self.bw = bw
        self.ah = ah
        self.bh = bh
        self.alpha = alpha
        self.c = c
        self.K = K
        self.smoothness = smoothness

        # sp.random.seed(98765)

        # initialize parameters
        # self.rhow = 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.nF,self.K))
        # self.tauw = 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.nF,self.K))
        # self.rhoh = 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.K,self.nT))
        # self.tauh = 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.K,self.nT))
        # self.rhot = self.K * 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.K,))
        # self.taut = 1./self.K * 10000 * sp.random.gamma(smoothness, 1./smoothness, size=(self.K,))

        # self.Ew = None
        # self.Ew_inv = None
        # self.Ew_inv_inv = None
        # self.Eh = None
        # self.Eh_inv = None
        # self.Eh_inv_inv = None
        # self.Et = None
        # self.Et_inv = None
        # self.Et_inv_inv = None
        # self._compute_expectations()
        self.initialize_parameters(supervised=False)

    def initialize_parameters(self, supervised=False):
        if not supervised:
            self.rhow = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.nF,self.K))
            self.tauw = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.nF,self.K))
        self.rhoh = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,self.nT))
        self.tauh = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,self.nT))
        self.rhot = self.K * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,))
        self.taut = 1./self.K * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.K,))

        if not supervised:
            self.Ew = None
            self.Ew_inv = None
            self.Ew_inv_inv = None
        self.Eh = None
        self.Eh_inv = None
        self.Eh_inv_inv = None
        self.Et = None
        self.Et_inv = None
        self.Et_inv_inv = None

        self.gigW = GIG_cy(self.aw, self.rhow, self.tauw)
        self.gigH = GIG_cy(self.ah, self.rhoh, self.tauh)
        self.gigT = GIG_cy(self.alpha/float(self.K), self.rhot, self.taut)

        self._compute_expectations(supervised=supervised)

    def infer(self, n_iter=100):
        """
        GaP-NMFパラメータの推論

        推論は変分ベイズ．
        """
        self.initialize_parameters(supervised=False)

        for it in range(n_iter):
            print( 'iterates: {0}'.format(it) )

            self._update_h()
            self._update_w()
            self._update_theta()

            D = self.KLdivergence()
            print( 'KLD={0}'.format(D) )

        self._clearBadK()

        W = self.Ew
        H = self.Eh
        theta = self.Et

        return W,H,theta

    def infer_activation(self, Ew, Ew_inv, n_iter=100, update_w=False):
        """
        Wは与えられたものを使い，Activation(Hとtheta)のみを推論する
        """
        self.Ew = Ew
        self.Ew_inv = Ew_inv
        self.Ew_inv_inv = 1.0 / self.Ew_inv
        self.K = self.Ew.shape[1]

        self.initialize_parameters(supervised=True)

        for it in range(n_iter):
            print( 'iterates: {0}'.format(it) )

            self._update_h(supervised=True)
            self._update_theta(supervised=True)
            # self._clearBadK(supervised=True)
            if update_w:
                self._update_w(goodk=goodk)

            D = self.KLdivergence()
            print( 'KLD={0}'.format(D) )

        H = self.Eh
        # theta = self.Et

        return H

    def _update_w(self):
        """
        Update parameters of W [rhow,tauw]
        """
        goodk = self._goodK()
        E_tWH = sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:])
        E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        XX = self.X * E_tWH_inv**(-2)
        
        self.rhow[:,goodk] = self.bw + sp.dot(E_tWH**(-1), self.Et[goodk] * self.Eh[goodk,:].T)
        self.tauw[:,goodk] = self.Ew_inv_inv[:,goodk]**2 * sp.dot(XX, self.Et_inv_inv[goodk] * self.Eh_inv_inv[goodk,:].T)
        self.tauw[self.tauw < 1e-100] = 0

        # gig_W = GIG(self.aw, self.rhow[:,goodk], self.tauw[:,goodk])
        self.gigW.update(self.aw, self.rhow[:,goodk], self.tauw[:,goodk])
        self.Ew[:,goodk] = self.gigW.expectation()
        self.Ew_inv[:,goodk] = self.gigW.inv_expectation()
        self.Ew_inv_inv[:,goodk] = 1.0 / self.Ew_inv[:,goodk]

    def _update_h(self, supervised=False):
        """
        Update parameters of H [rhoh,tauh]
        """
        if supervised:
            goodk = sp.arange(self.K)
        else:
            goodk = self._goodK()
        E_tWH = sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:])
        E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        XX = self.X * E_tWH_inv**(-2)

        self.rhoh[goodk,:] = self.bh + sp.dot(self.Et[goodk,sp.newaxis]*self.Ew[:,goodk].T, E_tWH**(-1))
        self.tauh[goodk,:] = self.Eh_inv_inv[goodk,:]**2 * sp.dot(self.Et_inv_inv[goodk,sp.newaxis]*self.Ew_inv_inv[:,goodk].T, XX)
        self.tauh[self.tauh < 1e-100] = 0

        # gig_H = GIG(self.ah, self.rhoh[goodk,:], self.tauh[goodk,:])
        self.gigH.update(self.ah, self.rhoh[goodk,:], self.tauh[goodk,:])
        self.Eh[goodk,:] = self.gigH.expectation()
        self.Eh_inv[goodk,:] = self.gigH.inv_expectation()
        self.Eh_inv_inv[goodk,:] = 1.0 / self.Eh_inv[goodk,:]
        
    def _update_theta(self, supervised=False):
        """
        Update parameters of theta [rhot,taut]
        """
        if supervised:
            goodk = sp.arange(self.K)
        else:
            goodk = self._goodK()

        E_tWH = sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:])
        E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        XX = self.X * E_tWH_inv**(-2)

        self.rhot[goodk] = self.alpha*self.c + (sp.dot(self.Ew[:,goodk].T, E_tWH**(-1)) * self.Eh[goodk,:]).sum(1)
        self.taut[goodk] = self.Et_inv_inv[goodk]**2 * (sp.dot(self.Ew_inv_inv[:,goodk].T, XX) * self.Eh_inv_inv[goodk,:]).sum(1)
        self.taut[self.taut < 1e-100] = 0

        # gig_theta = GIG(self.alpha/float(self.K), self.rhot[goodk], self.taut[goodk])
        self.gigT.update(self.alpha/float(self.K), self.rhot[goodk], self.taut[goodk])
        self.Et[goodk] = self.gigT.expectation()
        self.Et_inv[goodk] = self.gigT.inv_expectation()
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

        return sp.dot(Ew[:,goodk], Et[goodk,sp.newaxis]*Eh[goodk,:])
        
    def KLdivergence(self):
        """
        入力スペクトログラムXと推論されたスペクトログラムtheta*W*HのKLダイバージェンス
        """
        goodk = self._goodK()
        X = self.X
        Y = sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:])
        return (X*(sp.log(X)-sp.log(Y)) + (Y-X)).sum()

        # goodk = self._goodK()
        # E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        
        # XX = self.X * E_tWH_inv**(-2)
        # score = (XX * sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis] * self.Eh_inv_inv[goodk,:])).sum() - sp.log(sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:]).sum())

        # return score

    def _compute_expectations(self, supervised=False):
        """
        W,H,thetaの期待値E[y]とE[1/y]を計算する
        """
        if not supervised:
            # gig_W = GIG(self.a, self.rhow, self.tauw)
            self.gigW.update(self.aw, self.rhow, self.tauw)
            self.Ew = self.gigW.expectation()
            self.Ew_inv = self.gigW.inv_expectation()
            self.Ew_inv_inv = 1.0 / self.Ew_inv

        # gig_H = GIG(self.b, self.rhoh, self.tauh)
        self.gigH.update(self.ah, self.rhoh, self.tauh)
        self.Eh = self.gigH.expectation()
        self.Eh_inv = self.gigH.inv_expectation()
        self.Eh_inv_inv = 1.0 / self.Eh_inv
        
        # gig_theta = GIG(self.alpha/float(self.K), self.rhot, self.taut)
        self.gigT.update(self.alpha/float(self.K), self.rhot, self.taut)
        self.Et = self.gigT.expectation()
        self.Et_inv = self.gigT.inv_expectation()
        self.Et_inv_inv = 1.0 / self.Et_inv
        
    def _goodK(self, cutoff=None):
        if cutoff is None:
            cutoff = 1e-10 * self.X.max()

        powers = self.Et * self.Ew.max(0) * self.Eh.max(1)
        sorted_powers = sp.flipud(sp.argsort(powers))
        
        idx = sp.where(powers[sorted_powers] > cutoff * powers.max())[0]
        goodk = sorted_powers[:(idx[-1]+1)]
        if powers[goodk[-1]] < cutoff:
            goodk = sp.delete(goodk,-1)

        goodk = sp.sort(goodk)

        # Etが1を超えているindexは削除するようにする
        too_large_k = sp.where(self.Et > 1.0)[0]
        goodk = list(set(goodk) - set(too_large_k))

        return goodk

    def _clearBadK(self, supervised=False):
        goodk = self._goodK()
        badk = sp.setdiff1d(sp.arange(self.K), goodk)
        if not supervised:
            self.rhow[:,badk] = self.a
            self.tauw[:,badk] = 0.0
        self.rhoh[badk,:] = self.b
        self.tauh[badk,:] = 0.0
        self._compute_expectations(supervised=supervised)
        self.Et[badk] = 0.0