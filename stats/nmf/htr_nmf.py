# -*- coding: utf-8 -*-

"""
HTRNMF(HierarchicalTimbreRhythm NMF)

音色カテゴリ,リズムパターンを同時に推定するノンパラベイズNMFの実装
"""

import scipy as sp
from pylufia.stats import Gamma_cy

import itertools

import time

class HTRNMF():
    """
    _update*_bruteの方だと正しく動いてそうだが，
    _update*の方はバグがありそう．array演算の次元まわりか？
    """
    def __init__(self, X, params):
        self.X = X / X.mean()
        self.nF,self.nT = self.X.shape
        self.ah = params['ah']
        self.bh = params['bh']
        self.aw = params['aw']
        self.bw = params['bw']
        self.alphag = params['alphag']
        self.cg = params['cg']
        self.Kg = params['Kg']
        self.alphat = params['alphat']
        self.ct = params['ct']
        self.Kt = params['Kt']
        self.smoothness = params['smoothness']

        sc = 10000.0
        self.rhow = sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.nF,self.Kt))
        self.tauw = sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.nF,self.Kt))
        self.rhoh = sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt,self.nT))
        self.tauh = sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt,self.nT))
        self.rhot = self.ct * sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt))
        self.taut = 1./self.ct * sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt))
        self.rhog = self.cg * sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg))
        self.taug = self.cg * sc * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg))

        self.Ew = None
        self.Eh = None
        self.Et = None
        self.Eg = None
        self._compute_expectations()

        self.M = sp.zeros( (self.Kg,self.Kt,self.nT,self.nF) )

    def infer(self, n_iter=100):
        for it in range(n_iter):
            print( 'iterates: {0}'.format(it) )

            self._updateM()
            self._updateW()
            self._updateH()
            self._updateTheta()
            self._updateG()
            
            # self._updateM_brute()
            # self._updateW_brute()
            # self._updateH_brute()
            # self._updateTheta_brute()
            # self._updateG_brute()

            D = self.KLDivergence()
            print( 'KLD={0}'.format(D) )

        return self.Ew,self.Eh,self.Et,self.Eg

    def infer_supervised(self, Ew, n_iter=100):
        self.Ew[:Ew.shape[0]] = Ew

        for it in range(n_iter):
            print( 'iterates: {0}'.format(it) )

            self._updateM()
            self._updateH()
            self._updateW_supervised(Ew)
            self._updateTheta()
            self._updateG()

            D = self.KLDivergence()
            print( 'KLD={0}'.format(D) )

        return self.Ew,self.Eh,self.Et,self.Eg

    def _updateM(self):
        # Ew = sp.transpose(self.Ew, (0,2,1)) #Ew[g,j,f]に
        # self.M = self.Eg[:,sp.newaxis,sp.newaxis,sp.newaxis] * self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]
        for g in range(self.Kg):
            Eht = self.Eh[g] * self.Et[g,:,sp.newaxis]
            Ew_T = self.Ew[g].T
            self.M[g,:,:,:] = self.Eg[g] * Eht[:,:,sp.newaxis] * Ew_T[:,sp.newaxis,:]

        # M.sum( (0,1) ) = X になるようにスケーリング
        # ちゃんとスケーリング係数求めるようにしたほうがよいかも
        re_X = self.M.sum( (0,1) ).T
        scale_M = (self.X / re_X).T[sp.newaxis,sp.newaxis,:,:]
        self.M *= scale_M

    def _updateM_brute(self):
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.M[g,j,t,f] = self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.M[g,j,t,f] = self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]

        ## M.sum((0,1)) = X となるようにスケーリング
        re_X = self.M.sum((0,1)).T
        # scale_M = self.X.sum() / re_X.sum() 
        scale_M = (self.X / re_X).T[sp.newaxis,sp.newaxis,:,:]
        self.M *= scale_M

        # print 'M:'
        # print self.M

    def _updateW(self):
        self.rhow = self.aw + self.M.sum(2)
        self.rhow = sp.transpose(self.rhow, (0,2,1))
        self.tauw[:,:,:] = (self.bw + (self.Eg[:,sp.newaxis,sp.newaxis] * self.Eh * self.Et[:,:,sp.newaxis]).sum(2))[:,sp.newaxis,:]

        self.Ew = Gamma_cy(self.rhow, self.tauw).expectation()

        # print 'rhow:'
        # print self.rhow
        # print 'tauw:'
        # print self.tauw
        # print 'Ew:'
        # print self.Ew

    def _updateW_supervised(self, Ew):
        self.rhow = self.aw + self.M.sum(2)
        self.rhow = sp.transpose(self.rhow, (0,2,1))
        self.tauw[:,:,:] = (self.bw + (self.Eg[:,sp.newaxis,sp.newaxis] * self.Eh * self.Et[:,:,sp.newaxis]).sum(2))[:,sp.newaxis,:]

        self.Ew = Gamma_cy(self.rhow, self.tauw).expectation()
        self.Ew[:Ew.shape[0]] = Ew

    def _updateW_brute(self):
        self.rhow[:,:,:] = self.aw
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.rhow[g,f,j] += self.M[g,j,t,f]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.rhow[g,f,j] += self.M[g,j,t,f]

        self.tauw[:,:,:] = self.bw
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.tauw[g,f,j] += self.Eg[g] * self.Eh[g,j,t] * self.Et[g,j]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.tauw[g,f,j] += self.Eg[g] * self.Eh[g,j,t] * self.Et[g,j]

        self.Ew = Gamma_cy(self.rhow, self.tauw).expectation()

        # print 'Ew:'
        # print self.Ew

    def _updateH(self):
        self.rhoh = self.ah + self.M.sum(3)
        self.tauh[:,:,:] = (self.bh + (self.Eg[:,sp.newaxis,sp.newaxis] * self.Ew * self.Et[:,sp.newaxis,:]).sum(1))[:,:,sp.newaxis]
        self.Eh = Gamma_cy(self.rhoh, self.tauh).expectation()

        # print 'Eh:'
        # print self.Eh

    def _updateH_brute(self):
        self.rhoh[:,:,:] = self.ah
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
                        # self.rhoh[g,j,t] += self.M[g,j,t,f]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.rhoh[g,j,t] += self.M[g,j,t,f]

        self.tauh[:,:,:] = self.bh
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.tauh[g,j,t] += self.Eg[g] * self.Ew[g,f,j] * self.Et[g,j]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.tauh[g,j,t] += self.Eg[g] * self.Ew[g,f,j] * self.Et[g,j]

        self.Eh = Gamma_cy(self.rhoh, self.tauh).expectation()

        # print 'Eh:'
        # print self.Eh

    def _updateTheta(self):
        # Ew = sp.transpose(self.Ew, (0,2,1))

        # self.rhot = self.alphat + self.M.sum((2,3))
        # self.taut = self.alphat / self.ct + (self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]).sum((2,3))
        self.rhot = self.alphat + self.M.sum( (2,3) )

        for g in range(self.Kg):
            Eh_g = self.Eh[g]
            Ew_g = self.Ew[g].T
            self.taut[g,:] = self.alphat / self.ct + (Eh_g[:,:,sp.newaxis] * Ew_g[:,sp.newaxis,:]).sum((1,2))
            # self.taut[g,:] = (self.Eg[g] * Eh_g[:,:,sp.newaxis] * Ew_g[:,sp.newaxis,:]).sum((1,2))

        self.Et = Gamma_cy(self.rhot, self.taut).expectation()

        # print 'Et:'
        # print self.Et

    def _updateTheta_brute(self):
        self.rhot[:,:] = self.alphat
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
                        # self.rhot[g,j] += self.M[g,j,t,f]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.rhot[g,j] += self.M[g,j,t,f]

        self.taut[:,:] = self.alphat / self.ct
        # self.taut[:,:] = self.alphat / self.Kt
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.taut[g,j] += self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.taut[g,j] += self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j]

        self.Et = Gamma_cy(self.rhot, self.taut).expectation()

        # print 'Et:'
        # print self.Et

    def _updateG(self):
        Ew = sp.transpose( self.Ew, (0,2,1) )
        self.rhog = self.alphag + self.M.sum( (1,2,3) )
        # self.taug = self.alphag / self.cg + (self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]).sum((1,2,3))
        for g in range(self.Kg):
            self.taug[g] = self.alphag / self.cg + (self.Eh[g,:,:,sp.newaxis] * Ew[g,:,sp.newaxis,:] * self.Et[g,:,sp.newaxis,sp.newaxis]).sum()
        self.Eg = Gamma_cy(self.rhog, self.taug).expectation()

        print 'Eg:'
        print self.Eg

    def _updateG_brute(self):
        self.rhog[:] = self.alphag
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.rhog[g] += self.M[g,j,t,f]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.rhog[g] += self.M[g,j,t,f]

        self.taug[:] = self.alphag / self.cg
        # self.taug[:] = self.alphag / self.Kg
        # for g in xrange(self.Kg):
        #     for j in xrange(self.Kt):
        #         for t in xrange(self.nT):
        #             for f in xrange(self.nF):
        #                 self.taug[g] += self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]
        for g,j,t,f in itertools.product(range(self.Kg), range(self.Kt), range(self.nT), range(self.nF)):
            self.taug[g] += self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]

        self.Eg = Gamma_cy(self.rhog, self.taug).expectation()

        print 'Eg:'
        print self.Eg

    def _compute_expectations(self):
        self.Ew = Gamma_cy(self.rhow, self.tauw).expectation()
        self.Eh = Gamma_cy(self.rhoh, self.tauh).expectation()
        self.Et = Gamma_cy(self.rhot, self.taut).expectation()
        self.Eg = Gamma_cy(self.rhog, self.taug).expectation()

    def reconstruct(self):
        Ex = sp.zeros((self.nF,self.nT))

        for g in range(self.Kg):
            Exg = sp.dot(self.Ew[g], self.Et[g,:,sp.newaxis]*self.Eh[g])
            Ex += Exg

        # self._updateM_brute()
        # Ex = self.M.sum((0,1)).T

        return Ex

    def KLDivergence(self):
        """
        入力スペクトログラムXと推論されたスペクトログラムtheta*W*HのKLダイバージェンス
        """
        X = self.X
        Y = self.reconstruct()
        return (X*(sp.log(X+1e-10)-sp.log(Y+1e-10)) + (Y-X)).sum()

        # goodk = self._goodK()
        # E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        
        # XX = self.X * E_tWH_inv**(-2)
        # score = (XX * sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis] * self.Eh_inv_inv[goodk,:])).sum() - sp.log(sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:]).sum())

        # return score
