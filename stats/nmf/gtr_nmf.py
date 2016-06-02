# -*- coding: utf-8 -*-

"""
@file gtr_nmf.py
@brief GTRNMF(GenreTimbreRhythm NMF)
@author ふぇいと (@stfate)

@description
ジャンル,音色,リズムパターンを同時に推定するノンパラベイズNMFの実装
"""

import scipy as sp
from pylufia.stats import Gamma

import time


class GTRNMF():
    def __init__(self, X, options):
        self.X = X / X.mean()
        self.nF,self.nT = self.X.shape
        self.ah = options['ah']
        self.bh = options['bh']
        self.aw = options['aw']
        self.bw = options['bw']
        self.alphag = options['alphag']
        self.cg = options['cg']
        self.Kg = options['Kg']
        self.alphat = options['alphat']
        self.ct = options['ct']
        self.Kt = options['Kt']
        self.smoothness = options['smoothness']

        sp.random.seed(98765)

        self.rhow = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.nF,self.Kt))
        self.tauw = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.nF,self.Kt))
        self.rhoh = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt,self.nT))
        self.tauh = 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt,self.nT))
        self.rhot = self.ct * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt))
        self.taut = 1./self.ct * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg,self.Kt))
        self.rhog = self.cg * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg))
        self.taug = self.cg * 10000 * sp.random.gamma(self.smoothness, 1./self.smoothness, size=(self.Kg))

        self.Ew = None
        self.Eh = None
        self.Et = None
        self.Eg = None
        self._compute_expectations()

        self.M = sp.zeros((self.Kg,self.Kt,self.nT,self.nF))

    def infer(self, n_iter=100):
        for it in xrange(n_iter):
            print 'iterates: {0}'.format(it)

            self._updateM()
            self._updateW()
            self._updateH()
            self._updateTheta()
            self._updateG()
            
            # self._updateM()
            # self._updateW_brute()
            # self._updateH_brute()
            # self._updateTheta_brute()
            # self._updateG_brute()

            D = self.KLDivergence()
            print 'KLD={0}'.format(D)

        return self.Ew,self.Eh,self.Et,self.Eg

    def _updateM(self):
        # Ew = sp.transpose(self.Ew, (0,2,1)) #Ew[g,j,f]に
        # self.M = self.Eg[:,sp.newaxis,sp.newaxis,sp.newaxis] * self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]
        for g in xrange(self.Kg):
            Eht = self.Eh[g] * self.Et[g,:,sp.newaxis]
            Ew_T = self.Ew[g].T
            self.M[g,:,:,:] = self.Eg[g] * Eht[:,:,sp.newaxis] * Ew_T[:,sp.newaxis,:]

        re_X = self.M.sum((0,1)).T
        scale_M = (self.X / re_X).mean()
        # scale_M = (self.X / re_X).T[sp.newaxis,sp.newaxis,:,:]
        self.M *= scale_M

    def _updateM_brute(self):
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.M[g,j,t,f] = self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]

        re_X = self.M.sum((0,1)).T
        scale_M = (self.X / re_X).mean()
        # scale_M = (self.X / re_X).T[sp.newaxis,sp.newaxis,:,:]
        self.M *= scale_M

        # print 'M:'
        # print self.M

    def _updateW(self):
        self.rhow = self.aw + self.M.sum(2)
        self.rhow = sp.transpose(self.rhow, (0,2,1))
        self.tauw[:,:,:] = (self.bw + (self.Eg[:,sp.newaxis,sp.newaxis] * self.Eh * self.Et[:,:,sp.newaxis]).sum(2))[:,sp.newaxis,:]

        self.Ew = Gamma(self.rhow, self.tauw).expectation()

        # print 'rhow:'
        # print self.rhow
        # print 'tauw:'
        # print self.tauw
        print 'Ew:'
        print self.Ew

    def _updateW_brute(self):
        self.rhow[:,:,:] = self.aw
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.rhow[g,f,j] += self.M[g,j,t,f]

        self.tauw[:,:,:] = self.bw
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.tauw[g,f,j] += self.Eg[g] * self.Eh[g,j,t] * self.Et[g,j]

        self.Ew = Gamma(self.rhow, self.tauw).expectation()

        # print 'Ew:'
        # print self.Ew

    def _updateH(self):
        self.rhoh = self.ah + self.M.sum(3)
        self.tauh[:,:,:] = (self.bh + (self.Eg[:,sp.newaxis,sp.newaxis] * self.Ew * self.Et[:,sp.newaxis,:]).sum(1))[:,:,sp.newaxis]
        self.Eh = Gamma(self.rhoh, self.tauh).expectation()

        print 'Eh:'
        print self.Eh

    def _updateH_brute(self):
        self.rhoh[:,:,:] = self.ah
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.rhoh[g,j,t] += self.M[g,j,t,f]

        self.tauh[:,:,:] = self.bh
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.tauh[g,j,t] += self.Eg[g] * self.Ew[g,f,j] * self.Et[g,j]

        self.Eh = Gamma(self.rhoh, self.tauh).expectation()

        # print 'Eh:'
        # print self.Eh

    def _updateTheta(self):
        # Ew = sp.transpose(self.Ew, (0,2,1))

        # self.rhot = self.alphat + self.M.sum((2,3))
        # self.taut = self.alphat / self.ct + (self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]).sum((2,3))
        self.rhot = self.alphat / self.Kt + self.M.sum( (2,3) )

        for g in xrange(self.Kg):
            Eh_g = self.Eh[g]
            Ew_g = self.Ew[g].T
            self.taut[g,:] = self.alphat * self.ct + (Eh_g[:,:,sp.newaxis] * Ew_g[:,sp.newaxis,:]).sum((1,2))
            # self.taut[g,:] = (self.Eg[g] * Eh_g[:,:,sp.newaxis] * Ew_g[:,sp.newaxis,:]).sum((1,2))

        self.Et = Gamma(self.rhot, self.taut).expectation()

        print 'Et:'
        print self.Et

    def _updateTheta_brute(self):
        self.rhot[:,:] = self.alphat
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.rhot[g,j] += self.M[g,j,t,f]

        self.taut[:,:] = self.alphat / self.Kt
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.taut[g,j] += self.Eg[g] * self.Eh[g,j,t] * self.Ew[g,f,j]

        self.Et = Gamma(self.rhot, self.taut).expectation()

        # print 'Et:'
        # print self.Et

    def _updateG(self):
        Ew = sp.transpose(self.Ew, (0,2,1))
        self.rhog = self.alphag / self.Kg + self.M.sum((1,2,3))
        self.taug = self.alphag * self.cg + (self.Eh[:,:,:,sp.newaxis] * Ew[:,:,sp.newaxis,:] * self.Et[:,:,sp.newaxis,sp.newaxis]).sum((1,2,3))
        self.Eg = Gamma(self.rhog, self.taug).expectation()

        print 'Eg:'
        print self.Eg

    def _updateG_brute(self):
        self.rhog[:] = self.alphag
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.rhog[g] += self.M[g,j,t,f]

        self.taug[:] = self.alphag / self.cg
        for g in xrange(self.Kg):
            for j in xrange(self.Kt):
                for t in xrange(self.nT):
                    for f in xrange(self.nF):
                        self.taug[g] += self.Eh[g,j,t] * self.Ew[g,f,j] * self.Et[g,j]

        self.Eg = Gamma(self.rhog, self.taug).expectation()

        # print 'Eg:'
        # print self.Eg

    def _compute_expectations(self):
        self.Ew = Gamma(self.rhow, self.tauw).expectation()
        self.Eh = Gamma(self.rhoh, self.tauh).expectation()
        self.Et = Gamma(self.rhot, self.taut).expectation()
        self.Eg = Gamma(self.rhog, self.taug).expectation()

    def reconstruct(self):
        # Ex = sp.zeros((self.nF,self.nT))

        # for g in xrange(self.Kg):
        #     Exg = sp.dot(self.Ew[g], self.Et[g,:,sp.newaxis]*self.Eh[g])
        #     Ex += Exg

        self._updateM()
        Ex = self.M.sum((0,1)).T

        return Ex

    def KLDivergence(self):
        """
        入力スペクトログラムXと推論されたスペクトログラムtheta*W*HのKLダイバージェンス
        """
        X = self.X
        Y = self.reconstruct()
        return (X*(sp.log(X)-sp.log(Y)) + (Y-X)).sum()

        # goodk = self._goodK()
        # E_tWH_inv = sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis]*self.Eh_inv_inv[goodk,:])
        
        # XX = self.X * E_tWH_inv**(-2)
        # score = (XX * sp.dot(self.Ew_inv_inv[:,goodk], self.Et_inv_inv[goodk,sp.newaxis] * self.Eh_inv_inv[goodk,:])).sum() - sp.log(sp.dot(self.Ew[:,goodk], self.Et[goodk,sp.newaxis]*self.Eh[goodk,:]).sum())

        # return score
