# -*- coding: utf-8 -*-

"""
@file prob_dists_cy.pyx
@brief classes for computing probabilistic distributions (cython version)
@author ふぇいと (@stfate)

@description
確率分布これくしょん
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy.misc
import scipy.special
from common import digamma
import scipy.linalg as linalg


class Bern_cy():
    """
    Bernoulli分布
    """
    def __init__(self, mu):
        self.mu = mu

    def update(self, mu):
        self.mu = mu

    def prob(self, x):
        return self.mu**x - (1-self.mu)**(1-x)

    def expectation(self):
        return self.mu

    def var(self):
        return self.mu * (1-self.mu)

class Binomial_cy():
    """
    二項分布
    """
    def __init__(self, mu, N):
        self.mu = mu
        self.N = N

    def update(self, mu, N):
        self.mu = mu
        self.N = N

    def prob(self, x):
        factorial = scipy.misc.factorial
        fact = factorial(self.N)/ (factorial(self.N-x) * factorial(x))
        return fact * (self.mu**x) * ((1-self.mu)**(self.N-x))

    def expectation(self):
        return self.N * self.mu

    def var(self):
        return self.N * self.mu*(1.0-self.mu)

class Multinomial_cy():
    """
    多項分布
    """
    def __init__(self, mu):
        self.mu = mu

    def update(self, mu):
        self.mu = mu

    def prob(self, x):
        return (self.mu**x).prod()

    def expectation(self):
        return self.mu

    def var(self):
        return self.mu * (1-self.mu)

class Gauss_cy():
    """
    正規分布
    """
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def update(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def prob(self, x):
        return 1.0/(np.sqrt(2*np.pi)*self.sigma)*np.exp(-1.0/(2*self.sigma**2)*(x-self.mean)**2)

    def expectation(self):
        return self.mean

    def var(self):
        return self.sigma**2

class MultiGauss_cy():
    """
    多変量正規分布
    """
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def update(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def prob(self, x):
        pdf = np.zeros(len(x), dtype=np.float64)
        pdf = 1.0/np.sqrt( (2*np.pi)**len(x) * (linalg.det(self.sigma) + 1e-10) ) * np.exp(-0.5 * np.dot( np.dot( (x-self.mean).T, linalg.inv(self.sigma)), x-self.mean) )

        return pdf

    def log_prob(self, x):
        pdf = np.zeros(len(x), dtype=np.float64)
        term1 = -0.5 * ( len(x) * np.log(2*np.pi) + np.log(linalg.det(self.sigma) + 1e-100) )
        term2 = -0.5 * np.dot( x-self.mean, np.dot(linalg.inv(self.sigma), x-self.mean) )
        pdf = term1 + term2

        return pdf

    def expectation(self):
        return self.mean

    def var(self):
        return self.sigma**2

class Uniform_cy():
    """
    一様分布
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def update(self, a, b):
        self.a = a
        self.b = b

    def prob(self, x):
        p = np.zeros(len(x))
        p[:] = 1.0/(self.b-self.a)
        return p

    def expectation(self):
        return (self.b+self.a)/2.0

    def var(self):
        return (self.b-self.a)**2/12.0

class Gamma_cy():
    """
    Gamma分布
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def update(self, a, b):
        self.a = a
        self.b = b

    def prob(self, x):
        gamma = scipy.special.gamma
        return x**(self.a-1) * np.exp(-x/self.b) / (gamma(self.a)*(self.b**self.a))

    def expectation(self):
        return self.a / self.b

    def log_expectation(self):
        return digamma(self.a) - np.log(self.b)

    def var(self):
        return self.a / self.b**2

class Poisson_cy():
    """
    Poisson分布
    """
    def __init__(self, lam):
        self.lam = lam

    def update(self, lam):
        self.lam = lam

    def prob(self, x):
        gamma = scipy.special.gamma
        return (np.exp(x*np.log(self.lam) - self.lam - np.log(gamma(x+1))))

    def expectation(self):
        return self.lam

    def var(self):
        return self.lam

class Beta_cy():
    """
    Beta分布
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def update(self, a, b):
        self.a = a
        self.b = b

    def prob(self, mu):
        gamma = scipy.special.gamma

        pdf = gamma(self.a+self.b) / (gamma(self.a)*gamma(self.b)) * mu**(self.a-1) * (1.0-mu)**(self.b-1)
        return pdf

    def expectation(self):
        return self.a / (self.a+self.b)

    def var(self):
        return self.a*self.b / ((self.a+self.b)**2 * (self.a+self.b+1))

class Dirichlet_cy():
    """
    Dirichlet分布
    """
    def __init__(self, alpha):
        self.alpha = alpha
        self.alpha0 = self.alpha.sum()

    def update(self, alpha):
        self.alpha = alpha
        self.alpha0 = self.alpha.sum()

    def prob(self, mu):
        gammafunc = scipy.special.gamma
        fact = 1.0
        for _a in self.alpha:
            fact *= gammafunc(_a)
        nCr = gammafunc(self.alpha0) / fact
        return nCr * (mu**(self.alpha-1)).prod()

    def expectation(self):
        return self.alpha / self.alpha0

    def log_expectation(self):
        return digamma(self.alpha) - digamma(self.alpha0)

    def var(self):
        return self.alpha * (self.alpha0 - self.alpha) / (self.alpha0**2 * (self.alpha0 + 1))

class GIG_cy():
    """
    Generalized Inverse Gaussian
    """
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def update(self, new_alpha, new_beta, new_gamma):
        self.alpha = new_alpha
        self.beta = new_beta
        self.gamma = new_gamma

    def prob(self, y):
        numer = np.exp((self.alpha-1)*np.log(y)-self.beta*y-self.gamma/y) * self.beta**(self.alpha/2)
        denom = 2*self.gamma**(self.alpha/2) * scipy.special.kve(self.alpha, 2*np.sqrt(self.beta*self.gamma))
        pdf = numer / denom
        return pdf

    def expectation(self):
        """
        E[y]

        betaかgammaが0の場合期待値がNaNになってしまう問題
        """

        E = np.zeros_like(self.beta, dtype=np.double)
        gig_inds = (self.gamma > 1e-200)
        gam_inds = (self.gamma <= 1e-200)

        sqrt_beta = np.sqrt(self.beta[gig_inds])
        sqrt_gamma = np.sqrt(self.gamma[gig_inds])

        bessel_alpha_plus = scipy.special.kve(self.alpha[gig_inds]+1, 2*sqrt_beta*sqrt_gamma)
        bessel_alpha = scipy.special.kve(self.alpha[gig_inds], 2*sqrt_beta*sqrt_gamma)
        E[gig_inds] = bessel_alpha_plus * sqrt_gamma/sqrt_beta / bessel_alpha

        E[gam_inds] = self.alpha[gam_inds] / self.beta[gam_inds]

        # bessel_alpha_plus = scipy.special.kve(self.alpha+1, 2*sp.sqrt(self.beta*self.gamma))
        # bessel_alpha = scipy.special.kve(self.alpha, 2*sp.sqrt(self.beta*self.gamma))
        # E = bessel_alpha_plus * sp.sqrt(self.gamma/self.beta) / bessel_alpha

        return E

    def inv_expectation(self):
        """
        E[1/y]
        """
        Einv = np.zeros_like(self.beta, dtype=np.double)
        gig_inds = (self.gamma > 1e-200)
        gam_inds = (self.gamma <= 1e-200)

        sqrt_beta = np.sqrt(self.beta[gig_inds])
        sqrt_gamma = np.sqrt(self.gamma[gig_inds])

        bessel_alpha_minus = scipy.special.kve(self.alpha[gig_inds]-1, 2*sqrt_beta*sqrt_gamma)
        bessel_alpha = scipy.special.kve(self.alpha[gig_inds], 2*sqrt_beta*sqrt_gamma)
        Einv[gig_inds] = bessel_alpha_minus / (sqrt_gamma/sqrt_beta*bessel_alpha)

        Einv[gam_inds] = self.beta[gam_inds] / (self.alpha[gam_inds] - 1)

        # bessel_alpha_minus = scipy.special.kve(self.alpha-1, 2*sp.sqrt(self.beta*self.gamma))
        # bessel_alpha = scipy.special.kve(self.alpha, 2*sp.sqrt(self.beta*self.gamma))
        # Einv = bessel_alpha_minus / (sp.sqrt(self.gamma/self.beta)*bessel_alpha)

        return Einv

    def expectation_both(self):
        """
        E[y]とE[1/y]を同時に求める
        """
        E = np.zeros_like(self.beta, dtype=np.double)
        Einv = np.zeros_like(self.beta, dtype=np.double)
        gig_inds = (self.gamma > 1e-200)
        gam_inds = (self.gamma <= 1e-200)

        sqrt_beta = np.sqrt(self.beta[gig_inds])
        sqrt_gamma = np.sqrt(self.gamma[gig_inds])

        bessel_alpha_plus = scipy.special.kve(self.alpha[gig_inds]+1, 2*sqrt_beta*sqrt_gamma)
        bessel_alpha = scipy.special.kve(self.alpha[gig_inds], 2*sqrt_beta*sqrt_gamma)
        bessel_alpha_minus = scipy.special.kve(self.alpha[gig_inds]-1, 2*sqrt_beta*sqrt_gamma)

        E[gig_inds] = bessel_alpha_plus * sqrt_gamma/sqrt_beta / bessel_alpha
        E[gam_inds] = self.alpha[gam_inds] / self.beta[gam_inds]

        Einv[gig_inds] = bessel_alpha_minus / (sqrt_gamma/sqrt_beta*bessel_alpha)
        Einv[gam_inds] = self.beta[gam_inds] / (self.alpha[gam_inds] - 1)

        return E,Einv

    def gamma_term(self, Ex, Ex_inv, rho, tau, a, b):
        score = 0.0
        cutoff = 1e-200
        zero_tau = (tau <= cutoff)
        non_zero_tau = (tau > cutoff)

        #score += Ex.size * ( a * np.log(b) - scipy.special.gammaln(a) )
        #score -= ( (b - rho) * Ex ).sum()

        #score -= non_zero_tau.sum() * np.log(0.5)
        #score += (tau[non_zero_tau] * Ex_inv[non_zero_tau]).sum()
        #score -= 0.5 * a * (np.log(rho[non_zero_tau]) - np.log(tau[non_zero_tau])).sum()
        #score += ( np.log( scipy.special.kve(a, 2*np.sqrt(rho[non_zero_tau]*tau[non_zero_tau])) ) - 2*np.sqrt(rho[non_zero_tau]*tau[non_zero_tau]) ).sum()
        #score += (-a * np.log(rho[non_zero_tau]) + scipy.special.gammaln(a)).sum()

        sub_score1 = Ex.size * ( a * np.log(b) - scipy.special.gammaln(a) )
        score += sub_score1
        sub_score2 = ( (b - rho) * Ex ).sum()
        score -= sub_score2

        sub_score3 = non_zero_tau.sum() * np.log(0.5)
        score -= sub_score3
        sub_score4 = (tau[non_zero_tau] * Ex_inv[non_zero_tau]).sum()
        score += sub_score4
        sub_score5 = 0.5 * a * (np.log(rho[non_zero_tau]) - np.log(tau[non_zero_tau])).sum()
        score -= sub_score5
        sub_score6 = ( np.log( scipy.special.kve(a, 2*np.sqrt(rho[non_zero_tau]*tau[non_zero_tau])) ) - 2*np.sqrt(rho[non_zero_tau]*tau[non_zero_tau]) ).sum()
        score += sub_score6
        sub_score7 = (-a * np.log(rho[non_zero_tau]) + scipy.special.gammaln(a)).sum()
        score += sub_score7
        #print "{} {} {} {} {} {} {}".format(sub_score1,sub_score2,sub_score3,sub_score4,sub_score5,sub_score6,sub_score7)

        return score

    def var(self):
        """
        var[y]
        """
        fact1 = scipy.special.kve(self.alpha+2, np.sqrt(self.beta*self.gamma)) / scipy.special.kve(self.alpha, np.sqrt(self.beta*self.gamma))
        fact2 = scipy.special.kve(self.alpha+1, np.sqrt(self.beta*self.gamma)) / scipy.special.kve(self.alpha, np.sqrt(self.beta*self.gamma))

        V = self.gamma/self.beta * (fact1 - fact2**2)
        return V
