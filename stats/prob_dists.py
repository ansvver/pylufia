# -*- coding: utf-8 -*-

"""
@file prob_dists.py
@brief classes for computing probabilistic distributions
@author ふぇいと (@stfate)

@description
確率分布これくしょん
"""

import scipy as sp
import scipy.misc
import scipy.special
from common import digamma
import scipy.linalg as linalg


class Bern():
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

class Binomial():
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
        return fact * (mu**x) * ((1-self.mu)**(self.N-x))

    def expectation(self):
        return self.N * self.mu

    def var(self):
        return self.N * self.mu*(1.0-self.mu)

class Multinomial():
    """
    多項分布
    """
    def __init__(self, mu):
        self.mu = mu

    def update(self, mu):
        self.mu = mu

    def prob(self, x):
        return (mu**x).prod()

    def expectation(self):
        return self.mu

    def var(self):
        return self.mu * (1-self.mu)

class Gauss():
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
        return 1.0/(sp.sqrt(2*sp.pi)*self.sigma)*sp.exp(-1.0/(2*self.sigma**2)*(x-self.mean)**2)

    def expectation(self):
        return self.mean

    def var(self):
        return self.sigma**2

class MultiGauss():
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
        pdf = sp.zeros(len(x), dtype=sp.float64)
        pdf = 1.0/sp.sqrt( (2*sp.pi)**len(x) * (linalg.det(self.sigma) + 1e-10) ) * sp.exp(-0.5 * sp.dot( sp.dot( (x-self.mean).T, linalg.inv(self.sigma)), x-self.mean) )

        return pdf

    def log_prob(self, x):
        pdf = sp.zeros(len(x), dtype=sp.float64)
        term1 = -0.5 * ( len(x) * sp.log(2*sp.pi) + sp.log(linalg.det(self.sigma) + 1e-100) )
        term2 = -0.5 * sp.dot( x-self.mean, sp.dot(linalg.inv(self.sigma), x-self.mean) )
        pdf = term1 + term2

        return pdf

    def expectation(self):
        return self.mean

    def var(self):
        return self.sigma**2

class Uniform():
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
        p = sp.zeros(len(x))
        p[:] = 1.0/(self.b-self.a)
        return p

    def expectation(self):
        return (self.b+self.a)/2.0

    def var(self):
        return (self.b-self.a)**2/12.0

class Gamma():
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
        return x**(self.a-1) * sp.exp(-x/self.b) / (gamma(self.a)*(self.b**self.a))

    def expectation(self):
        return self.a / self.b

    def log_expectation(self):
        return digamma(self.a) - sp.log(self.b)

    def var(self):
        return self.a / self.b**2

class Poisson():
    """
    Poisson分布
    """
    def __init__(self, lam):
        self.lam = lam

    def update(self, lam):
        self.lam = lam

    def prob(self, x):
        gamma = scipy.special.gamma
        return (sp.exp(x*sp.log(self.lam) - self.lam - sp.log(gamma(x+1))))

    def expectation(self):
        return self.lam

    def var(self):
        return self.lam

class Beta():
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

        pdf = gamma(a+b) / (gamma(a)*gamma(b)) * mu**(a-1) * (1.0-mu)**(b-1)
        return pdf

    def expectation(self):
        return a / (a+b)

    def var(self):
        return a*b / ((a+b)**2 * (a+b+1))

class Dirichlet():
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
        gamma = scipy.special.gamma
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

class GIG():
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
        numer = sp.exp((self.alpha-1)*sp.log(y)-self.beta*y-self.gamma/y) * self.beta**(self.alpha/2)
        denom = 2*self.gamma**(self.alpha/2) * scipy.special.kve(self.alpha, 2*sp.sqrt(self.beta*self.gamma))
        pdf = numer / denom
        return pdf

    def expectation(self):
        """
        E[y]

        betaかgammaが0の場合期待値がNaNになってしまう問題
        """

        E = sp.zeros_like(self.beta)
        gig_inds = (self.gamma > 1e-200)
        gam_inds = (self.gamma <= 1e-200)

        sqrt_beta = sp.sqrt(self.beta[gig_inds])
        sqrt_gamma = sp.sqrt(self.gamma[gig_inds])

        bessel_alpha_plus = scipy.special.kve(self.alpha+1, 2*sqrt_beta*sqrt_gamma)
        bessel_alpha = scipy.special.kve(self.alpha, 2*sqrt_beta*sqrt_gamma)
        E[gig_inds] = bessel_alpha_plus * sqrt_gamma/sqrt_beta / bessel_alpha

        E[gam_inds] = self.alpha / self.beta[gam_inds]

        # bessel_alpha_plus = scipy.special.kve(self.alpha+1, 2*sp.sqrt(self.beta*self.gamma))
        # bessel_alpha = scipy.special.kve(self.alpha, 2*sp.sqrt(self.beta*self.gamma))
        # E = bessel_alpha_plus * sp.sqrt(self.gamma/self.beta) / bessel_alpha

        return E

    def inv_expectation(self):
        """
        E[1/y]
        """
        Einv = sp.zeros_like(self.beta)
        gig_inds = (self.gamma > 1e-200)
        gam_inds = (self.gamma <= 1e-200)

        sqrt_beta = sp.sqrt(self.beta[gig_inds])
        sqrt_gamma = sp.sqrt(self.gamma[gig_inds])

        bessel_alpha_minus = scipy.special.kve(self.alpha-1, 2*sqrt_beta*sqrt_gamma)
        bessel_alpha = scipy.special.kve(self.alpha, 2*sqrt_beta*sqrt_gamma)
        Einv[gig_inds] = bessel_alpha_minus / (sqrt_gamma/sqrt_beta*bessel_alpha)

        Einv[gam_inds] = self.beta[gam_inds] / (self.alpha - 1)

        # bessel_alpha_minus = scipy.special.kve(self.alpha-1, 2*sp.sqrt(self.beta*self.gamma))
        # bessel_alpha = scipy.special.kve(self.alpha, 2*sp.sqrt(self.beta*self.gamma))
        # Einv = bessel_alpha_minus / (sp.sqrt(self.gamma/self.beta)*bessel_alpha)

        return Einv

    def var(self):
        """
        var[y]
        """
        fact1 = bessel(self.alpha+2, sp.sqrt(self.beta*self.gamma)) / bessel(self.alpha, sp.sqrt(self.beta*self.gamma))
        fact2 = bessel(self.alpha+1, sp.sqrt(self.beta*self.gamma)) / bessel(self.alpha, sp.sqrt(self.beta*self.gamma))

        V = self.gamma/self.beta * (fact1 - fact2**2)
        return V

class Student_t():
    def __init__(self, ):
        pass

    def function(self):
        pass
