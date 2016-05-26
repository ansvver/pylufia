# -*- coding: utf-8 -*-

"""
vb_nmf.py

Variational Bayes NMF
"""

import scipy as sp
from ..bayes import *

def vb_nmf(X, a_w, b_w, a_h, b_h, n_iter=100):
    """
    Variational Bayes NMF

    変分ベイズ法によるNMF
    """

    # initialize
    Winit = gamma(x, a_w, b_w/a_w)
    Hinit = gamma(x, a_h, b_h/a_h)

    Lw = Winit
    Ew = Winit
    Lh = Hinit
    Eh = Hinit

    # update Lw,Ew,Lh,Eh
    for it in xrange(n_iter):



