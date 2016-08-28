# -*- coding: utf-8 -*-

import scipy as sp
from scipy.special import polygamma


def digamma(x):
    return polygamma(0, x)

def trigamma(x):
    return polygamma(1, x)
