# -*- coding: utf-8 -*-

"""
============================================================
@file   whiten.py
@date   2012/06/29
@author sasai

@brief  clustering functions

============================================================
"""

import scipy as sp


def whiten(features):
    """
    Normalize a group of observations on a per feature basis
    """
    return features / ( features.std(0) + 1e-10 )