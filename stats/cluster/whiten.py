# -*- coding: utf-8 -*-

"""
@file whiten.py
@brief whiten functions for clustering
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def whiten(features):
    """
    Normalize a group of observations on a per feature basis
    """
    return features / ( features.std(0) + 1e-10 )