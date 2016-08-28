# -*- coding: utf-8 -*-

"""
@file tfidf.py
@brief TF-IDF computation
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def compute_idf(documents):
    D = documents.shape[0]
    idf = sp.log( D / documents.sum(0).astype("float") )
    return idf
