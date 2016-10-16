# -*- coding: utf-8 -*-

"""
@file tfidf.py
@brief TF-IDF computation
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def compute_tf(documents):
    """ compute TF(term frequency)
    @param documents bag-of-words features of documents [n_docs,n_dims]
    @return tf of each words [n_docs,n_dims]
    """
    tf = documents / documents.sum(1)
    return tf

def compute_idf(documents):
    """ compute IDF(inverse document frequency)
    @param documents bag-of-words features of documents [n_docs,n_dims]
        we assums feature dimensions as counts of each vocabularies
    @return idf of each words [n_dims]
    """
    D = documents.shape[0]
    idf = sp.log( D / documents.sum(0).astype("float") )
    return idf

def compute_tfidf(documents):
    """ compute TFIDF
    @param documents bag-of-words features of documents [n_docs,n_dims]
    @return tdidf of 
    """
    return compute_tf(documents) * compute_idf(documents)[sp.newaxis,:]
