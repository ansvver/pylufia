# -*- coding: utf-8 -*-

"""
@package doc2vec.py
@brief
@author Dan SASAI (YCJ,RDD)
"""

import scipy as sp
import gensim.models.doc2vec as doc2vec


class Doc2Vec(doc2vec.Doc2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        model = cls()
        _model = super(Doc2Vec, cls).load(*args, **kwargs)
        model.__dict__ = _model.__dict__.copy()
        return model
