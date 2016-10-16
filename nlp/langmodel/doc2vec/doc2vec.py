# -*- coding: utf-8 -*-

"""
@package
@brief
@author ふぇいと (@stfate)
"""

from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument, TaggedBrownCorpus, TaggedLineDocument


class Doc2Vec(doc2vec.Doc2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
