# -*- coding: utf-8 -*-

"""
@package word2vec.py
@brief a wrapper class of gensim.models.word2vec
@author stfate
"""

import scipy as sp
import gensim.models.word2vec as word2vec
from six import string_types
from copy import deepcopy
import pylufia.utils.array as array


class Word2Vec():
    def __init__(self, *args, **kwargs):
        self.model = word2vec.Word2Vec(*args, **kwargs)

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # return self.syn0[self.vocab[words].index]
            # return self.syn0[self.vocab[words].index] + self.syn1neg[self.vocab[words].index]
            return self.get_wordvector(words)

        # return vstack( [self.syn0[self.vocab[word].index] for word in words] )
        # return vstack( [self.syn0[self.vocab[word].index] + self.syn1neg[self.vocab[word].index] for word in words] )
        return vstack( [self.get_wordvector(word) for word in words] )

    def get_wordvector(self, word):
        if word in self.model.wv.vocab:
            return self.model.wv[word]
        else:
            return sp.zeros(self.model.vector_size)

    def get_vocab_size(self):
        return len(self.model.wv.vocab)

    def get_vector_size(self):
        return self.model.vector_size

    def get_window_size(self):
        return self.model.window

    def score(self, sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1):
        """
        compute log-likelihood of each sentences

        hierarchical softmaxが有効な場合(hs=1 and negative=0)はgensimの実装をそのまま用いる．
        そうでない場合(つまりnegative samplingによる学習をする場合)はgensimに実装がないので
        対数尤度の計算を自前で行う．
        """
        if self.hs == 1 and self.negative == 0:
            return self.model.score(sentences, total_sentences, chunksize, queue_factor, report_delay)
        else:
            return self._score_sg(sentences, total_sentences, chunksize, queue_factor, report_delay)

    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        rtn = cls()
        rtn.model = word2vec.Word2Vec.load(*args, **kwargs)
        return rtn

    def save_c_format(self, dirname):
        vocabs = sorted( list( self.model.wv.vocab.keys() ) )
        fn_vocab = "{}/vocab.txt".format(dirname)
        fn_wv = "{}/wv.bin".format(dirname)
        
        # vocablaries
        with open(fn_vocab, "w", encoding="utf-8") as fo:
            for _vocab in vocabs:
                fo.write(_vocab+"\n")

        # word vectors
        n_vocab = len(vocabs)
        n_vec_size = self.model.vector_size
        wv_mat = sp.zeros( (n_vocab,n_vec_size), dtype=sp.float32 )
        for v,_vocab in enumerate(vocabs):
            wv_mat[v,:] = self.model.wv[_vocab]
        array.write_matrix_binary( fn_wv, wv_mat, dtype="float" )

    def _score_sg(self, sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1):
        log_prob_sentences = []
        for sentence in sentences:
            log_prob_sentences.append( self._log_likelihood_sg(sentence) )
        return log_prob_sentences

    def _log_likelihood_sg(self, sentence):
        log_prob_sentence = 0.0

        word_vocabs = [self.model.vocab[w] for w in sentence if w in self.model.vocab]
        for pos,word in enumerate(word_vocabs):
            if word is None:
                continue

            start = max(0, pos - self.model.window)
            for pos2, word2 in enumerate(word_vocabs[start:pos+self.model.window+1], start):
                if word2 is not None and pos2 != pos:
                    log_prob_sentence += self._softmax_sg(word, word2)

        return log_prob_sentence

    def _softmax_sg(self, word_o, word_i):
        v_o = self.model.syn1neg[word_o.index]
        v_i = self.model.syn0[word_i.index]
        numer = sp.exp( sp.dot(v_o, v_i) )
        # denom = 0.0
        # for _w in range(self.syn1neg.shape[0]):
        #     denom += sp.exp( sp.dot(self.syn1neg[_w], v_i) )
        denom = sp.exp( sp.dot(self.model.syn1neg, v_i) ).sum()
        log_prob = sp.log(numer/denom)
        return log_prob
