# -*- coding: utf-8 -*-

"""
@package word2vec.py
@brief a wrapper class of gensim.models.word2vec
@author Dan SASAI (YCJ,RDD)
"""

import scipy as sp
import gensim.models.word2vec as word2vec
from six import string_types
from copy import deepcopy
import ymh_mir.utils.array as array


class Word2Vec(word2vec.Word2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, words):
        if isinstance(words, string_types):
            return self.get_wordvector(words)

        return vstack( [self.get_wordvector(word) for word in words] )

    def get_wordvector(self, word):
        if word in self.wv.vocab:
            return self.wv[word]
        else:
            return sp.zeros(self.vector_size)

    def get_vocab_size(self):
        return len(self.wv.vocab)

    def get_vector_size(self):
        return self.vector_size

    def get_window_size(self):
        return self.window

    def score(self, sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1):
        """
        compute log-likelihood of each sentences

        hierarchical softmaxが有効な場合(hs=1 and negative=0)はgensimの実装をそのまま用いる．
        そうでない場合(つまりnegative samplingによる学習をする場合)はgensimに実装がないので
        対数尤度の計算を自前で行う．
        """
        if self.hs == 1 and self.negative == 0:
            return super().score(sentences, total_sentences, chunksize, queue_factor, report_delay)
        else:
            return self._score_sg(sentences, total_sentences, chunksize, queue_factor, report_delay)

    @classmethod
    def load(cls, *args, **kwargs):
        model = cls()
        _model = super(Word2Vec, cls).load(*args, **kwargs)
        model.__dict__ = _model.__dict__.copy()
        return model

    def save_c_format(self, dirname):
        import progressbar
        vocabs = sorted( list( self.wv.vocab.keys() ) )
        fn_vocab = "{}/vocab.txt".format(dirname)
        fn_wv = "{}/wv.bin".format(dirname)
        # fn_wv = "{}/wv.npz".format(dirname)
        
        # vocablaries
        with open(fn_vocab, "w", encoding="utf-8") as fo:
            bar = progressbar.ProgressBar( max_value=len(vocabs) )
            for i,_vocab in enumerate(vocabs):
                fo.write(_vocab+"\n")
                bar.update(i+1)

        # word vectors
        n_vocab = len(vocabs)
        n_vec_size = self.vector_size
        wv_mat = sp.zeros( (n_vocab,n_vec_size), dtype=sp.float32 )
        bar = progressbar.ProgressBar( max_value=len(vocabs) )
        for v,_vocab in enumerate(vocabs):
            wv_mat[v,:] = self.wv[_vocab].astype(sp.float32)
            bar.update(v+1)
        array.write_matrix_binary( fn_wv, wv_mat, dtype="float" )
        # sp.savez(fn_wv, wv=wv_mat)

    def _score_sg(self, sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1):
        log_prob_sentences = []
        for sentence in sentences:
            log_prob_sentences.append( self._log_likelihood_sg(sentence) )
        return log_prob_sentences

    def _log_likelihood_sg(self, sentence):
        log_prob_sentence = 0.0

        word_vocabs = [self.vocab[w] for w in sentence if w in self.vocab]
        for pos,word in enumerate(word_vocabs):
            if word is None:
                continue

            start = max(0, pos - self.window)
            for pos2, word2 in enumerate(word_vocabs[start:pos+self.window+1], start):
                if word2 is not None and pos2 != pos:
                    log_prob_sentence += self._softmax_sg(word, word2)

        return log_prob_sentence

    def _softmax_sg(self, word_o, word_i):
        v_o = self.syn1neg[word_o.index]
        v_i = self.syn0[word_i.index]
        numer = sp.exp( sp.dot(v_o, v_i) )
        # denom = 0.0
        # for _w in range(self.syn1neg.shape[0]):
        #     denom += sp.exp( sp.dot(self.syn1neg[_w], v_i) )
        denom = sp.exp( sp.dot(self.syn1neg, v_i) ).sum()
        log_prob = sp.log(numer/denom)
        return log_prob
