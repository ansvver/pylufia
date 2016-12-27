# -*- coding: utf-8 -*-

"""
@package word2vec.py
@brief a wrapper class of gensim.models.word2vec
@author ふぇいと (@stfate)
"""

import scipy as sp
import gensim.models.word2vec as word2vec
from six import string_types


class Word2Vec(word2vec.Word2Vec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # return self.syn0[self.vocab[words].index]
            return self.syn0[self.vocab[words].index] + self.syn1neg[self.vocab[words].index]

        # return vstack( [self.syn0[self.vocab[word].index] for word in words] )
        return vstack( [self.syn0[self.vocab[word].index] + self.syn1neg[self.vocab[word].index] for word in words] )

    def score(self, sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1):
        """
        compute log-likelihood of each sentences

        hierarchical softmaxが有効な場合(hs=1 and negative=0)はgensimの実装をそのまま用いる．
        そうでない場合(つまりnegative samplingによる学習を行う場合)はgensimに実装がないので
        対数尤度の計算を自前で行う．
        """
        if self.hs == 1 and self.negative == 0:
            return super().score(sentences, total_sentences, chunksize, queue_factor, report_delay)
        else:
            return self._score_sg(sentences, total_sentences, chunksize, queue_factor, report_delay)

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
