# -*- coding: utf-8 -*-

"""
@file labeled_lda.py
@brief Labeled LDA implementation
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


class LabeledLDA():
    """
    教師ありLDA
    """
    def __init__(self, documents, labels, label_list, alpha=0.5, beta=0.5):
        self.documents = documents
        self.labels = labels
        self.label_list = label_list
        self.n_labels = len(label_list)
        self.n_docs,self.n_words = documents.shape
        self.alpha = alpha
        self.beta = beta
        self.label_map = self._make_label_map(self.labels, self.label_list)
        self.alpha_d = self.label_map * self.alpha

        self.phi = sp.zeros( (self.n_labels,self.n_words), dtype=sp.double )
        self.theta = sp.zeros( (self.n_docs,self.n_labels), dtype=sp.double )
        self.topics = sp.zeros( (self.n_docs,self.n_words), dtype=sp.int32 )
        self.n_kw = sp.zeros( (self.n_labels,self.n_words), dtype=sp.int32 )
        self.n_k = sp.zeros(self.n_labels, dtype=sp.int32)
        self.n_dk = sp.zeros( (self.n_docs,self.n_labels), dtype=sp.int32 )
        self.n_d = sp.zeros(self.n_docs, dtype=sp.int32)
        self.W = 0

    def _make_label_map(self, labels, label_list):
        label_map = sp.zeros( (len(labels),len(label_list) ) )
        for d,label in enumerate(labels):
            for each_label in label:
                k = label_list.index(each_label)
                label_map[d,k] = 1

        return label_map

    def _initialize_parameters(self):
        self.W = 0

        for d in xrange(self.n_docs):
            self.topics[d,:] = sp.random.randint(0, self.n_labels, self.n_words)
            exist_words_idx = sp.where(self.documents[d] > 0)[0]
            n_words_cur = len(exist_words_idx)
            for w in exist_words_idx:
                k = self.topics[d,w]
                self.n_dk[d,k] += self.documents[d,w]
                self.n_d[d] += self.documents[d,w]
                self.n_kw[k,w] += self.documents[d,w]
                self.n_k[k] += self.documents[d,w]
        
        self.W = len( sp.where(self.documents.sum(0) > 0)[0] )

    def infer(self, n_iter=100):
        self._initialize_parameters()

        P_arr = sp.zeros(n_iter)
        for it in xrange(n_iter):
            self._update_counters()
            self._update_parameters()

            print 'phi:'
            print self.phi
            print 'theta:'
            print self.theta

            P = self.perplexity()
            P_arr[it] = P
            print 'iterates: {} perplexity={}'.format(it,P)

        return P_arr

    def _update_counters(self):
        for d in xrange(self.n_docs):
            exist_words_idx = sp.where(self.documents[d] > 0)[0]
            for w in exist_words_idx:
                k = self.topics[d,w]
                # w番目の単語t(トピックk)についてカウンタを減算
                self.n_dk[d,k] -= self.documents[d,w]
                self.n_d[d] -= self.documents[d,w]
                self.n_kw[k,w] -= self.documents[d,w]
                self.n_k[k] -= self.documents[d,w]

                # トピック事後分布に基づきトピックを再設定
                p_z = (self.n_kw[:,w] + self.beta) * (self.n_dk[d] + self.alpha_d[d]) / (self.n_k + self.W * self.beta)
                new_k = sp.random.multinomial(1, p_z/p_z.sum()).argmax()

                # 新しいトピックを割当&カウンタ増加
                self.topics[d,w] = new_k
                self.n_dk[d,new_k] += self.documents[d,w]
                self.n_d[d] += self.documents[d,w]
                self.n_kw[new_k,w] += self.documents[d,w]
                self.n_k[new_k] += self.documents[d,w]

    def _update_parameters(self):
        """
        更新されたカウンタに基づきphiとthetaを更新する
        """
        self.phi = (self.n_kw + self.beta) / (self.n_k[:,sp.newaxis] + self.W * self.beta)
        self.theta = (self.n_dk + self.alpha_d) / (self.n_d[:,sp.newaxis] + self.n_labels * self.alpha_d)

    def worddist(self):
        return self.phi
    
    def perplexity(self):
        perplexity = 0.0

        for d in xrange(self.n_docs):
            words_exist_idx = sp.where(self.documents[d] > 0)[0]
            for w in words_exist_idx:
                theta_phi_sum = 0.0
                for k in xrange(self.n_labels):
                    theta_phi_sum += self.theta[d,k]*self.phi[k,w]
                perplexity -= sp.log(theta_phi_sum)

        perplexity = sp.exp(1/float(self.W) * perplexity)

        return perplexity
