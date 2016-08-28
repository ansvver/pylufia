# -*- coding: utf-8 -*-

"""
cgs_lda_cy.pyx

Collapsed Gibbs SamplingによるLDA(Cython実装)
"""

cimport cython
import numpy as np
cimport numpy as np
import scipy.sparse as sp_sparse

import time #perfmeas


class CGSLDA_cy():
    """
    Inputs:
        documents: 学習に用いる文書集合のbag-of-words特徴量[n_docs,n_words]
        n_topics: トピック数
        alpha: トピック事前分布のハイパーパラメータ
        beta: トピック-単語事前分布のハイパーパラメータ

    Parameters:
        n_kw: トピックkに単語wが出てきた回数
        n_k: トピックkの単語総数
        n_dk: 文書dにトピックkに属する単語が出てきた回数
        n_d: 文書dの単語総数
    """
    def __init__(self, documents, n_topics=10, alpha=0.5, beta=0.5):
        self.documents = documents
        self.n_topics = n_topics
        self.n_docs,self.n_words = documents.shape
        self.alpha = alpha
        self.beta = beta

        self.phi = np.zeros( (self.n_topics,self.n_words), dtype=np.double )
        self.phi_history = []
        self.theta = np.zeros( (self.n_docs,self.n_topics), dtype=np.double )
        self.theta_history = []
        self.history_size = 200
        self.topics = np.zeros( (self.n_docs,self.n_words), dtype=np.int32 )
        # self.topics = sp.sparse.lil_matrix( (self.n_docs,self.n_words), dtype=np.int32 )
        self.n_kw = np.zeros( (self.n_topics,self.n_words), dtype=np.int32 )
        # self.n_kw = sp.sparse.lil_matrix( (self.n_topics,self.n_words), dtype=np.int32 )
        self.n_k = np.zeros(self.n_topics, dtype=np.int32)
        # self.n_k = sp.sparse.lil_matrix(self.n_topics, dtype=np.int32)
        self.n_dk = np.zeros( (self.n_docs,self.n_topics), dtype=np.int32 )
        # self.n_dk = sp.sparse.lil_matrix( (self.n_docs,self.n_topics), dtype=np.int32 )
        self.n_d = np.zeros(self.n_docs, dtype=np.int32)
        # self.n_d = sp.sparse.lil_matrix(self.n_docs, dtype=np.int32)
        self.W = 0

    def _initialize_parameters(self):
        self.W = 0

        cdef int d = 0
        cdef int w = 0
        cdef int n_words_cur = 0
        for d in range(self.n_docs):
            self.topics[d,:] = np.random.randint(0, self.n_topics, self.n_words)
            exist_words_idx = np.where(self.documents[d] > 0)[0]
            n_words_cur = len(exist_words_idx)
            for w in exist_words_idx:
                k = self.topics[d,w]
                self.n_dk[d,k] += self.documents[d,w]
                self.n_d[d] += self.documents[d,w]
                self.n_kw[k,w] += self.documents[d,w]
                self.n_k[k] += self.documents[d,w]
        
        # self.W = len( np.where(self.documents.sum(0) > 0)[0] )
        self.W = self.n_words

    def infer(self, n_iter=100):
        self._initialize_parameters()

        P_arr = np.zeros(n_iter, dtype=float)
        for it in range(n_iter):
            st = time.clock() #perfmeas

            self._update_counters()

            ed = time.clock() #perfmeas
            print( "PROCTIME for updateCounters()={}sec".format(ed-st) ) #perfmeas

            st = time.clock() #perfmeas

            self._update_parameters()

            ed = time.clock() #perfmeas
            print( "PROCTIME for updateParameters()={}sec".format(ed-st) ) #perfmeas

            P = self.perplexity(self.documents)
            P_arr[it] = P
            print( 'iterates: {} perplexity={}'.format(it,P) )

        return P_arr

    def _update_counters(self):
        cdef int d,w,k,new_k,dw
        cdef int n_docs = self.n_docs

        #for d in xrange(self.n_docs):
        for d from 0 <= d < n_docs:
            exist_words_idx = np.where(self.documents[d] > 0)[0]
            for w in exist_words_idx:
                k = self.topics[d,w]
                dw = self.documents[d,w]
                # w番目の単語t(トピックk)についてカウンタを減算
                self.n_dk[d,k] -= dw
                # self.n_d[d] -= self.documents[d,w]
                self.n_kw[k,w] -= dw
                self.n_k[k] -= dw

                # トピック事後分布に基づきトピックを再設定
                # p_z = (self.n_kw[:,w] + self.beta) * (self.n_dk[d] + self.alpha) / ( (self.n_k + self.W * self.beta) * (self.n_d[d] + self.n_topics * self.alpha) )
                p_z = (self.n_kw[:,w] + self.beta) * (self.n_dk[d] + self.alpha) / (self.n_k + self.W * self.beta)
                new_k = np.random.multinomial( 1, p_z/p_z.sum() ).argmax()

                # 新しいトピックを割当&カウンタ増加
                self.topics[d,w] = new_k
                self.n_dk[d,new_k] += dw
                # self.n_d[d] += self.documents[d,w]
                self.n_kw[new_k,w] += dw
                self.n_k[new_k] += dw

    def _update_parameters(self):
        """
        更新されたカウンタに基づきphiとthetaを更新する
        """
        self.phi = (self.n_kw + self.beta) / (self.n_k[:,np.newaxis] + self.W * self.beta)
        self.theta = (self.n_dk + self.alpha) / (self.n_d[:,np.newaxis] + self.n_topics * self.alpha)

        self._push_value_to_history(self.phi_history, self.phi)
        self._push_value_to_history(self.theta_history, self.theta)

    def predict_topic_prob_old(self, document):
        """
        新しいdocumentに対してtopic-word distributionとtopic distributionを計算する
        (既存のtopic-word distributionとdocumentのbag-of-wordsから計算する．たぶん正しくないやり方)
        """
        return (self.phi * document[np.newaxis,:]).sum(1) / document.sum()

    def predict_topic_prob(self, document, n_iter=100):
        """
        新しいdocumentに対してtopic-word distributionとtopic distributionを計算する
        """
        # n_dk = self.n_dk.copy()
        # n_dk = np.r_[n_dk, np.zeros(self.n_topics, dtype=int)[np.newaxis,:]]
        # n_d = self.n_d.copy()
        # n_d = np.r_[n_d, [0]]
        # n_kw = self.n_kw.copy()
        # n_k = self.n_k.copy()
        cdef np.ndarray[int,ndim=1] n_dk_new = np.zeros( (self.n_topics), dtype=np.int32 )
        cdef int n_d_new = 0
        cdef np.ndarray[int,ndim=2] n_kw_new = np.zeros( (self.n_topics,self.n_words), dtype=np.int32 )
        cdef np.ndarray[int,ndim=2] n_k_new = np.zeros( (self.n_topics), dtype=np.int32 )

        cdef np.ndarray[int,ndim=2] topics = np.random.randint(0, self.n_topics, self.n_words)
        cdef int it,w,k,dw

        theta_log = []

        exist_words_idx = np.where(document > 0)[0]
        n_words_cur = len(exist_words_idx)
        for w in exist_words_idx:
        # for w in xrange(self.n_words):
            dw = document[w]
            k = topics[w]
            n_dk_new[k] += dw
            n_d_new += dw
            n_kw_new[k,w] += dw
            n_k_new[k] += dw

        #for it in xrange(n_iter):
        for it from 0 <= it < n_iter:
            for w in exist_words_idx:
            # for w in xrange(self.n_words):
                k = topics[w]
                # w番目の単語t(トピックk)についてカウンタを減算
                n_dk_new[k] -= dw
                # n_d[-1] -= document[w]
                n_kw_new[k,w] -= dw
                n_k_new[k] -= dw

                # トピック事後分布に基づきトピックを再設定
                # p_z = (n_kw[:,w] + self.beta) * (n_dk[-1] + self.alpha) / ( (n_k + self.W * self.beta) * (n_d[-1] + self.n_topics * self.alpha) )
                p_z = (self.n_kw[:,w] + n_kw_new[:,w] + self.beta) * (n_dk_new + self.alpha) / (self.n_k + n_k_new + self.W * self.beta)

                new_k = np.random.multinomial(1, p_z/p_z.sum()).argmax()

                # 新しいトピックを割当&カウンタ増加
                topics[w] = new_k
                n_dk_new[new_k] += dw
                # n_d[-1] += document[w]
                n_kw_new[new_k,w] += dw
                n_k_new[new_k] += dw

            #phi = (self.n_kw + n_kw_new + self.beta) / (self.n_k[:,np.newaxis] + n_k_new[:,np.newaxis] + self.W * self.beta)
            theta = (n_dk_new + self.alpha) / (n_d_new + self.n_topics * self.alpha)
            self._pushValueToLog(theta_log, theta)

            
        #P = self.perplexity_for_new_document(document, theta, phi)
        #print 'Perplexity={}'.format(P)
        theta_final = np.array(theta_log).mean(0)

        return theta_final

    def get_topic_word_dist(self):
        # return self.phi
        return np.array(self.phi_history[-100:]).mean(0)

    def get_topic_dist(self):
        # return self.theta
        return np.array(self.theta_history[-100:]).mean(0)

    def get_document_topic_dist(self, d):
        return self.theta[d]
    
    def perplexity(self, documents):
        perplexity = 0.0
        cdef int n_docs = documents.shape[0]
        cdef int N = documents.sum()
        cdef int d,w
        for d in xrange(n_docs):
            words_exist_idx = np.where(documents[d] > 0)[0]
            for w in words_exist_idx:
                theta_phi_sum = 0.0
                for k in xrange(self.n_topics):
                    theta_phi_sum += self.theta[d,k] * self.phi[k,w]
                perplexity -= np.log(theta_phi_sum)

        perplexity = np.exp(1/float(N) * perplexity)

        return perplexity

    def perplexity_for_new_document(self, document, theta, phi):
        perplexity = 0.0
        N = document.sum()

        words_exist_idx = np.where(document > 0)[0]
        for w in words_exist_idx:
            theta_phi_sum = 0.0
            for k in xrange(self.n_topics):
                theta_phi_sum += theta[k] * phi[k,w]
            perplexity -= np.log(theta_phi_sum)

        perplexity = np.exp(1/float(N) * perplexity)

        return perplexity

    def _push_value_to_history(self, history, val):
        if len(history) < self.history_size:
            history.append(val)
        else:
            history.pop(0)
            history.append(val)
