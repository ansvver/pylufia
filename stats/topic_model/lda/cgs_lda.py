# -*- coding: utf-8 -*-

"""
gibbs_lda.py

Collapsed Gibbs SamplingによるLDA
"""

import scipy as sp
import scipy.sparse as sp_sparse


class CgsLda():
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
    def __init__(self, n_topics=10, alpha=0.5, beta=0.5):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.history_size = 200
        self.W = 0

    def _alloc_arrays(self):
        self.phi = sp.zeros( (self.n_topics,self.n_words), dtype=sp.double )
        self.phi_history = []
        self.theta = sp.zeros( (self.n_docs,self.n_topics), dtype=sp.double )
        self.theta_history = []
        self.topics = sp.zeros( (self.n_docs,self.n_words), dtype=sp.int32 )
        # self.topics = sp_sparse.lil_matrix( (self.n_docs,self.n_words), dtype=sp.int32 )
        self.n_kw = sp.zeros( (self.n_topics,self.n_words), dtype=sp.int32 )
        # self.n_kw = sp_sparse.lil_matrix( (self.n_topics,self.n_words), dtype=sp.int32 )
        self.n_k = sp.zeros(self.n_topics, dtype=sp.int32)
        # self.n_k = sp_sparse.lil_matrix(self.n_topics, dtype=sp.int32)
        self.n_dk = sp.zeros( (self.n_docs,self.n_topics), dtype=sp.int32 )
        # self.n_dk = sp_sparse.lil_matrix( (self.n_docs,self.n_topics), dtype=sp.int32 )
        self.n_d = sp.zeros(self.n_docs, dtype=sp.int32)
        # self.n_d = sp_sparse.lil_matrix(self.n_docs, dtype=sp.int32)

    def _initialize_parameters(self):
        for d in range(self.n_docs):
            self.topics[d,:] = sp.random.randint(0, self.n_topics, self.n_words)
            exist_words_idx = sp.where(self.documents[d] > 0)[0]
            n_words_cur = len(exist_words_idx)
            for w in exist_words_idx:
                k = self.topics[d,w]
                self.n_dk[d,k] += self.documents[d,w]
                self.n_d[d] += self.documents[d,w]
                self.n_kw[k,w] += self.documents[d,w]
                self.n_k[k] += self.documents[d,w]
        
        # self.W = len( sp.where(self.documents.sum(0) > 0)[0] )
        self.W = self.n_words

    def infer(self, documents, labels=None, n_iter=100):
        self.documents = documents
        self.labels = labels
        self.n_docs,self.n_words = self.documents.shape
        self._alloc_arrays()
        self._initialize_parameters()

        P_arr = sp.zeros(n_iter)
        for it in range(n_iter):
            self._update_counters()
            self._update_parameters()

            P = self.perplexity(self.documents)
            P_arr[it] = P
            print( 'iterates: {} perplexity={}'.format(it,P) )

        self.theta = sp.array(self.theta_history).mean(0)
        self.phi = sp.array(self.phi_history).mean(0)

        self.theta_history = []
        self.phi_history = []

        return P_arr

    def _update_counters(self):
        for d in range(self.n_docs):
            exist_words_idx = sp.where(self.documents[d] > 0)[0]
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
                new_k = sp.random.multinomial( 1, p_z/p_z.sum() ).argmax()

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
        self.phi = (self.n_kw + self.beta) / (self.n_k[:,sp.newaxis] + self.W * self.beta)
        self.theta = (self.n_dk + self.alpha) / (self.n_d[:,sp.newaxis] + self.n_topics * self.alpha)

        self._push_value_to_history(self.phi_history, self.phi)
        self._push_value_to_history(self.theta_history, self.theta)

    def predict_topic_prob_old(self, document):
        """
        新しいdocumentに対してtopic-word distributionとtopic distributionを計算する
        (既存のtopic-word distributionとdocumentのbag-of-wordsから計算する．たぶん正しくないやり方)
        """
        return (self.phi * document[sp.newaxis,:]).sum(1) / document.sum()

    def predict_topic_prob(self, document, n_iter=100):
        """
        新しいdocumentに対してtopic-word distributionとtopic distributionを計算する
        """
        # n_dk = self.n_dk.copy()
        # n_dk = sp.r_[n_dk, sp.zeros(self.n_topics, dtype=int)[sp.newaxis,:]]
        # n_d = self.n_d.copy()
        # n_d = sp.r_[n_d, [0]]
        # n_kw = self.n_kw.copy()
        # n_k = self.n_k.copy()
        n_dk_new = sp.zeros( (self.n_topics), dtype=sp.int32 )
        n_d_new = 0
        n_kw_new = sp.zeros( (self.n_topics,self.n_words), dtype=sp.int32 )
        n_k_new = sp.zeros( (self.n_topics), dtype=sp.int32 )

        topics = sp.random.randint(0, self.n_topics, self.n_words)
        theta_history = []

        exist_words_idx = sp.where(document > 0)[0]
        n_words_cur = len(exist_words_idx)
        for w in exist_words_idx:
        # for w in xrange(self.n_words):
            dw = document[w]
            k = topics[w]
            n_dk_new[k] += dw
            n_d_new += dw
            n_kw_new[k,w] += dw
            n_k_new[k] += dw

        for it in range(n_iter):
            for w in exist_words_idx:
            # for w in xrange(self.n_words):
                k = topics[w]
                dw = document[w]
                # w番目の単語t(トピックk)についてカウンタを減算
                n_dk_new[k] -= dw
                # n_d[-1] -= document[w]
                n_kw_new[k,w] -= dw
                n_k_new[k] -= dw

                # トピック事後分布に基づきトピックを再設定
                # p_z = (n_kw[:,w] + self.beta) * (n_dk[-1] + self.alpha) / ( (n_k + self.W * self.beta) * (n_d[-1] + self.n_topics * self.alpha) )
                p_z = (self.n_kw[:,w] + n_kw_new[:,w] + self.beta) * (n_dk_new + self.alpha) / (self.n_k + n_k_new + self.W * self.beta)

                new_k = sp.random.multinomial(1, p_z/p_z.sum()).argmax()

                # 新しいトピックを割当&カウンタ増加
                topics[w] = new_k
                n_dk_new[new_k] += dw
                # n_d[-1] += document[w]
                n_kw_new[new_k,w] += dw
                n_k_new[new_k] += dw

            # phi = (self.n_kw + n_kw_new + self.beta) / (self.n_k[:,sp.newaxis] + n_k_new[:,sp.newaxis] + self.W * self.beta)
            theta = (n_dk_new + self.alpha) / (n_d_new + self.n_topics * self.alpha)
            self._push_value_to_history(theta_history, theta)

        theta_final = sp.array(theta_history).mean(0)

        return theta_final

    def get_topic_word_dist(self):
        return self.phi
        # return sp.array(self.phi_log[-100:]).mean(0)

    def get_topic_dist(self):
        return self.theta
        # return sp.array(self.theta_log[-100:]).mean(0)

    def get_document_topic_dist(self, d):
        return self.theta[d]

    def get_documents(self):
        return self.documents

    def get_labels(self):
        return self.labels
    
    def perplexity(self, documents):
        perplexity = 0.0
        n_docs = documents.shape[0]
        N = documents.sum()
        for d in range(n_docs):
            words_exist_idx = sp.where(documents[d] > 0)[0]
            for w in words_exist_idx:
                theta_phi_sum = 0.0
                for k in range(self.n_topics):
                    theta_phi_sum += self.theta[d,k] * self.phi[k,w]
                perplexity -= sp.log(theta_phi_sum)

        perplexity = sp.exp(1/float(N) * perplexity)

        return perplexity

    def perplexity_for_new_document(self, document, theta, phi):
        perplexity = 0.0
        N = document.sum()

        words_exist_idx = sp.where(document > 0)[0]
        for w in words_exist_idx:
            theta_phi_sum = 0.0
            for k in range(self.n_topics):
                theta_phi_sum += theta[k] * phi[k,w]
            perplexity -= sp.log(theta_phi_sum)

        perplexity = sp.exp(1/float(N) * perplexity)

        return perplexity

    def save(self, dirname):
        import json
        params = {"alpha": self.alpha, "beta": self.beta, "n_topics": self.n_topics, "n_docs": self.n_docs, "n_words": self.n_words}
        json.dump(params, open( "{}/params.json".format(dirname), "w" ) )
        sp.save("{}/documents.npy".format(dirname), self.documents)
        sp.save("{}/labels.npy".format(dirname), self.labels)
        sp.save("{}/phi.npy".format(dirname), self.phi)
        sp.save("{}/theta.npy".format(dirname), self.theta)
        sp.save("{}/topics.npy".format(dirname), self.topics)
        sp.save("{}/n_kw.npy".format(dirname), self.n_kw)
        sp.save("{}/n_k.npy".format(dirname), self.n_k)
        sp.save("{}/n_dk.npy".format(dirname), self.n_dk)
        sp.save("{}/n_d.npy".format(dirname), self.n_d)

    @classmethod
    def load(cls, dirname):
        import json
        model = cls()
        params = json.load( open( "{}/params.json".format(dirname) ) )
        model.alpha = params["alpha"]
        model.beta = params["beta"]
        model.n_topics = params["n_topics"]
        model.n_docs = params["n_docs"]
        model.n_words = params["n_words"]
        model.documents = sp.load( "{}/documents.npy".format(dirname) )
        model.labels = sp.load( "{}/labels.npy".format(dirname) )
        model.phi = sp.load( "{}/phi.npy".format(dirname) )
        model.theta = sp.load( "{}/theta.npy".format(dirname) )
        model.topics = sp.load( "{}/topics.npy".format(dirname) )
        model.n_kw = sp.load( "{}/n_kw.npy".format(dirname) )
        model.n_k = sp.load( "{}/n_k.npy".format(dirname) )
        model.n_dk = sp.load( "{}/n_dk.npy".format(dirname) )
        model.n_d = sp.load( "{}/n_d.npy".format(dirname) )
        return model

    def _push_value_to_history(self, history, val):
        if len(history) < self.history_size:
            history.append(val)
        else:
            history.pop(0)
            history.append(val)
