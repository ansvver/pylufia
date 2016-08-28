# -*- coding: utf-8 -*-

"""
@package btm.py
@brief an implementation of BTM(Biterm Topic Model)
@author Dan SASAI (YCJ,RDD)

@description
X. Yang "A Biterm Topic Model for Short Texts" (WWW2013)を実装．
"""

import scipy as sp


class Biterm:
    """
    @class Biterm
    @brief a class for Biterm data structure
      This class contains (wi:word index,wj:word index,z:topic)
    """
    def __init__(self, wi, wj, z):
        self.wi = wi
        self.wj = wj
        self.z = z

    def show(self):
        print "wi: {} wj: {} z: {}".format(self.wi,self.wj,self.z)


class BTM():
    """
    @class BTM
    @brief a class of Biterm Topic Model implemantation
    """
    def __init__(self, documents, n_topics=100, alpha=1.0, beta=1.0):
        self.documents = documents
        # self.all_biterms = []
        self.biterm_set = []
        self.B = 0
        self.n_topics = n_topics
        self.n_docs,self.n_words = self.documents.shape
        self.alpha = alpha
        self.beta = beta
        self.phi = sp.zeros( (self.n_topics,self.n_words), dtype=sp.double )
        self.phi_history = []
        self.theta = sp.zeros(self.n_topics, dtype=sp.double)
        self.theta_history = []
        self.history_size = 200
        self.n_z = sp.zeros(self.n_topics, dtype=sp.int32)
        self.n_zw = sp.zeros( (self.n_topics,self.n_words), dtype=sp.int32 )

        self._initializeParameters()

    def _initialize_parameters(self):
        # 文書データからbiterm生成
        all_biterms = []
        for _document in self.documents:
            _biterms = self._gen_biterms_from_one_document(_document)
            all_biterms += _biterms
        self.biterm_set = self._make_biterm_set(all_biterms)
        self.B = len(self.biterm_set)

        # n_z,n_zw計算
        # for _biterm in self.biterms:
        for _biterm in self.biterm_set:
            wi = _biterm["biterm"].wi
            wj = _biterm["biterm"].wj
            k = _biterm["biterm"].z
            count = _biterm["count"]
            self.n_z[k] += count
            self.n_zw[k,wi] += count
            self.n_zw[k,wj] += count

    def _gen_biterms_from_one_document(self, document):
        biterms = []
        exist_word_idxs = sp.where(document > 0)[0]
        for i in range( len(exist_word_idxs) ):
            wi = exist_word_idxs[i]
            for j in range( i+1, len(exist_word_idxs) ):
                wj = exist_word_idxs[j]
                k = sp.random.randint(0, self.n_topics)
                _b = Biterm(wi, wj, k)
                biterms.append(_b)
        return biterms

    def _make_biterm_set(self, all_biterms):
        biterm_set = []
        for _b in all_biterms:
            found_idx = self._find_biterm_in_biterm_list(biterm_set, _b)
            if found_idx is not None:
                biterm_set[found_idx]["count"] += 1
            else:
                biterm_set.append({"biterm":_b, "count": 1})
        return biterm_set

    def _find_biterm_in_biterm_list(self, biterm_list, target_biterm):
        found_idx = None
        for i,_b in enumerate(biterm_list):
            if _b["biterm"].wi == target_biterm.wi and _b["biterm"].wj == target_biterm.wj:
                found_idx = i
                break
        return found_idx

    def infer(self, n_iter=100):
        for it in range(n_iter):
            print( "iterates: {}".format(it) )

            # for i,_biterm in enumerate(self.biterms):
            for i,_biterm in enumerate(self.biterm_set):
                wi = _biterm["biterm"].wi
                wj = _biterm["biterm"].wj
                k = _biterm["biterm"].z
                count = _biterm["count"]
                self.n_z[k] -= count
                self.n_zw[k,wi] -= count
                self.n_zw[k,wj] -= count

                new_biterm = self._update_biterm_topic(_biterm["biterm"])
                self.biterm_set[i]["biterm"] = new_biterm

                new_k = new_biterm.z
                self.n_z[new_k] += count
                self.n_zw[new_k,wi] += count
                self.n_zw[new_k,wj] += count

            self._compute_parameters()
            self._push_value_to_history(self.phi_history, self.phi)
            self._push_value_to_history(self.theta_history, self.theta)

        self.phi = sp.array(self.phi_history).mean(0)
        self.phi_history = []
        self.theta = sp.array(self.theta_history).mean(0)
        self.theta_history = []

    def _update_biterm_topic(self, biterm):
        wi = biterm.wi
        wj = biterm.wj
        k = biterm.z
        p_z = (self.n_z + self.alpha) * (self.n_zw[:,wi] + self.beta) * (self.n_zw[:,wj] + self.beta) / ( (self.n_zw.sum(1) + self.n_words*self.beta)**2 )
        p_z /= p_z.sum()
        if self._assert_pz(p_z):
            print( "n_z={}".format(self.n_z) )
            print( "n_zw[:,wi]={}".format(self.n_zw[:,wi]) )
            print( "n_zw[:,wj]={}".format(self.n_zw[:,wj]) )
            print( "n_zw.sum(1)={}".format(self.n_zw.sum(1)) )
            print( "p_z={}".format(p_z) )
        new_k = sp.random.multinomial( 1, p_z ).argmax()
        new_biterm = Biterm(wi, wj, new_k)
        return new_biterm

    def _compute_parameters(self):
        self.phi = (self.n_zw + self.beta) / (self.n_zw.sum(1)[:,sp.newaxis] + self.n_words*self.beta)
        self.theta = (self.n_z + self.alpha) / (self.B + self.n_topics*self.alpha)

    def get_topic_word_dist(self):
        return self.phi

    def get_topic_dist(self):
        return self.theta

    def compute_document_topic_dist(self, document):
        """ compute topic distribution over one document
        @param document bag-of-words of a document
        @return topic distribution of document (theta_d)

        文書単位のトピック尤度の計算は，p(z|d)をb(biterm)で周辺化することで行う． 
        p(z|d) = \sum p(z|b)p(b|d)

        ここで，p(z|b) = p(z)p(wi|z)p(wj|z) / ( \sum p(z)p(wi|z)p(wj|z) )
        (ベイズの定理による)
        p(z) = theta, p(w|z) = phi(w,z)である．

        p(b|d) = n_d(b) / \sum n_d(b)
        """
        doc_all_biterms = self._gen_biterms_from_one_document(document)
        doc_biterm_set = self._make_biterm_set(doc_all_biterms)

        p_zd = 0.0
        if len(doc_biterm_set) > 0:
            n_db = sp.array([_b["count"] for _b in doc_biterm_set])
            n_db_sum = n_db.sum()

            for i,_b in enumerate(doc_biterm_set):
                p_bd = n_db[i] / float(n_db_sum)

                wi = _b["biterm"].wi
                wj = _b["biterm"].wj
                p_zb = self.theta * self.phi[:,wi] * self.phi[:,wj] / (self.theta * self.phi[:,wi] * self.phi[:,wj]).sum()
                p_zd += p_zb * p_bd
        else:
            exist_word_idxs = sp.where(document > 0)[0]
            for wi in exist_word_idxs:
                p_zd += self.theta * self.phi[:,wi] / (self.theta * self.phi[:,wi]).sum()

        return p_zd

    def save(self, dirname):
        import json
        params = {"alpha": self.alpha, "beta": self.beta, "n_topics": self.n_topics, "n_docs": self.n_docs, "n_words": self.n_words, "B": self.B}
        json.dump(params, open( "{}/params.json".format(dirname), "wb" ) )
        sp.save("{}/phi.npy".format(dirname), self.phi)
        sp.save("{}/theta.npy".format(dirname), self.theta)

    def load(self, dirname):
        import json
        params = json.load( open( "{}/params.json".format(dirname) ) )
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.n_topics = params["n_topics"]
        self.n_docs = params["n_docs"]
        self.n_words = params["n_words"]
        self.B = params["B"]
        self.phi = sp.load( "{}/phi.npy".format(dirname) )
        self.theta = sp.load( "{}/theta.npy".format(dirname) )

    def _push_value_to_history(self, history, val):
        if len(history) < self.history_size:
            history.append(val)
        else:
            history.pop(0)
            history.append(val)

    def _assert_pz(self, p_z):
        lower_judge = (p_z >= 0.0).all()
        upper_judge = (p_z <= 1.0).all()
        if lower_judge and upper_judge:
            return False
        else:
            return True
