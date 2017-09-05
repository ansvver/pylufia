# -*- coding: utf-8 -*-

"""
@package tokenizer.py
@brief sentence tokenizer
@author stfate
"""

import MeCab


class Tokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Ochasen")
        self.tagger.parse("")

    def tokenize(self, sentence, normalize=False):
        tokens = []
        node = self.tagger.parseToNode(sentence)
        while node:
            surface = node.surface
            feature = node.feature
            pos = feature.split(",")[0]
            if normalize:
                token = feature.split(",")[6]
            else:
                token = surface
            if pos != "BOS/EOS":
                tokens.append(token)
            node = node.next 
        return tokens
