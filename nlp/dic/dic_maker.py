# -*- coding: utf-8 -*-

"""
@file dic_maker.py
@brief dictionary maker
@author ふぇいと (@stfate)

@description
テキストファイル集合からbag-of-words構築用辞書を作成する
"""

import pylufia.nlp.textio as textio
import pylufia.nlp.feature as feature
import re


ptn_stop_words = re.compile(ur"[０-９]|する|ある|いる|これ|あれ|それ|どれ|いう|間奏|れる|てる|なる|よう|さ|せる|の|く|ん|こ|\*|％|・|月|日|こと|もの|ため")

def make_word_dic(filelist, part="all"):
    word_dic = {}
    id = 0
    for fn in filelist:
        print fn
        input_txt = textio.loadtxt(fn)
        keywords,parts = feature.extract_keywords(input_txt, part)

        for w,p in zip(keywords,parts):
            # if w not in stop_words:
            if not ptn_stop_words.search(w):
                w = w.encode("utf-8")
                p = p.encode("utf-8")
                if w not in word_dic:
                    word_dic[w] = {"id": id, "part": p}
                    id += 1

    return word_dic
