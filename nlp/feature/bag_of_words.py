# -*- coding: utf-8 -*-

"""
@file bag_of_words.py
@brief bag-of-words extractor
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import MeCab
import re


def extract_keywords(txtdata, part='all'):
    """
    Extract keywords from text data
    """
    if part == 'all':
        ptn = re.compile(r'.+')
    elif part == 'nva':
        ptn = re.compile(r'名詞|動詞|形容詞')
    elif part == 'nv':
        ptn = re.compile(r'名詞|動詞')
    elif part == 'va':
        ptn = re.compile(r'動詞|形容詞')
    elif part == 'v':
        ptn = re.compile(r'動詞')
    elif part == 'n':
        ptn = re.compile(r'名詞')

    mecab = MeCab.Tagger('-Ochasen')
    node = mecab.parseToNode(txtdata)
    keywords = []
    speech_parts = []
    while node:
        speech_part = node.feature.split(',')[0]
        if ptn.match(speech_part):
            word = node.feature.split(',')[6]
            u_word = word.decode('utf-8')
            u_speech_part = speech_part.decode('utf-8')
            if len(u_word) == 1 and re.match(r'[ぁ-ん]|[ァ-ン]', u_word):
                print len(u_word)
                print u_word.encode('cp932')
            else:
                keywords.append(u_word)
                speech_parts.append(u_speech_part)
        node = node.next

    return keywords,speech_parts

def bag_of_words(words, dictionary, count=True):
    """
    Compute Bag-of-Words from word list and dictionary
    """
    n_feature_words = len( dictionary.keys() )

    BOW = sp.zeros(n_feature_words, dtype=int)
    for word in words:
        if word in dictionary.keys():
            if count:
                BOW[ dictionary[word]["id"] ] += 1
            else:
                BOW[ dictionary[word]["id"] ] = 1

    return BOW

def bag_of_words_from_doc(fname_doc, word_dict, mode='all', count=True):
    """
    Compute Bag-of-Words from a document file
    """
    txtdata = loadtxt(fname_doc)
    keywords = extract_keywords(txtdata, mode)
    bow = bag_of_words(keywords, word_dict, count=count)

    return bow

def sequence_of_words(fname_doc, dictionary):
    """
    Compute Sequence-of-Words from word list and dictionary
    """
    txtdata = loadtxt(fname_doc)
    txtdata = txtdata.encode("utf-8")
    words = extract_keyword(txtdata, "all")

    SOW = []
    for i,word in enumerate(words):
        print word
        if word in dictionary.keys():
            SOW.append(dictionary[word]["id"])
    SOW = sp.array(SOW)

    return SOW
