# -*- coding: utf-8 -*-

"""
@file text_reader.py
@brief 
@author ふぇいと (@stfate)

@description

"""

class TextReader:
    def __init__(self, fn, encoding="utf-8"):
        self.open(fn, encoding)

    def open(self, fn, encoding="utf-8"):
        self.fi = open(fn, "r", encoding=encoding)

    def read(self):
        for _l in self.fi:
            yield _l
