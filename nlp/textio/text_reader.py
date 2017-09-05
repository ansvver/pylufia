# -*- coding: utf-8 -*-

"""
@file text_reader.py
@brief 
@author stfate

@description

"""

class TextReader:
    def __init__(self, fn=None, encoding="utf-8"):
        if fn:
            self.open(fn, encoding)

    def open(self, fn, encoding="utf-8"):
        self.fi = open(fn, "r", encoding=encoding)

    def read(self):
        for _l in self.fi:
            yield _l

    def get_read_iterator(self):
        def read_iterator():
            for line in self.fi:
                yield line
        
        return read_iterator
