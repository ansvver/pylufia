# -*- coding: utf-8 -*-

"""
@file text_reader.py
@brief
@author ふぇいと (@stfate)

@description

"""

def loadtxt(fname):
    txtdata = ""

    with open(fname, "r", encoding="utf-8") as fid:
        for line in fid:
            # txtdata += line.decode("utf-8")
            txtdata += line
    # txtdata = txtdata.encode("utf-8")

    return txtdata
