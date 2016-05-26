# -*- coding: utf-8 -*-

"""
@file text_reader.py
@brief
@author ふぇいと (@stfate)

@description

"""

def loadtxt(fname):
    txtdata = u""

    with open(fname, "r") as fid:
        for line in fid:
            txtdata += line.decode("utf-8")
    txtdata = txtdata.encode("utf-8")

    return txtdata
