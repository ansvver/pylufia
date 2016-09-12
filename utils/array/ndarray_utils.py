# -*- coding: utf-8 -*-

"""
@package
@brief
@author Dan SASAI (YCJ,RDD)
"""

import scipy as sp
import struct


def write_3darray(fn, arr):
    """ write 3d-array to ascii file
    @param 
    @return
    """
    with open(fn, "w") as fo:
        for w0 in range(arr.shape[0]):
            fo.write( "{}\n".format(w0) )
            for w1 in range(arr.shape[1]):
                for w2 in range(arr.shape[2]):
                    fo.write( "{} ".format(arr[w0,w1,w2]) )
                fo.write("\n")
            fo.write("\n")

def write_matrix_binary(fn, X, dtype="int"):
    """ write 2darray to binary file
    @param
    @return
    """
    with open(fn, "wb") as fo:
        outdata = ""
        if dtype == "int":
            fmt = "i"
        elif dtype == "short":
            fmt = "h"
        elif dtype == "float":
            fmt = "f"
        elif dtype == "double":
            fmt = "d"
        else:
            fmt = "i"
        n_rows,n_cols = X.shape
        outdata += struct.pack("i", n_rows)
        outdata += struct.pack("i", n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                outdata += struct.pack(fmt, X[r,c])
        fo.write(outdata)

def read_matrix_binary(fn, dtype="int"):
    """ read 2darray from binary file
    @param
    @return
    """
    with open(fn, "rb") as fi:
        if dtype == "int":
            fmt = "i"
            data_bytes = 4
        elif dtype == "short":
            fmt = "h"
            data_bytes = 2
        elif dtype == "float":
            fmt = "f"
            data_bytes = 4
        elif dtype == "double":
            fmt = "d"
            data_bytes = 8
        else:
            fmt = "i"
            data_bytes = 4

        n_rows = struct.unpack("i", fi.read(4) )[0]
        n_cols = struct.unpack("i", fi.read(4) )[0]
        X = sp.zeros( (n_rows,n_cols) )
        for r in range(n_rows):
            for c in range(n_cols):
                X[r,c] = struct.unpack( fmt, fi.read(data_bytes) )[0]
        return X

def write_vector_binary(fn, X, dtype="int"):
    """ write 1darray to binary file
    @param
    @return
    """
    with open(fn, "wb") as fo:
        outdata = ""
        if dtype == "int":
            fmt = "i"
        elif dtype == "short":
            fmt = "h"
        elif dtype == "float":
            fmt = "f"
        elif dtype == "double":
            fmt = "d"
        else:
            fmt = "i"
        length = len(X)
        outdata += struct.pack("i", length)
        for i in range(length):
            outdata += struct.pack(fmt, X[i])
        fo.write(outdata)

def read_vector_binary(fn, dtype="int"):
    """ read 1darray from binary file
    @param
    @return
    """
    with open(fn, "rb") as fi:
        if dtype == "int":
            fmt = "i"
            data_bytes = 4
        elif dtype == "short":
            fmt = "h"
            data_bytes = 2
        elif dtype == "float":
            fmt = "f"
            data_bytes = 4
        elif dtype == "double":
            fmt = "d"
            data_bytes = 8
        else:
            fmt = "i"
            data_bytes = 4

        length = struct.unpack("i", fi.read(4) )[0]
        X = sp.zeros(length)
        for i in range(length):
            X[i] = struct.unpack( fmt, fi.read(data_bytes) )[0]
        return X