# -*- coding: utf-8 -*-

"""
@file common.py
@brief common functions for audioio
@author ふぇいと (@stfate)

@description

"""

import scipy as sp


def stereo_to_mono(input):
    """
    Mix stereo wave data into monoral wave data
    
    Parameters:
      input: ndarray
        input wave data
      
    Returns:
      result: ndarray
        monoral downmixed wave data
    """
    if (len(input) != 2):
        return input
    else:
        output = input.mean(0)
        return output
