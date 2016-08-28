# -*- coding: utf-8 -*-

"""
====================================================================
Common function for audioio
====================================================================
"""

import scipy as sp


def stereo_to_mono(input):
    """
    Mix stereo wave data into monoral wave data
    
    Parameters:
      dataIn: ndarray
        input wave data
      
    Returns:
      result: ndarray
        monoral downmixed wave data
    """
    # if (len(input) != 2):
    #     return input
    # else:
    #     # output = input.mean(0).astype("int16")
    #     output = input.mean(0)
    #     return output
    output = input.mean(0)
    return output
