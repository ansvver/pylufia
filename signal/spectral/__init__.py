# -*- coding: utf-8 -*-

"""
====================================================================
signal __init__.py
====================================================================
"""

from .fft import *
from .stft import *
from .cepstrum import *
from .cqt import *
from .cqt_oct import *

# import numpy
# import os
# import pyximport

# if os.name == 'nt':
#     if os.environ.has_key('CPATH'):
#         os.environ['CPATH'] = os.environ['CPATH'] + ';' + numpy.get_include()
#     else:
#         os.environ['CPATH'] = numpy.get_include()

#     # XXX: we're assuming that MinGW is installed in C:\MinGW (default)
#     # if os.environ.has_key('PATH'):
#     #     os.environ['PATH'] = os.environ['PATH'] + ';C:\MinGW\bin'
#     # else:
#     #     os.environ['PATH'] = 'C:\MinGW\bin'

#     # mingw_setup_args = { 'options': { 'build_ext': { 'compiler': 'mingw32' } } }
#     # pyximport.install(setup_args=mingw_setup_args)
#     msvc_setup_args = {
#         'options': {
#             'build_ext': {
#                 'compiler': 'msvc',
#                 'include_dirs': numpy.get_include()
#             }
#         }
#     }
#     pyximport.install(setup_args=msvc_setup_args)

# elif os.name == 'posix':
#     if os.environ.has_key('CFLAGS'):
#         os.environ['CFLAGS'] = os.environ['CFLAGS'] + ' -I' + numpy.get_include()
#     else:
#         os.environ['CFLAGS'] = ' -I' + numpy.get_include()

#     pyximport.install()
    
from .cqt_cy import *
