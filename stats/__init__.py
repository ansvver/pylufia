# -*- coding: utf-8 -*-

"""
@package stats
"""

from .prob_dists import *

# import pyximport
# import os
# import numpy
# mingw_setup_args={'options': {'build_ext': {'compiler': 'mingw32'}}}
# pyximport.install(setup_args=mingw_setup_args)
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
#     if os.environ.has_key('PATH'):
#         os.environ['PATH'] = os.environ['PATH'] + numpy.get_include()
#     else:
#         os.environ['PATH'] = numpy.get_include()

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

from .prob_dists_cy import *
