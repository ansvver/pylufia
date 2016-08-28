# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

if os.name == "nt":
    extensions = [Extension('*', ['*.pyx'],
                    extra_compile_args = ['-O3'],
                    # extra_link_args=[
                    #     "/LIBPATH:C:\Users\dan_sasai\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\VC\lib\\amd64", 
                    #     "/LIBPATH:C:\Users\san_sasai\AppData\Local\Programs\Common\Microsoft\Visual C++ for Python\9.0\WinSDK\Lib\\x64"
                    # ],
                    include_dirs = [numpy.get_include()]
    )]
else:
    extensions = [Extension('*', ['*.pyx'],
                    extra_compile_args = ['-O3'],
                    include_dirs = [numpy.get_include()]
    )]

setup(
    ext_modules = cythonize(extensions, language='c++')
    )