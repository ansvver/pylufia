# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [Extension('*', ['*.pyx'],
                        extra_compile_args = ['-O3'],
                        include_dirs = [numpy.get_include()]
                        )
            ]

setup(
    ext_modules = cythonize(extensions, language='c++')
    )