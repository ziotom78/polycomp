#!/usr/bin/env python
# -*- mode: python -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

modules = [Extension("pypolycomp",
                     sources=["pypolycomp.pyx"],
                     libraries=["polycomp"])]

setup(name="pypolycomp",
      ext_modules=cythonize(modules))
