#!/usr/bin/env python
# -*- mode: python -*-

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

modules = [Extension("pypolycomp",
                     sources=["pypolycomp.pyx"],
                     libraries=["polycomp"])]

setup(name="pypolycomp",
      ext_modules=cythonize(modules))
