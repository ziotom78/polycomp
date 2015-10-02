#!/usr/bin/env python
# -*- mode: python -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

modules = [Extension("pypolycomp",
                     sources=["pypolycomp.pyx"],
                     libraries=["polycomp"])]

setup(name="pypolycomp",
      version="1.0",
      author="Maurizio Tomasi",
      author_email="ziotom78@gmail.com",
      description="Python bindings to the libpolycomp C library",
      license="MIT",
      url="",
      ext_modules=cythonize(modules),
      packages=find_packages())
