#!/usr/bin/env python3
# -*- mode: python -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import os.path as path

modules = [Extension("pypolycomp._bindings",
                     sources=["pypolycomp/_bindings.pyx"],
                     libraries=["polycomp"])]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(name="polycomp",
      version="1.0",
      author="Maurizio Tomasi",
      author_email="ziotom78@gmail.com",
      description="Python bindings to the libpolycomp C library",
      long_description=long_description,
      license="MIT",
      url="",
      install_requires=["cython", "pytoml", "docopt"],
      ext_modules=cythonize(modules),
      scripts=['polycomp'],
      packages=['pypolycomp'],
      keywords='compression astronomy fits',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: System :: Archiving :: Compression',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
      ])
