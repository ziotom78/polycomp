#!/usr/bin/env python3
# -*- mode: python -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
import os.path as path
from distutils.version import LooseVersion as Version
import numpy as np

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'pypolycomp', 'version.py'), encoding='utf-8') as f:
    exec(f.read())

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    from Cython import __version__ as cython_version
except ImportError:
    use_cython = False
else:
    use_cython = Version(cython_version) >= Version('0.18.0')


if use_cython:
    print('using cython')
    cython_ext = '.pyx'
else:
    print('NOT using cython')
    cython_ext = '.c'

modules = [Extension("pypolycomp._bindings",
                     sources=["pypolycomp/_bindings" + cython_ext],
                     libraries=["polycomp"])]


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

if use_cython:
    modules = cythonize(modules)

setup(name="polycomp",
      version=__version__,
      author="Maurizio Tomasi",
      author_email="ziotom78@gmail.com",
      description="Python bindings to the libpolycomp C library",
      long_description=long_description,
      license="MIT",
      url="https://github.com/ziotom78/polycomp",
      install_requires=["cython >= 0.18", "numpy >= 1.8.2", "astropy >= 0.4", "click"],
      include_dirs=[np.get_include()],
      ext_modules=modules,
      scripts=['polycomp.py'],
      packages=['pypolycomp'],
      keywords='compression astronomy fits',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: System :: Archiving :: Compression',
          'License :: OSI Approved :: MIT License',
          'Operating System :: POSIX',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
      ])
