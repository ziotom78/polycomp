========
Polycomp
========

This package provides a set of Python bindings to the libpolycomp
library, as well as a stand-alone program which can be used to
compress/decompress FITS files into polycomp files (still FITS files
in disguise).

Requirements
------------

1. Either Python 2.7 or 3.4 will work

2. Libpolycomp (https://github.com/ziotom78/libpolycomp) must already
   be installed

3. The following Python libraries are required:
   - `docopt`;
   - `pyfits`.

Basic usage
-----------

To use the bindings in your code, simply import ``pypolycomp`` in your
project::

  import pypolycomp

This package provides also a standalone program, ``polycomp``. Use the
``--help`` flag to get some help about how to use it::

  $ polycomp --help
