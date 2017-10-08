Polycomp
========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

This package provides a set of Python bindings to the libpolycomp library
(https://github.com/ziotom78/libpolycomp), as well as a stand-alone program
which can be used to compress/decompress FITS files into polycomp files (still
FITS files in disguise).

This code has been indexed in the `Astrophysics Source Code Library
<http://ascl.net>`_ as `[ascl:1604.002] <http://ascl.net/1604.002>`_.

The paper *Polycomp: efficient and configurable compression of
astronomical timelines* has been accepted by the Journal "Astronomy &
Computing" (http://dx.doi.org/10.1016/j.ascom.2016.04.004). The
preprint is available on `arXiv <http://arxiv.org/abs/1604.07980v1>`_.

Requirements
------------

1. Either Python 2.7 or 3.4 will work

2. Libpolycomp (https://github.com/ziotom78/libpolycomp) must already
   be installed

3. The following Python libraries are required:

   - `click` (http://click.pocoo.org/5/);
   - `numpy` (http://www.numpy.org/);
   - Either `astropy` (version 0.4 or greater, http://www.astropy.org/) or
     `pyfits` (http://www.stsci.edu/institute/software_hardware/pyfits). (The
     `setup.py` script installs Astropy.);
   - Cython (http://cython.org/).  

**Note**: Recent versions of NumPy (1.10) seem to be incompatible with PyFits
3.3: saving files will lead to strange assertion errors. NumPy 1.8.2 and PyFits
3.3 are fine. Or better, install AstroPy.


Installation
------------

Once you have installed all the requirements, run the following command to install
the library::

    pip install .
    
You might have to make Python aware of the directory where `libpolycomp.so` was
installed, if this is a non-standard location (this might include `/usr/local/lib`,
which is the default path used by Libpolycomp when running `make install`).
Just set the `LD_LIBRARY_PATH` variable to the correct path::

    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH


Basic usage
-----------

To use the bindings in your code, simply import ``pypolycomp`` in your
project::

    import pypolycomp

This package provides also a standalone program, ``polycomp``. Use the
``--help`` flag to get some help about how to use it::

    $ polycomp --help

How to cite this library
------------------------

If you use this library in your work, please cite the paper `Polycomp:
efficient and configurable compression of astronomical timelines`
(http://dx.doi.org/10.1016/j.ascom.2016.04.004)::

        @article{Tomasi201688,
                title = "Polycomp: Efficient and configurable compression of astronomical timelines",
                journal = "Astronomy and Computing",
                volume = "16",
                number = "",
                pages = "88-98",
                year = "2016",
                issn = "2213-1337",
                doi = "http://dx.doi.org/10.1016/j.ascom.2016.04.004",
                author = "M. Tomasi",
        }

License
-------

This code is released under the MIT license.
