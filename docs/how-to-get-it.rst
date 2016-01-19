How to get the latest Polycomp version
======================================

To obtain the latest version of Polycomp, first be sure to have an
updated version of the ``libpolycomp`` library. It is available at the
link https://github.com/ziotom78/libpolycomp.

The source code for Polycomp is available at
https://github.com/ziotom78/polycomp: download and install it using
``git`` (https://git-scm.com/). Refer to that page for any prerequisite.
The typical sequence of commands used to install ``polycomp`` is the
following::

  git clone https://github.com/ziotom78/polycomp
  cd polycomp
  python3 setup.py build

If you want to install ``polycomp`` and the Python bindings to
``libpolycomp``, you must run the command ``install``::

  python3 setup.py install

either as a super-user or using ``sudo``.

Requisites
----------

Polycomp requires the following tools:

- Python (both version 2 and 3 are fine);
- `click` (http://click.pocoo.org/5/), automatically installed by `setup.py`;
- `numpy` (http://www.numpy.org/), automatically installed by
  `setup.py`;
- Either `astropy` (version 0.4 or greater, http://www.astropy.org/,
  the preferred solution) or `pyfits`
  (http://www.stsci.edu/institute/software_hardware/pyfits). (The
  `setup.py` script installs Astropy.)
