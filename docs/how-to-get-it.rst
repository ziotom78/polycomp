How to get the latest Polycomp version
======================================

To obtain the latest version of Polycomp, first be sure to have an
updated version of the ``libpolycomp`` package. It is available at the
link https://github.com/ziotom78/libpolycomp.

The source code for Polycomp is available at
https://github.com/ziotom78/polycomp: download and install it using
``git`` (https://git-scm.com/). Refer to that page for any prerequisite.
The typical sequence of commands used to install ``polycomp`` is the
following::

  git clone https://github.com/ziotom78/polycomp
  cd polycomp
  python setup.py build

If you want to install ``polycomp`` and the Python bindings to
``libpolycomp``, you must run the command ``install``::

  python setup.py install

either as a super-user or using ``sudo``.
