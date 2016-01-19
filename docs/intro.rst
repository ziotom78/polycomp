An introduction to Polycomp
===========================

``polycomp`` is a Python program that relies on the `libpolycomp`
library (https://github.com/ziotom78/libpolycomp) to compress
one-dimensional datasets in FITS format. It has been optimized for
datasets produced by astronomical experiments where large FITS tables
store time-ordered measurements.

The program takes as input tables in FITS files and assembles them
into files with extension ``.pcomp``. The latters are FITS files in
disguise, where each table taken from the input files is compressed
using one of the following compression schemas:

Run-length encoding
   This works extremely well for columns containing long sequences of
   repeated integer values, such as quality flags.

Differenced run-length encoding
   This is a variant of the preceding scheme, and it works with
   integer numbers which increase regularly. A typical example is the
   time recorded by a digital clock.

Quantization
   This lossy compression scheme reduces the number of bits used for
   storing floating-point numbers. It is useful for compressing
   sequences of numbers measured by digital instruments, if the number
   of bits of the raw measurement is significantly lower than the
   standard size of floating-point numbers (32-bit/64-bit).

Polynomial compression
   This compression works for smooth, noiseless data. It is a lossy
   compression scheme. The program allows to set an upper bound to the
   compression error.

Zlib
   This widely used compression is implemented using the ``zlib``
   library (http://www.zlib.net/).

Bzip2
   Another widely used compression scheme, implemented using the
   ``bzip2`` library (http://www.bzip.org/).

``polycomp`` is a command-line program which can either compress or
decompress data. In the following sections we illustrate how to
compress a set of FITS files into one single ``.pcomp`` file, and then
we show how to decompress it.

Compressing files using polycomp
--------------------------------

To compress tables from one or more FITS files, the user must prepare
a `configuration file` which specifies where to look for the tables
and which kind of compression apply to each of them. Such
configuration files are parsed using the widely used ``configparse``
Python library: the syntax of such files is described in the standard
Python documentation, available at
https://docs.python.org/3/library/configparser.html.

Let's consider as an example how to compress the TOI files available
on the Planck Legacy Archive
(http://www.cosmos.esa.int/web/planck/pla). Such TOIs contain the
time-ordered information acquired by the two instruments onboard the
spacecraft, LFI and HFI. In order to cut download times, we consider
the smallest TOIs in the database: those produced by the radiometer
named LFI-27M. There are more than one kind of TOIs: we concentrate on
the two most useful ones, the differenced TOIs (containing the actual
measurements of the radiometer) and the pointing TOIs (containing the
direction of view as a function of time).

Download the TOIs at the following addresses:
   - http://pla.esac.esa.int/pla-sl/data-action?TIMELINE.TIMELINE_OID=36063
     (differenced scientific data, 96 MB, save it into file ``sci.fits``)
   - http://pla.esac.esa.int/pla-sl/data-action?TIMELINE.TIMELINE_OID=2062650
     (pointings, 223 MB, save it into file ``ptg.fits``)

Such TOIs contain the time-ordered information for all the 30 GHz
radiometers (LFI-27M, LFI-27S, LFI-28M, LFI-28S): each HDU contains
information about just one radiometer. We are interested only in
LFI-27M, so we must tell ``polycomp`` which data to extract from both
files.

Create a text file named ``pcomp_LFI27M.conf`` with the following
content::

  [polycomp]
  tables = obt_time, theta, phi, psi, diff, flags

  [obt_time]
  file = sci.fits
  hdu = 1
  column = OBT
  compression = diffrle
  datatype = int64

  [theta]
  file = ptg.fits
  hdu = LFI27M
  column = THETA
  compression = polynomial
  num_of_coefficients = 8
  samples_per_chunk = 80
  max_error = 4.8e-6
  use_chebyshev = True

  [phi]
  file = ptg.fits
  hdu = LFI27M
  column = PHI
  compression = polynomial
  num_of_coefficients = 8
  samples_per_chunk = 80
  max_error = 4.8e-6
  use_chebyshev = True

  [psi]
  file = ptg.fits
  hdu = LFI27M
  column = PSI
  compression = polynomial
  num_of_coefficients = 8
  samples_per_chunk = 80
  max_error = 4.8e-6
  use_chebyshev = True

  [diff]
  file = sci.fits
  hdu = LFI27M
  column = LFI27M
  compression = quantization
  bits_per_sample = 20

  [flags]
  file = sci.fits
  hdu = LFI27M
  column = FLAG
  compression = rle
  datatype = int8

This file describes the way input data will be compressed by
``polycomp``. Run the program with the following syntax::

  polycomp.py compress pcomp_LFI27M.conf compressed.pcomp

This command will produce a file named ``compressed.pcomp``, which
contains the six compressed columns of data specified in the
configuration file. The file format used for ``compressed.pcomp`` is
based on the FITS standard, and you can therefore use any FITS
library/program to access its data. (Of course, to actually decompress
the data in it you must use ``libpolycomp``.)


Decompressing files using polycomp
----------------------------------

Decompression is considerably simpler than compression, as it does not
require to prepare a configuration file. You have to specify the input
``.pcomp`` file and the output FITS file, as in the following
example::

  polycomp.py decompress compressed.pcomp decompressed.fits

By default, ``polycomp`` will save every column of data in a separated
HDU file within ``decompressed.fits``. If all the columns in
``compressed.pcomp`` are of the same length, you can use the
``--one-hdu`` flag to save everything in one HDU::

  polycomp.py decompress --one-hdu compressed.pcomp decompressed.fits
