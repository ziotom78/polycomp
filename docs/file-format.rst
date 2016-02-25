Polycomp file format
====================

We present here a detailed description of the compressed files written
by Polycomp.

Polycomp files are FITS files where each 1-D datastream is saved in
its own HDU. A number of FITS keyword defined here are used to store
details about the datastream, such as the compression algorithm and
other information needed for the decompression. It is possible to
access Polycomp files using any FITS library. Notable examples are
cfitsio and Astropy (through the ``io.fits`` module).

Keywords
--------

Any Polycomp file contains one or more binary table HDUs. Each HDU
contains the data required to decompress one 1-D table. The order of
the HDUs is the same as the order specified in the Polycomp
configuration file used for the compression.

The following keyword are defined in each of the table HDUs:

==================== =========== ============================================
Keyword              Type        Meaning
==================== =========== ============================================
``PCSRCTP``          String      NumPy type of the input data
``PCCOMPR``          String      Kind of compression algorithm used
``PCNUMSA``          Integer     Number of uncompressed samples
``PCNUMSA``          Integer     Number of uncompressed samples
``PCUNCSZ``          Integer     Size of the uncompressed samples, in bytes
``PCCOMSZ``          Integer     Size of the compressed samples, in bytes
``PCTIME``           Float       Time needed to compress the data
``PCCR``             Float       Compression ratio
==================== =========== ============================================

The allowed strings for ``PCCOMPR`` are the following:

- ``none``: no compression;
- ``rle``: Run-Length Encoding;
- ``diffrle``: Differenced Run-Length Encoding;
- ``quantization``: Floating-point quantization;
- ``polynomial``: Polynomial compression (with or without the
  Chebyshev step);
- ``zlib``: zlib-based compression;
- ``bzip2``: bzip2-based compression.

If the algorithm is ``quantization``, the following keywords are saved
in the HDU header as well:

==================== =========== ============================================
Keyword              Type        Meaning
==================== =========== ============================================
``PCELEMSZ``         Integer     Number of bits per uncompressed sample
``PCBITSPS``         Integer     Number of bits per compressed sample
``PCNORM``           Float       Normalization factor
``PCOFS``            Float       Offset
==================== =========== ============================================

Table data
----------

Table data is saved in one fixed-size column (with one notable exception,
see below). The type of this column depends on the input data type
and/or the compression algorithm:

================= ========================================
Compression       Column type
================= ========================================
``none``          Same as input data
``rle``           Same as input data
``diffrle``       Same as input data
``quantization``  8-bit integer
``polynomial``    8-bit integer (but see below)
``zlib``          8-bit integer
``bzip2``         8-bit integer
================= ========================================


Debug-mode for polynomial compression
-------------------------------------

Due to its relative complexity, polynomial compression can be saved
using a special debug mode. In this mode, instead of coding the
compressed stream as a raw sequence of 8-bit integers, the compressor
saves information about each chunk separately. This mode is not as
efficient as the default mode, but it allows to understand the
compressor's performance more easily.

If a data stream is compressed using polynomial compression in debug
mode, the keyword ``PCDEBUG`` in the HDU header is set to 1, and the
following columns are saved:

=========== ============== ============ ======================================================
Name        Type           Row size     Description
=========== ============== ============ ======================================================
``ISCOMPR`` Logical        1            True if this chunk has been compressed
``CKLEN``   Integer        1            Length of this chunk
``UNCOMPR`` Double         Variable     If ``ISCOMPR`` is False, the uncompressed samples
``POLY``    Double         Variable     If ``ISCOMPR`` is True, the polynomial coefficients
``CHMASK``  8-bit integer  Variable     If ``ISCOMPR`` is True, the Chebyshev bit mask
``CHEBY``   Double         Variable     If ``ISCOMPR`` is True, the Chebyshev coefficients
=========== ============== ============ ======================================================

The length of the rows in variable-length columns is the following:

- ``UNCOMPR``: a number of elements equal to ``CKLEN``;
- ``POLY``: equal to :math:`\deg p(x) + 1`;
- ``CHMASK``: equal to :math:`\lceil N + 1\rceil`, where :math:`N` is ``CKLEN``;
- ``CHEBY``: equal to the number of bits in ``CHMASK`` that are equal
  to 1.
