Writing a configuration file
============================

In this section we describe how to write a configuration file which
tells Polycomp which data to compress, and how.

Note for Python developers: Polycomp configuration files are parsed
using Python's `configparse` library. Refer to its documentation for
more information about all the facilities provided by this format.

Basic syntax
------------

Empty lines and lines starting with ``#`` are ignored.

Polycomp configuration files are divided in sections, each marked by
its name enclosed in square brackets, like in the following example::

  [section 1]
  # Here are the contents of "section 1"

  [section 2]
  # Et cetera

Within each section, the file is expected to contain a sequence of
key/value pairs, as in the following example::

  cmb_temperature = 2.7
  speed_of_light = 3.0e8

Strings must be specified without single/double quotes::

  file_path = /data/my_test_data.fits


Specifying which data to compress
---------------------------------

Every configuration file must at least contain one section named
``polycomp``. This section should contain the key ``tables``, which
specifies a comma-separated list of tables that will be included in
the compressed ``.pcomp`` file generated by the program. For every
table specified here there must be a section (either before or after
this one) with the same name, where details about the table are to be
provided.

Here is an example::

  [polycomp]
  tables = time, temperature, pressure

  [time]
  # Here we specify where to take time data, and how to compress them

  [temperature]
  # Ditto for the temperature...

  [pressure]
  # ...and for the pressure

In the next sections we are going to explain what should every "table
section" contain.

Where to take the data to compress
----------------------------------

In each table section the following key/value pairs must be present:

- The `file` key specifies the path to the FITS file containing the
  data to be compressed.
- The `hdu` key specifies the number/name of the HDU within the FITS
  file (the first HDU is 1).
- The `column` key specifies the number/name of the column in the HDU
  (the first column is 1).

Here is an example::

  file = /opt/data/experiment_1.fits
  hdu = 1
  column = TIME

How to compress the data
------------------------

The `compression` key must be present in each table section. It
contains a string which identifies the compression algorithm to use,
and it can be one of the following:

================ ======================================
Value            Algorithm
================ ======================================
``none``         No compression
``rle``          Run-Length Encoding (RLE)
``diffrle``      Differential RLE
``quantization`` Quantization of floating-point values
``polynomial``   Polynomial compression
``zlib``         Zlib-based compression
``bzip2``        Bzip2-based compression
================ ======================================

The ``none``, ``rle`` and ``diffrle`` compression algorithms do not
require other key/value pairs. For all the other cases, they are
presented in the next subsections.

Quantization parameters
.......................

The only parameter required for this kind of compression is
``bits_per_sample``, which specifies the number of bits to be used
with each sample. Typical values are integers less than 32 or 64 bits,
depending on the width of the floating-point type used in the input
data.

Polynomial compression parameters
.................................

There are a number of key/value pairs that are understood when using
this algorithm. Not every pair is required; in a handful of cases,
Polycomp can provide a default value.

============================== =====================================================================
Key                            Default value
============================== =====================================================================
``num_of_coefficients``        (Required)
``samples_per_chunk``          (Required)
``max_error``                  (Required)
``use_chebyshev``              ``True``
``period``                     If not specified, the input data will be assumed not to be periodic.
``no_smart_optimization``      False
``opt_delta_coeffs``           1
``opt_delta_samples``          1
``opt_max_num_of_iterations``  0 (no upper limit)
============================== =====================================================================

The meaning of the parameters is the following:

- ``num_of_coefficients`` specifies the number of coefficients of the
  fitting polynomial, i.e., ``deg p(x) + 1``. The best value for this
  parameter depends heavily on the input data to be compressed.
- ``samples_per_chunk`` specifies the number of samples in each chunk.
  This number must always be greater than ``num_of_coefficients``.
- If ``use_chebyshev`` is set to ``False``, the so-called "simple
  compression algorithm" will be used. In some situations the code
  might run faster, but it can produce significantly worse compression
  ratios.
- If the input data are periodic, ``period`` should be set to their
  period (e.g., 2π for angles measured in radians). The default is not
  to assume the input data periodic.

The remaining parameters (``no_smart_optimization``,
``opt_delta_coeffs``, ``opt_delta_samples``, and
``opt_max_num_of_iterations``) are used when the user wants to search
the best possible configuration for the polynomial compressor.

Zlib/Bzip2 parameters
.....................
