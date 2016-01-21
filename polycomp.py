#!/usr/bin/env python3
# -*- mode: python -*-

"""Compress/decompress FITS files using polynomial compression and other
algorithms."""

from collections import namedtuple
from datetime import datetime
import click
import itertools
import numbers
import os.path
import sys
import time
import zlib
import bz2
import pypolycomp as ppc
import logging as log
import re
import numpy as np
import warnings
from pypolycomp import __version__

try:
    # Python 3
    import configparser
except ImportError:
    # Python 2
    import ConfigParser as configparser

try:
    import astropy.io.fits as pyfits
    from astropy import __version__ as fits_lib_version

    fits_lib = 'Astropy'
except ImportError:
    import pyfits

    fits_lib = 'PyFits'
    fits_lib_version = pyfits.__version__

########################################################################

def parse_intset(val):
    """Parse a list of integers.

    The integers must be comma-separated. Ranges in the form
    NN-MM or NN:MM are allowed: in this case, beware that the
    last element is *always* included (unlike Python). It is
    also possible to specify a step value through the syntax
    START:STEP:STOP. In the latter case, the last element is
    included whenever possible. See the examples below.

    Examples
    --------
    >>> parse_intset(10)
    [10]
    >>> parse_intset("10")
    [10]
    >>> parse_intset("10, 11, 14-16, 7, 5:2:15")
    [5, 7, 9, 10, 11, 13, 14, 15, 16]
    """

    if isinstance(val, numbers.Integral):
        return [val]
    elif isinstance(val, str):
        # Parse the string
        chunks = val.split(',')
        result = set()
        integer_re = re.compile('^[ \t]*\\d+[ \t]*$')
        range_step_re = re.compile('^[ \t]*(\\d+)[ \t]*:[ \t]*(\\d+)[ \t]*:[ \t]*(\\d+)[ \t]*$')
        range_re = re.compile('^[ \t]*(\\d+)[ \t]*[-:][ \t]*(\\d+)[ \t]*$')
        for chunk in chunks:
            if integer_re.match(chunk):
                result.add(int(chunk))
                continue

            range_step_match = range_step_re.match(chunk)
            if range_step_match:
                result = result.union(np.arange(start=int(range_step_match.group(1)),
                                                step=int(range_step_match.group(2)),
                                                stop=int(range_step_match.group(3)) + 1,
                                                dtype='int'))
                continue

            range_match = range_re.match(chunk)
            if range_match is None:
                raise ValueError('invalid value(s): "{0}"'
                                 .format(val))
            result = result.union(range(int(range_match.group(1)),
                                        int(range_match.group(2)) + 1))
        return sorted(list(result))
    else:
        raise ValueError('invalid value: {0}'.format(val))

########################################################################

def humanize_size(val):
    "Return a string representing 'val' using kB, MB... units"

    float_val = float(val)
    if float_val < 10 * 1024.0:
        return '{0} bytes'.format(val)
    elif float_val < 10 * 1024.0**2:
        return '{0:.1f} kB'.format(float_val / 1024.0)
    elif float_val < 10 * 1024.0**3:
        return '{0:.1f} MB'.format(float_val / (1024.0**2))
    elif float_val < 10 * 1024.0**4:
        return '{0:.1f} GB'.format(float_val / (1024.0**3))
    else:
        return '{0:.1f} TB'.format(float_val / (1024.0**4))

########################################################################

def shannon_entropy(values):
    """Compute Shannon's entropy for a set of samples

    The "values" variable must be a NumPy array of integer/floating
    point numbers. Such numbers will be rounded as 64-bit integers
    before the computation takes place."""

    bc = np.bincount(values.astype(np.int64))
    ii = np.nonzero(bc)[0]
    prob = bc[ii] / float(values.size)

    return np.sum(-prob * np.log2(prob))

########################################################################

def get_hdu_and_column_from_schema(parser, table, input_file):
    """Determine which HDU and column to read for the specified table.

    The "parser" variable is a ConfigurationParser object used to
    retrieve the requested information. The "table" variable is a
    string, and it indicates the section in the schema file.

    Return a (HDU, COLUMN) pair."""

    hdu = parser.get(table, 'hdu')
    column = parser.get(table, 'column')

    integer_re = re.compile('^[ \t]*[0-9]+[ \t]*$')
    if integer_re.match(hdu):
        hdu = int(hdu)
    else:
        hdu = input_file.index_of(hdu)

    if integer_re.match(column):
        column = int(column) - 1
    else:
        column = input_file[hdu].columns.names.index(column)

    if (hdu < 0) or (column < 0):
        raise ValueError('invalid HDU and column: ({0}, {1})'
                         .format(hdu, column))

    return (hdu, column)

########################################################################

def compress_and_encode_none(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (uncompressed form)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser
    del debug

    col = pyfits.Column(name=table, format=samples_format, array=samples)
    return (pyfits.BinTableHDU.from_columns([col]),
            samples.itemsize * samples.size)

########################################################################

def compress_and_encode_rle(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser
    del debug

    compr_samples = ppc.rle_compress(samples)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)]),
            compr_samples.itemsize * compr_samples.size)

########################################################################

def compress_and_encode_diffrle(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (differenced RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser
    del debug

    compr_samples = ppc.diffrle_compress(samples)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)]),
            compr_samples.itemsize * compr_samples.size)

########################################################################

def compress_and_encode_quant(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (quantization)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format
    del debug

    bits_per_sample = parser.getint(table, 'bits_per_sample')
    quant = ppc.QuantParams(element_size=samples.dtype.itemsize,
                            bits_per_sample=bits_per_sample)
    compr_samples = quant.compress(samples)

    hdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format='1B',
                      array=compr_samples)])
    hdu.header['PCBITSPS'] = (quant.bits_per_sample(),
                              'Number of bits per quantized sample')
    hdu.header['PCELEMSZ'] = (quant.element_size(),
                              'Number of bytes per sample')
    hdu.header['PCNORM'] = (quant.normalization(),
                            'Normalization factor')
    hdu.header['PCOFS'] = (quant.offset(),
                           'Offset factor')

    # Compute and store the entropy of the quantized samples (this is
    # useful to estimate how much an additional statistical encoding
    # stage would improve the compression ratio)
    amin, amax = [f(samples) for f in (np.amin, np.amax)]
    scaled_samples = (samples - amin) / (amax - amin) * (2**bits_per_sample)
    hdu.header['PCENTROP'] = (shannon_entropy(scaled_samples),
                              'Shannon''s entropy for the quantized data')

    return (hdu, compr_samples.size * compr_samples.itemsize)

########################################################################

def polycomp_chunks_to_FITS_table_debug(chunks, params):
    is_compressed = np.empty(len(chunks), dtype='bool')
    chunk_length = np.empty(len(chunks), dtype='uint64')
    uncompressed = np.empty(len(chunks), dtype=np.object)
    poly_coeffs = np.empty(len(chunks), dtype=np.object)
    cheby_mask = np.empty(len(chunks), dtype=np.object)
    cheby_coeffs = np.empty(len(chunks), dtype=np.object)

    for chunk_idx in range(len(chunks)):
        cur_chunk = chunks[chunk_idx]
        is_compressed[chunk_idx] = cur_chunk.is_compressed()
        chunk_length[chunk_idx] = cur_chunk.num_of_samples()
        uncompressed[chunk_idx] = cur_chunk.uncompressed_samples()
        poly_coeffs[chunk_idx] = cur_chunk.poly_coeffs()
        cheby_mask[chunk_idx] = cur_chunk.cheby_mask()
        cheby_coeffs[chunk_idx] = cur_chunk.cheby_coeffs()

    hdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name='ISCOMPR', format='1L', array=is_compressed),
        pyfits.Column(name='CKLEN', format='1I', array=chunk_length),
        pyfits.Column(name='UNCOMPR', format='PD()', array=uncompressed),
        pyfits.Column(name='POLY', format='PD()', array=poly_coeffs),
        pyfits.Column(name='CHMASK', format='PB()', array=cheby_mask),
        pyfits.Column(name='CHEBY', format='PD()', array=cheby_coeffs)])

    hdu.header['PCNPOLY'] = (params.num_of_poly_coeffs(),
                             '# of coeffs of the interpolating polynomial')
    hdu.header['PCSMPCNK'] = (params.samples_per_chunk(),
                              '# of samples in each chunk')
    hdu.header['PCALGOR'] = (params.algorithm(),
                             'Kind of polynomial compression ("algorithm")')
    hdu.header['PCMAXERR'] = (params.max_error(),
                              'Maximum compression error')
    hdu.header['PCNCOMPR'] = (np.sum(is_compressed.astype(np.uint8)),
                              '# of compressed chunks')
    hdu.header['PCNCHEBC'] = (len(np.concatenate(cheby_coeffs)),
                              '# of Chebyshev coeffs in the table')
    hdu.header['PCDEBUG'] = (1,
                             'This table uses the extended format')

    return hdu

########################################################################

def polycomp_chunks_to_FITS_table(chunks, params):

    raw_bytes = chunks.encode()
    hdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name='BYTES', format='1B', array=raw_bytes)])

    # Compute a few interesting quantities
    is_compressed = np.empty(len(chunks), dtype='bool')
    cheby_coeffs = np.empty(len(chunks), dtype=np.object)

    for chunk_idx in range(len(chunks)):
        cur_chunk = chunks[chunk_idx]
        is_compressed[chunk_idx] = cur_chunk.is_compressed()
        cheby_coeffs[chunk_idx] = cur_chunk.cheby_coeffs()

    hdu.header['PCNPOLY'] = (params.num_of_poly_coeffs(),
                             '# of coeffs of the interpolating polynomial')
    hdu.header['PCSMPCNK'] = (params.samples_per_chunk(),
                              '# of samples in each chunk')
    hdu.header['PCALGOR'] = (params.algorithm(),
                             'Kind of polynomial compression ("algorithm")')
    hdu.header['PCMAXERR'] = (params.max_error(),
                              'Maximum compression error')
    hdu.header['PCNCOMPR'] = (np.sum(is_compressed.astype(np.uint8)),
                              '# of compressed chunks')
    hdu.header['PCNCHEBC'] = (len(np.concatenate(cheby_coeffs)),
                              '# of Chebyshev coeffs in the table')
    hdu.header['PCDEBUG'] = (0,
                             'This table uses the compact format')

    return hdu

########################################################################

def save_polycomp_parameter_space(parameter_space,
                                  file_name,
                                  uncompressed_size,
                                  table_name):
    """Save the table used for the optimization of the polycomp parameters

    Save the points in the parameter space that have been sampled for
    the optimization of the polynomial compression in a FITS file. The
    variable "parameter_space" must be a list of ppc.ParameterPoint objects.
    The value "uncompressed_size" is the size in bytes of the uncompressed
    data, and it is used to compute the compression ratios.

    The variables "num_of_coefficients" and "samples_per_chunk" are
    used to build the image HDUs.

    Return the name of the FITS file that has been created by the function."""

    # Build the matrices
    mesh = ppc.get_param_space_mesh(parameter_space)

    hdu_list = [pyfits.PrimaryHDU(),
                pyfits.ImageHDU(name='SMPPERCK',
                                data=mesh.samples_per_chunk),
                pyfits.ImageHDU(name='NUMCOEFS',
                                data=mesh.num_of_poly_coeffs)]
    hdu_list[0].header['TBLNAME'] = (table_name,
                                     'The parameter that has been compressed')
    hdu_list[0].header['UNCSIZE'] = (uncompressed_size,
                                     'Size of the uncompressed data [bytes]')
    # We take the first element of parameter_space because we assume
    # they all have the same max_error value
    hdu_list[0].header['MAXERR'] = (parameter_space[0].params.max_error(),
                                    'Maximum allowable error')
    hdu_list[0].header['ALGOR'] = (parameter_space[0].params.algorithm(),
                                   'Type of compression algorithm')

    period = parameter_space[0].params.period()
    if period is None:
        period = 0.0
    hdu_list[0].header['PERIOD'] = (period,
                                    'Periodicity of the variable (0=none)')

    for (field_name, hdu_name) in [('compr_data_size', 'COMPRSIZ'),
                                   ('cr', 'CR'),
                                   ('num_of_chunks', 'NUMCK'),
                                   ('num_of_compr_chunks', 'NUMCRCK'),
                                   ('num_of_cheby_coeffs', 'NCHEBYC'),
                                   ('elapsed_time', 'TIME')]:
        hdu_list.append(pyfits.ImageHDU(name=hdu_name,
                                        data=mesh.__dict__[field_name]))

    log.info('writing optimization information in file %s',
             os.path.abspath(file_name))

    pyfits.HDUList(hdu_list).writeto(file_name, clobber=True)

    return file_name

################################################################################

def numpy_array_size(arr):
    return arr.itemsize * arr.size

################################################################################

def must_explore_param_space(num_of_coefficients_space, samples_per_chunk_space):
    return (len(num_of_coefficients_space) > 1) or \
        (len(samples_per_chunk_space) > 1)

################################################################################

def polycomp_configuration_cost(params):
    return cur_hdu.filebytes() - len(str(cur_hdu.header))

################################################################################

def parameter_space_survey(samples, num_of_coefficients_space,
                           samples_per_chunk_space, max_error,
                           algorithm, period):

    """Performs a detailed survey of every configuration in the parameter
    space given by "num_of_coefficients_space" and
    "samples_per_chunk_space"."""

    configurations = list(itertools.product(num_of_coefficients_space,
                                            samples_per_chunk_space))

    param_points = ppc.PointCache(samples=samples,
                                  max_allowable_error=max_error,
                                  algorithm=algorithm,
                                  period=period)
    for iter_idx, cur_conf in enumerate(configurations):
        num_of_coeffs, samples_per_chunk = cur_conf

        chunks, cur_point = param_points.get_point(num_of_coeffs, samples_per_chunk)

        if must_explore_param_space(num_of_coefficients_space,
                                    samples_per_chunk_space):
            log.info('configuration %d/%d with num_of_coefficients=%d, '
                     'samples_per_chunk=%d requires %s',
                     iter_idx + 1,
                     len(configurations),
                     num_of_coeffs, samples_per_chunk,
                     humanize_size(cur_point.compr_data_size))

    best_chunks, best_point = param_points.best_point
    return best_point, list(param_points.parameter_space.values())

################################################################################

def compress_and_encode_poly(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (polynomial compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format

    num_of_coefficients_space = parse_intset(parser.get(table, 'num_of_coefficients'))
    samples_per_chunk_space = parse_intset(parser.get(table, 'samples_per_chunk'))
    max_error = parser.getfloat(table, 'max_error')
    algorithm = ppc.PCOMP_ALG_USE_CHEBYSHEV
    if not parser.getboolean(table, 'use_chebyshev'):
        algorithm = ppc.PCOMP_ALG_NO_CHEBYSHEV

    if parser.has_option(table, 'period'):
        period = parser.getfloat(table, 'period')
    else:
        period = None

    exhaustive_search = parser.has_option(table, 'no_smart_optimization') and \
                        parser.getboolean(table, 'no_smart_optimization')

    if exhaustive_search:
        # Scan the whole grid of parameters
        best_parameter_point, parameter_space = \
            parameter_space_survey(samples, num_of_coefficients_space,
                                   samples_per_chunk_space,
                                   max_error, algorithm, period)
    else:
        # Do a smart search
        num_of_coeffs_range = [f(num_of_coefficients_space) for f in (np.min, np.max)]
        samples_per_chunk_range = [f(samples_per_chunk_space) for f in (np.min, np.max)]

        delta_coeffs, delta_samples = (1, 1)
        max_iterations = 0
        if parser.has_option(table, 'opt_delta_coeffs'):
            delta_coeffs = parser.getint(table, 'opt_delta_coeffs')
        if parser.has_option(table, 'opt_delta_samples'):
            delta_samples = parser.getint(table, 'opt_delta_samples')
        if parser.has_option(table, 'opt_max_num_of_iterations'):
            max_iterations = parser.getint(table, 'opt_max_num_of_iterations')

        callback = (lambda x, y, params, steps: \
                    log.debug("Point (%d, %d) sampled (step %d), size is %s",
                              x, y, steps, humanize_size(params.compr_data_size)))
        best_parameter_point, parameter_space, num_of_steps = \
            ppc.simplex_downhill(samples, num_of_coeffs_range,
                                 samples_per_chunk_range,
                                 max_error=max_error,
                                 algorithm=algorithm,
                                 delta_coeffs=delta_coeffs,
                                 delta_samples=delta_samples,
                                 period=period,
                                 callback=callback,
                                 max_iterations=max_iterations)
        log.debug("Best configuration found in %d iterations", num_of_steps)

    optimization_file_name = None

    if parser.has_option(table, 'optimization_file_name'):
        file_name = parser.get(table, 'optimization_file_name')
        optimization_file_name = \
            save_polycomp_parameter_space(parameter_space,
                                          file_name,
                                          uncompressed_size=numpy_array_size(samples),
                                          table_name=table)

    if must_explore_param_space(num_of_coefficients_space,
                                samples_per_chunk_space):
        log.info('the best compression parameters for "%s" are '
                 'num_of_coefficients=%d, samples_per_chunk=%d (%s)',
                 table,
                 best_parameter_point.params.num_of_poly_coeffs(),
                 best_parameter_point.params.samples_per_chunk(),
                 humanize_size(best_parameter_point.compr_data_size))

    if debug:
        hdu = polycomp_chunks_to_FITS_table_debug(best_parameter_point.chunks,
                                                  best_parameter_point.params)
    else:
        hdu = polycomp_chunks_to_FITS_table(best_parameter_point.chunks,
                                            best_parameter_point.params)

    hdu.header['PCRAWCS'] = (best_parameter_point.compr_data_size,
                                                  'Size of the encoded data [bytes]')
    if optimization_file_name is not None:
        hdu.header['PCOPTIM'] = (optimization_file_name,
                                 'Optimization info file')
    return (hdu, best_parameter_point.compr_data_size)

########################################################################

def compress_and_encode_zlib(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (zlib compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format

    if parser.has_option(table, 'compression_level'):
        level = parser.getint(table, 'compression_level')
    else:
        level = 9

    compr_samples = zlib.compress(samples.tostring(), level)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format='1B',
                      array=np.array(list(compr_samples)))]),
            len(compr_samples))

########################################################################

def compress_and_encode_bzip2(parser, table, samples_format, samples, debug):
    """Save "samples" in a binary table (bz2 compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format

    if parser.has_option(table, 'compression_level'):
        level = parser.getint(table, 'compression_level')
    else:
        level = 9

    compr_samples = bz2.compress(samples.tostring(), level)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format='1B',
                      array=np.array(list(compr_samples)))]),
            len(compr_samples))

########################################################################

def compress_to_FITS_table(parser, table, samples_format, samples, debug):
    """Compress samples and save them in a FITS binary table

    The compression parameters are taken from the section named "table"
    in the ConfigParser object "parser". The samples to be compressed
    are in the "samples" variable. The compressed datastream is returned
    as a binary FITS table (pyfits.BinTableHDU).
    """

    compr_fns = {'none': compress_and_encode_none,
                 'rle': compress_and_encode_rle,
                 'diffrle': compress_and_encode_diffrle,
                 'quantization': compress_and_encode_quant,
                 'polynomial': compress_and_encode_poly,
                 'zlib': compress_and_encode_zlib,
                 'bzip2': compress_and_encode_bzip2}

    try:
        compr_and_encode_fn = compr_fns[parser.get(table, 'compression')]
    except KeyError as e:
        log.error('unknown compression method "%s"', e.message)
        sys.exit(1)

    start_time = time.clock()
    result = compr_and_encode_fn(parser, table, samples_format, samples, debug)
    end_time = time.clock()

    return result + (end_time - start_time,)

########################################################################

def read_and_compress_table(parser, table, debug):
    """Read data from a FITS file and save it into a FITS binary table

    The data are read from the FITS file specified in the "table"
    section of the "parser" object (an instance of
    ConfigurationParser). If "debug" is true, save additional
    information (useful for debugging) in the table."""

    input_file_name = os.path.normpath(parser.get(table, 'file'))
    cur_hdu = None
    with pyfits.open(input_file_name) as input_file:
        hdu, column = get_hdu_and_column_from_schema(parser, table,
                                                     input_file)
        compression = parser.get(table, 'compression')
        log.info('compressing file %s (HDU %s, column %s) '
                 'into table "%s", '
                 'compression is "%s"',
                 input_file_name, str(hdu), str(column),
                 table, compression)

        samples_format = input_file[hdu].columns.formats[column]
        samples = ppc.to_native_endianness(input_file[hdu].data.field(column))
        if parser.has_option(table, 'datatype'):
            samples = np.array(samples, dtype=parser.get(table, 'datatype'))

        cur_hdu, num_of_bytes, elapsed_time = compress_to_FITS_table(parser, table,
                                                                     samples_format,
                                                                     samples,
                                                                     debug)
        cur_hdu.name = table
        cur_hdu.header['PCNUMSA'] = (len(samples),
                                     'Number of uncompressed samples')
        cur_hdu.header['PCCOMPR'] = (compression,
                                     'Polycomp compression algorithm')
        cur_hdu.header['PCSRCTP'] = (str(samples.dtype),
                                     'Original NumPy type of the data')
        cur_hdu.header['PCUNCSZ'] = (samples.itemsize * samples.size,
                                     'Size of the uncompressed data [bytes]')
        cur_hdu.header['PCCOMSZ'] = (num_of_bytes,
                                     'Size of the compressed data [bytes]')
        cur_hdu.header['PCTIME'] = (elapsed_time,
                                    'Time used for compression [s]')
        cr = float(cur_hdu.header['PCUNCSZ']) / float(cur_hdu.header['PCCOMSZ'])
        cur_hdu.header['PCCR'] = (cr, 'Compression ratio')
        log.info('table "%s" compressed, %s compressed to %s (cr: %.4f)',
                 table, humanize_size(cur_hdu.header['PCUNCSZ']),
                 humanize_size(cur_hdu.header['PCCOMSZ']), cr)

    return cur_hdu

########################################################################

def add_metadata_to_HDU(parser, hdu_header):
    "Read the [metadata] section and copy each key/value in an HDU"

    # Since parser.options lists the variables in the [DEFAULT]
    # section as well, we must perform a set subtraction in order to
    # loop only over the true metadata.

    if not parser.has_section('metadata'):
        return

    try:
        defaults = set(parser.defaults().keys())
        for name in set(parser.options('metadata')) - defaults:
            hdu_header[name] = parser.get('metadata', name)
    except ConfigParser.error as e:
        # If there are errors in the 'metadata' section, just warn
        # about them
        log.warn('while processing metadata: %s', str(e))

########################################################################

def decompress_none(hdu):
    "Decompress an HDU which uses no compression"

    return ppc.decompress_none_from_hdu(hdu), hdu.columns.formats[0]

########################################################################

def decompress_rle(hdu):
    "Decompress an HDU containing a RLE-compressed datastream"

    return ppc.decompress_rle_from_hdu(hdu), hdu.columns.formats[0]

########################################################################

def decompress_diffrle(hdu):
    "Decompress an HDU containing a diffRLE-compressed datastream"

    return ppc.decompress_diffrle_from_hdu(hdu), hdu.columns.formats[0]

########################################################################

def decompress_quant(hdu):
    "Decompress an HDU containing quantized data"

    quant = ppc.QuantParams(element_size=hdu.header['PCELEMSZ'],
                            bits_per_sample=hdu.header['PCBITSPS'])
    quant.set_normalization(normalization=hdu.header['PCNORM'],
                            offset=hdu.header['PCOFS'])
    size_to_fits_fmt = {4: '1E', 8: '1D'}
    compr_samples = ppc.to_native_endianness(hdu.data.field(0))

    try:
        return (quant.decompress(compr_samples,
                                 hdu.header['PCNUMSA']),
                size_to_fits_fmt[quant.element_size()])
    except KeyError:
        log.error('unable to handle floating-point types which '
                  'are %d bytes wide, allowed sizes are %s',
                  quant.element_size(), str(list(size_to_fits_fmt.keys())))
        sys.exit(1)

########################################################################

def decompress_poly_debug(hdu):
    "Decompress an HDU containing polynomial compressed data in debug form"

    if hdu.columns.names != ['ISCOMPR', 'CKLEN', 'UNCOMPR',
                             'POLY', 'CHMASK', 'CHEBY']:
        raise ValueError('unknown sequence of columns for polynomial '
                         'compression: %s',
                         str(hdu.columns.names))

    inv_cheby = None
    samples = np.empty(0, dtype='float64')
    is_compressed, chunk_len, uncompr, \
        poly, cheby_mask, cheby = [hdu.data.field(x)
                                   for x in range(6)]

    num_of_chunks = is_compressed.size
    poly_size = np.array([poly[idx].size for idx in range(num_of_chunks)],
                         dtype=np.uint8)
    cheby_size = np.array([cheby[idx].size for idx in range(num_of_chunks)],
                          dtype=np.uint16)
    chunk_array = ppc.build_chunk_array(is_compressed=is_compressed.astype(np.uint8),
                                        chunk_len=chunk_len.astype(np.uint16),
                                        uncompr=np.concatenate(uncompr),
                                        poly_size=poly_size,
                                        poly=np.concatenate(poly),
                                        cheby_size=cheby_size,
                                        cheby_mask=np.concatenate(cheby_mask),
                                        cheby=np.concatenate(cheby))

    return ppc.decompress_polycomp(chunk_array), 'D'

########################################################################

def decompress_poly_compact(hdu):
    "Decompress an HDU containing polynomial compressed data in compact form"

    return ppc.decompress_polynomial_from_hdu(hdu), 'D'

########################################################################

def decompress_poly(hdu):
    "Decompress an HDU containing polynomial compressed data"

    if hdu.header['PCDEBUG'] != 0:
        return decompress_poly_debug(hdu)
    else:
        return decompress_poly_compact(hdu)

########################################################################

def decompress_zlib(hdu):
    "Decompress an HDU containing a datastream compressed using zlib"

    source_type = np.dtype(hdu.header['PCSRCTP'])
    return (ppc.decompress_zlib_from_hdu(hdu),
            ppc.numpy_type_to_fits_type(source_type))

########################################################################

def decompress_bzip2(hdu):
    "Decompress an HDU containing a datastream compressed using bzip2"

    source_type = np.dtype(hdu.header['PCSRCTP'])
    return (ppc.decompress_bzip2_from_hdu(hdu),
            ppc.numpy_type_to_fits_type(source_type))

########################################################################

def decompress_FITS_HDU(hdu):
    """Read compressed samples from a binary table and decompress them.

    Return the pair (SAMPLES, FMT), where SAMPLES is a NumPy array
    containing the decompressed samples, and FMT is the FITSIO format
    associated with the data.
    """

    decompr_fns = {'none': decompress_none,
                   'rle': decompress_rle,
                   'diffrle': decompress_diffrle,
                   'quantization': decompress_quant,
                   'polynomial': decompress_poly,
                   'zlib': decompress_zlib,
                   'bzip2': decompress_bzip2}

    decompr_fn = None
    compression = hdu.header['PCCOMPR']
    try:
        decompr_fn = decompr_fns[compression]
    except KeyError as exc:
        log.error('unknown compression method "%s"', exc.message)
        return None

    samples, fmt = decompr_fn(hdu)
    log.info('table "%s" decompressed (compression type: %s), first value is %s',
             hdu.name, compression, str(samples[0]))
    # Convert the samples to their original NumPy type
    return np.asarray(samples, dtype=hdu.header['PCSRCTP']), fmt

########################################################################

def print_general_info():
    import platform

    log.info('Polycomp {ver} ({python} {pyver}, NumPy {npver}, {fits_lib} {fitsver})'
             .format(ver=__version__,
                     python=platform.python_implementation(),
                     pyver=platform.python_version(),
                     npver=np.__version__,
                     fits_lib=fits_lib,
                     fitsver=fits_lib_version))


########################################################################

@click.group()
@click.version_option(version=__version__)
@click.option('--log', 'log_file', type=click.Path(dir_okay=False),
              metavar='FILE', default=None,
              help='Write log messages to the specified file instead of stderr')
@click.option('--log-append/--no-log-append', default=False,
              help='If the file file specified via --log-file exists, \
append log messages to its tail instead of emptying it')
def main(log_file, log_append):
    '''Compress/decompress FITS files containing data in tabular format.'''

    if log_append:
        log_mode = 'a'
    else:
        log_mode = 'w'

    log_fmt = 'polycomp: %(levelname)s - %(message)s'

    # Without this "if", the code would fail with the ValueError
    # exception "Unrecognised argument(s): filemode" when "log_file"
    # is not specified. This happens for some old Python versions
    # (http://bugs.python.org/issue23207).
    if log_file:
        log.basicConfig(level=log.DEBUG,
                        format=log_fmt,
                        filename=log_file,
                        filemode=log_mode)
    else:
        log.basicConfig(level=log.DEBUG,
                        format=log_fmt)

    print_general_info()

################################################################################

@main.command('compress')
@click.argument('schema', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False, writable=True))
@click.argument('key-value', nargs=-1)
@click.option('-t', '--table', 'table_list',
              help='''Name of the tables to be compressed. It can be specified
more than once. If this flag is not used, all tables in the schema file are
processed.''',
              default=None, metavar='NAME', multiple=True)
@click.option('--debug/--no-debug', help='''When saving data compressed using the \
polynomial compression, use an extended format that is easier to debug. This wastes \
some space''')
def compress(schema, output, table_list, debug, key_value):
    '''Compress one or more HDUs from FITS files using the specified schema.'''

    default_conf = {'clobber': 'True',
                    'write_checksum': 'True'}
    table_list = [x.upper() for x in table_list]
    for key_val_pair in key_value:
        if '=' not in key_val_pair:
            log.error('wrong key=value parameter "%s"', key_val_pair)
            continue
        key, value = key_val_pair.split('=', 1)
        default_conf[key] = value

    parser = configparser.ConfigParser(defaults=default_conf)

    try:
        if parser.read(schema) == []:
            raise IOError(1, 'file not found')

    except (OSError, IOError) as exc:
        log.error('unable to load file "{0}": {1}'
                  .format(schema, exc.strerror))

    try:
        tables = (parser.get('polycomp', 'tables')
                  .replace(' ', '').split(','))
        output_hdus = pyfits.HDUList()

        for cur_table in tables:
            if (table_list is not None) and (len(table_list) > 0) \
              and (cur_table.upper() not in table_list):
                continue

            output_hdus.append(read_and_compress_table(parser, cur_table,
                                                       debug=debug))

        if len(output_hdus) > 0:
            add_metadata_to_HDU(parser, output_hdus[0].header)
        else:
            log.warning('no HDUs have been created')

        output_hdus.writeto(output,
                            clobber=parser.getboolean('DEFAULT', 'clobber'),
                            checksum=parser.getboolean('DEFAULT',
                                                       'write_checksum'))
        log.info('File "%s" written to disk', output)

    except configparser.Error as exc:
        log.error("invalid schema file \"{0}\": {1}"
                  .format(schema, str(exc)))

    except (IOError, OSError) as exc:
        log.error('unable create file "%s": %s',
                  output, str(exc))

################################################################################

@main.command()
@click.argument('input-file-name',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('output-file-name',
                type=click.Path(exists=False, dir_okay=False, writable=True))
@click.option('-t', '--table', 'table_list', metavar='NAME', multiple=True, default=None,
              help='Table to be decompressed. It can be repeated. If no --table \
              flags are used, all tables are decompressed')
@click.option('--one-hdu', is_flag=True,
              help='Write all the tables as columns of the same HDU, instead of \
              splitting them into multiple HDUs.')
def decompress(input_file_name, output_file_name, table_list, one_hdu):
    'Decompress a polycomp file into a FITS file.'

    if one_hdu:
        fits_tables = []
    else:
        fits_tables = pyfits.HDUList([])

    table_list = [x.upper() for x in table_list]
    with pyfits.open(input_file_name) as input_file:
        log.info('reading file "%s"', input_file_name)

        for cur_hdu in input_file:
            if type(cur_hdu) is not pyfits.BinTableHDU:
                continue

            if 'PCCOMPR' not in cur_hdu.header:
                log.warning('HDU %s seems not to have been created by '
                            'Polycomp, I will skip it',
                            cur_hdu.name)
                continue

            if (len(table_list) > 0) and (cur_hdu.name.upper() not in table_list):
                continue

            samples, samples_format = decompress_FITS_HDU(cur_hdu)
            if samples is None:
                continue

            cur_column = pyfits.Column(name=cur_hdu.name,
                                       format=samples_format,
                                       array=samples)
            if one_hdu:
                fits_tables.append(cur_column)
            else:
                fits_tables.append(pyfits.BinTableHDU.from_columns([cur_column]))

    if one_hdu:
        hdu = pyfits.BinTableHDU.from_columns(fits_tables)
        hdu.writeto(output_file_name, clobber=True)
    else:
        fits_tables.writeto(output_file_name, clobber=True)

################################################################################

if __name__ == "__main__":
    # Turn off annoying (and useless) PyFits warnings
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    main()
