#!/usr/bin/env python3
# -*- mode: python -*-

"""Compress/decompress FITS files using polynomial compression and other
algorithms.

Usage:
    polycomp compress <schema_file> <output_file> [<key=value>...]
    polycomp decompress [--output=FILE] [--one-hdu] <input_file>
    polycomp optimize <fits_file>
    polycomp info <input_file>
    polycomp (-h | --help)
    polycomp --version

The <key=value> options specify substitutions for the values in the
schema file.

Options:
    -h, --help              Show this help
    --version               Print the version number of the executable
    -o FILE, --output=FILE  Specify the name of the output file
    --one-hdu               Save all the data in columns of the same HDU.
                            This only works if all the tables in <input_file>
                            have the same number of elements when decompressed.
"""

from docopt import docopt
import itertools
import os.path
import sys
import time
import zlib
import bz2
import pypolycomp as ppc
import logging as log
import re
import numpy as np
import pyfits

try:
    # Python 3
    import configparser
except ImportError:
    # Python 2
    import ConfigParser as configparser

########################################################################

def parse_intset(val):
    """Parse a list of integers.

    The integers must be comma-separated. Ranges in the form
    NN-MM or NN:MM are allowed: in this case, beware that the
    last element is *always* included (unlike Python). See the
    examples below.

    Examples
    --------
    >>> parse_intset(10)
    [10]
    >>> parse_intset("10")
    [10]
    >>> parse_intset("10, 11, 14-16, 7")
    [7, 10, 11, 14, 15, 16]
    """

    if type(val) in (int, long):
        return [val]
    elif type(val) is str:
        # Parse the string
        chunks = val.split(',')
        result = set()
        integer_re = re.compile('^[ \t]*[0-9]+[ \t]*$')
        range_re = re.compile('^[ \t]*([0-9]+)[ \t]*[-:][ \t]*([0-9]+)[ \t]*$')
        for chunk in chunks:
            if integer_re.match(chunk):
                result.add(int(chunk))
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

def to_native_endianness(samples):
    # Check for endianness
    if str(samples.dtype)[0] in ('<', '>'):
        # Convert to native order
        new_dtype = '=' + str(samples.dtype)[1:]
        return samples.astype(new_dtype)
    else:
        return samples

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

def compress_and_encode_none(parser, table, samples_format, samples):
    """Save "samples" in a binary table (uncompressed form)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser

    col = pyfits.Column(name=table, format=samples_format, array=samples)
    return (pyfits.BinTableHDU.from_columns([col]),
            samples.itemsize * samples.size)

########################################################################

def compress_and_encode_rle(parser, table, samples_format, samples):
    """Save "samples" in a binary table (RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser

    compr_samples = ppc.rle_compress(samples)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)]),
            compr_samples.itemsize * compr_samples.size)

########################################################################

def compress_and_encode_diffrle(parser, table, samples_format, samples):
    """Save "samples" in a binary table (differenced RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser

    compr_samples = ppc.diffrle_compress(samples)
    return (pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)]),
            compr_samples.itemsize * compr_samples.size)

########################################################################

def compress_and_encode_quant(parser, table, samples_format, samples):
    """Save "samples" in a binary table (quantization)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format

    bits_per_sample = parser.getint(table, 'bits_per_sample')
    quant = ppc.QuantParams(element_size=samples.dtype.itemsize,
                            bits_per_sample=bits_per_sample)
    compr_samples = quant.compress(samples)

    hdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format='1B',
                      array=compr_samples)])
    hdu.header['PCBITSPS'] = (bits_per_sample,
                              'Number of bits per quantized sample')
    hdu.header['PCELEMSZ'] = (samples.dtype.itemsize,
                              'Number of bytes per sample')

    # Compute and store the entropy of the quantized samples (this is
    # useful to estimate how much an additional statistical encoding
    # stage would improve the compression ratio)
    amin, amax = [f(samples) for f in (np.amin, np.amax)]
    scaled_samples = (samples - amin) / (amax - amin) * (2**bits_per_sample)
    hdu.header['PCENTROP'] = (shannon_entropy(scaled_samples),
                              'Shannon''s entropy for the quantized data')

    return (hdu, compr_samples.size * compr_samples.itemsize)

########################################################################

def polycomp_chunks_to_FITS_table(chunks, params):
    is_compressed = np.empty(len(chunks), dtype='bool')
    chunk_length = np.empty(len(chunks), dtype='uint64')
    uncompressed = np.empty(len(chunks), dtype=np.object)
    poly_coeffs = np.empty(len(chunks), dtype=np.object)
    cheby_coeffs = np.empty(len(chunks), dtype=np.object)

    for chunk_idx in range(len(chunks)):
        cur_chunk = chunks[chunk_idx]
        is_compressed[chunk_idx] = cur_chunk.is_compressed()
        chunk_length[chunk_idx] = cur_chunk.num_of_samples()
        uncompressed[chunk_idx] = cur_chunk.uncompressed_samples()
        poly_coeffs[chunk_idx] = cur_chunk.poly_coeffs()
        cheby_coeffs[chunk_idx] = cur_chunk.cheby_coeffs()

    hdu = pyfits.BinTableHDU.from_columns([
        pyfits.Column(name='ISCOMPR', format='1L', array=is_compressed),
        pyfits.Column(name='CKLEN', format='1K', array=chunk_length),
        pyfits.Column(name='UNCOMPR', format='PD()', array=uncompressed),
        pyfits.Column(name='POLY', format='PD()', array=poly_coeffs),
        pyfits.Column(name='CHEBY', format='PD()', array=cheby_coeffs)])

    hdu.header['PCNPOLY'] = (params.num_of_poly_coeffs(),
                             'Number of coefficients of the interpolating polynomial')
    hdu.header['PCSMPCNK'] = (params.samples_per_chunk(),
                              'Number of samples in each chunk')
    hdu.header['PCALGOR'] = (params.algorithm(),
                             'Kind of polynomial compression ("algorithm")')
    hdu.header['PCMAXERR'] = (params.max_error(),
                              'Maximum compression error')
    hdu.header['PCNCOMPR'] = (np.sum(is_compressed.astype(np.uint8)),
                              'Number of compressed chunks')
    hdu.header['PCNCHEBC'] = (len(np.concatenate(cheby_coeffs)),
                              'Number of Chebyshev coefficients in the table')

    return hdu

########################################################################

def compress_and_encode_poly(parser, table, samples_format, samples):
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

    explore_param_space = (len(num_of_coefficients_space) > 1) or \
                          (len(samples_per_chunk_space) > 1)

    errors_in_param_space = []
    for num_of_coeffs, samples_per_chunk in itertools.product(num_of_coefficients_space,
                                                              samples_per_chunk_space):
        params = ppc.Polycomp(num_of_samples=samples_per_chunk,
                              num_of_coeffs=num_of_coeffs,
                              max_allowable_error=max_error,
                              algorithm=algorithm)
        chunks = ppc.compress_polycomp(samples, params)
        errors_in_param_space.append((chunks.num_of_bytes(), chunks, params))

        if explore_param_space:
            log.info('  configuration with num_of_coefficients=%d, '
                     'samples_per_chunk=%d requires %d bytes',
                     num_of_coeffs, samples_per_chunk, chunks.num_of_bytes())

    if len(errors_in_param_space) == 0:
        log.error('polynomial compression parameters expected for table "%s"',
                  table)
        sys.exit(1)

    errors_in_param_space.sort(key=lambda x: x[0])
    num_of_bytes, chunks, params = errors_in_param_space[0]
    if explore_param_space:
        log.info('the best compression parameters for "%s" are '
                 'num_of_coefficients=%d, samples_per_chunk=%d (%d bytes)',
                 table,
                 params.num_of_poly_coeffs(),
                 params.samples_per_chunk(),
                 num_of_bytes)

    return (polycomp_chunks_to_FITS_table(chunks, params), num_of_bytes)

########################################################################

def compress_and_encode_zlib(parser, table, samples_format, samples):
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

def compress_and_encode_bzip2(parser, table, samples_format, samples):
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

def compress_to_FITS_table(parser, table, samples_format, samples):
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
    result = compr_and_encode_fn(parser, table, samples_format, samples)
    end_time = time.clock()

    return result + (end_time - start_time,)

########################################################################

def read_and_compress_table(parser, table):
    """Read data from a FITS file and save it into a FITS binary table

    The data are read from the FITS file specified in the "table"
    section of the "parser" object (an instance of
    ConfigurationParser).
    """

    input_file_name = os.path.normpath(parser.get(table, 'file'))
    cur_hdu = None
    with pyfits.open(input_file_name) as input_file:
        hdu, column = get_hdu_and_column_from_schema(parser, table,
                                                     input_file)
        compression = parser.get(table, 'compression')
        log.info('compressing file %s (HDU %s, column %s) '
                 'into table %s, '
                 'compression is "%s"',
                 input_file_name, str(hdu), str(column),
                 table, compression)

        samples_format = input_file[hdu].columns.formats[column]
        samples = to_native_endianness(input_file[hdu].data.field(column))
        if parser.has_option(table, 'datatype'):
            samples = np.array(samples, dtype=parser.get(table, 'datatype'))

        cur_hdu, num_of_bytes, elapsed_time = compress_to_FITS_table(parser, table,
                                                                     samples_format,
                                                                     samples)
        cur_hdu.name = table
        cur_hdu.header['PCNUMSA'] = (len(samples),
                                     'Number of uncompressed samples')
        cur_hdu.header['PCCOMPR'] = (compression,
                                     'Polycomp compression algorithm')
        cur_hdu.header['PCSRCTP'] = (str(samples.dtype),
                                     'Original NumPy type of the data')
        cur_hdu.header['PCUNCSZ'] = (samples.itemsize * samples.size,
                                     'Size (in bytes) of the uncompressed data')
        cur_hdu.header['PCCOMSZ'] = (num_of_bytes,
                                     'Size (in bytes) of the compressed data')
        cur_hdu.header['PCTIME'] = (elapsed_time,
                                    'Time (in seconds) used for compression')
        cr = float(cur_hdu.header['PCUNCSZ']) / float(cur_hdu.header['PCCOMSZ'])
        cur_hdu.header['PCCR'] = (cr, 'Compression ratio')
        log.info('table %s compressed, %d bytes compressed to %d (cr: %.4f)',
                 table, cur_hdu.header['PCUNCSZ'], cur_hdu.header['PCCOMSZ'], cr)

    return cur_hdu

########################################################################

def add_metadata_to_HDU(parser, hdu_header):
    "Read the [metadata] section and copy each key/value in an HDU"

    # Since parser.options lists the variables in the [DEFAULT]
    # section as well, we must perform a set subtraction in order to
    # loop only over the true metadata.
    defaults = set(parser.defaults().keys())
    for name in set(parser.options('metadata')) - defaults:
        hdu_header[name] = parser.get('metadata', name)

########################################################################

def do_compress(arguments):
    """This function is called when the user uses the command-line
    'compress' command."""

    schema_file_name = arguments['<schema_file>']
    output_file_name = arguments['<output_file>']
    default_conf = {'clobber': 'True',
                    'write_checksum': 'True'}
    for key_val_pair in arguments['<key=value>']:
        key, value = key_val_pair.split('=', 1)
        default_conf[key] = value

    parser = configparser.ConfigParser(defaults=default_conf)

    try:
        if parser.read(schema_file_name) == []:
            raise IOError(1, 'file not found')

    except (OSError, IOError) as exc:
        log.error('unable to load file "{0}": {1}'
                  .format(schema_file_name, exc.strerror))

    try:
        tables = (parser.get('polycomp', 'tables')
                  .replace(' ', '').split(','))
        output_hdus = pyfits.HDUList()

        for table in tables:
            output_hdus.append(read_and_compress_table(parser, table))

        add_metadata_to_HDU(parser, output_hdus[0].header)
        output_hdus.writeto(output_file_name,
                            clobber=parser.getboolean('DEFAULT', 'clobber'),
                            checksum=parser.getboolean('DEFAULT',
                                                       'write_checksum'))
        log.info('File "%s" written to disk', output_file_name)

    except configparser.Error as exc:
        log.error("invalid schema file \"{0}\": {1}"
                  .format(schema_file_name, str(exc)))

    except (IOError, OSError) as exc:
        log.error('unable to write to file "%s": %s',
                  output_file_name, exc.strerror)

########################################################################

def decompress_none(hdu):
    return hdu.data.field(0), hdu.columns.formats[0]

########################################################################

def decompress_rle(hdu):
    compr_samples = np.asarray(to_native_endianness(hdu.data.field(0)),
                               dtype=hdu.header['PCSRCTP'])
    return ppc.rle_decompress(compr_samples), hdu.columns.formats[0]

########################################################################

def decompress_diffrle(hdu):
    compr_samples = np.asarray(to_native_endianness(hdu.data.field(0)),
                               dtype=hdu.header['PCSRCTP'])
    return ppc.diffrle_decompress(compr_samples), hdu.columns.formats[0]

########################################################################

def decompress_quant(hdu):
    quant = ppc.QuantParams(element_size=hdu.header['PCELEMSZ'],
                            bits_per_sample=hdu.header['PCBITSPS'])
    size_to_fits_fmt = {4: '1E', 8: '1D'}
    compr_samples = to_native_endianness(hdu.data.field(0))

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

def decompress_poly(hdu):
    if hdu.columns.names != ['ISCOMPR', 'CKLEN', 'UNCOMPR', 'POLY', 'CHEBY']:
        raise ValueError('unknown sequence of columns for polynomial '
                         'compression: %s',
                         str(hdu.columns.names))

    inv_cheby = None
    samples = np.empty(0, dtype='float64')
    is_compressed, chunk_len, uncompr, poly, cheby = [hdu.data.field(x)
                                                      for x in (0, 1, 2, 3, 4)]
    num_of_chunks = is_compressed.size
    poly_size = np.array([poly[idx].size for idx in range(num_of_chunks)],
                         dtype=np.uint64)
    cheby_size = np.array([cheby[idx].size for idx in range(num_of_chunks)],
                         dtype=np.uint64)
    chunk_array = ppc.build_chunk_array(is_compressed=is_compressed.astype(np.uint8),
                                        chunk_len=chunk_len.astype(np.uint64),
                                        uncompr=np.concatenate(uncompr),
                                        poly_size=poly_size,
                                        poly=np.concatenate(poly),
                                        cheby_size=cheby_size,
                                        cheby=np.concatenate(cheby))

    return ppc.decompress_polycomp(chunk_array), 'D'

########################################################################

def decompress_zlib(hdu):
    data = hdu.data.field(0)
    return (np.fromstring(zlib.decompress(data.tostring()),
                          dtype=hdu.header['PCSRCTP']),
            hdu.columns.formats[0])

########################################################################

def decompress_bzip2(hdu):
    data = hdu.data.field(0)
    return (np.fromstring(bz2.decompress(data.tostring()),
                          dtype=hdu.header['PCSRCTP']),
            hdu.columns.formats[0])

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

    log.info('decompressing table "%s" (compression type: %s)',
             hdu.name, compression)
    samples, fmt = decompr_fn(hdu)
    # Convert the samples to their original NumPy type
    return np.asarray(samples, dtype=hdu.header['PCSRCTP']), fmt

########################################################################

def do_decompress(arguments):
    """This function is called when the user uses the command-line
    'decompress' command."""

    input_file_name = arguments['<input_file>']
    output_file_name = arguments['--output']
    if output_file_name is None:
        output_file_name = input_file_name + "-decompressed.fits"

    one_hdu = arguments['--one-hdu']

    if one_hdu:
        list_of_tables = []
    else:
        list_of_tables = pyfits.HDUList([])

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

            samples, samples_format = decompress_FITS_HDU(cur_hdu)
            if samples is None:
                continue

            cur_column = pyfits.Column(name=cur_hdu.name,
                                       format=samples_format,
                                       array=samples)
            if one_hdu:
                list_of_tables.append(cur_column)
            else:
                list_of_tables.append(pyfits.BinTableHDU.from_columns([cur_column]))

    if one_hdu:
        hdu = pyfits.BinTableHDU.from_columns(list_of_tables)
        hdu.writeto(output_file_name, clobber=True)
    else:
        list_of_tables.writeto(output_file_name, clobber=True)

########################################################################

def main():
    "Main function"

    log.basicConfig(level=log.DEBUG,
                    format='polycomp: %(levelname)s - %(message)s')
    arguments = docopt(__doc__,
                       version='Polycomp {0}'.format(ppc.__version__))

    if arguments['--version']:
        print(ppc.__version__)
    elif arguments['compress']:
        do_compress(arguments)
    elif arguments['decompress']:
        do_decompress(arguments)
    elif arguments['optimize']:
        pass
    elif arguments['info']:
        pass
    else:
        log.error('don''t know what to do')

if __name__ == "__main__":
    main()
