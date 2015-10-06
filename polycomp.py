#!/usr/bin/env python3
# -*- mode: python -*-

"""Compress/decompress FITS files using polynomial compression and other
algorithms.

Usage:
    polycomp compress <schema_file> <output_file>
    polycomp decompress [--output=<output_file>] <input_file>
    polycomp optimize <fits_file>
    polycomp info <input_file>
    polycomp (-h | --help)
    polycomp --version

Options:
    -h, --help              Show this help
    --version               Print the version number of the executable
    --output=<output_file>  Specify the name of the output file
"""

from docopt import docopt
import os.path
import sys
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

    return pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format, array=samples)])

########################################################################

def compress_and_encode_rle(parser, table, samples_format, samples):
    """Save "samples" in a binary table (RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser

    compr_samples = ppc.rle_compress(samples)
    return pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)])

########################################################################

def compress_and_encode_diffrle(parser, table, samples_format, samples):
    """Save "samples" in a binary table (differenced RLE compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del parser

    compr_samples = ppc.diffrle_compress(samples)
    return pyfits.BinTableHDU.from_columns([
        pyfits.Column(name=table, format=samples_format,
                      array=compr_samples)])

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

    return hdu

########################################################################

def compress_and_encode_poly(parser, table, samples_format, samples):
    """Save "samples" in a binary table (polynomial compression)

    This function is called by "compress_to_FITS_table".
    """

    # Remove unused parameters
    del samples_format

    num_of_coefficients = parser.getint(table, 'num_of_coefficients')
    samples_per_chunk = parser.getint(table, 'samples_per_chunk')
    max_error = parser.getfloat(table, 'max_error')
    algorithm = ppc.PCOMP_ALG_USE_CHEBYSHEV
    if not parser.getboolean(table, 'use_chebyshev'):
        algorithm = ppc.PCOMP_ALG_NO_CHEBYSHEV

    params = ppc.Polycomp(num_of_samples=samples_per_chunk,
                          num_of_coeffs=num_of_coefficients,
                          max_allowable_error=max_error,
                          algorithm=algorithm)

    chunks = ppc.compress_polycomp(samples, params)

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

    hdu.header['PCNPOLY'] = (num_of_coefficients,
                             'Number of coefficients of the interpolating polynomial')
    hdu.header['PCSMPCNK'] = (samples_per_chunk,
                              'Number of samples in each chunk')
    hdu.header['PCALGOR'] = (algorithm,
                             'Kind of polynomial compression ("algorithm")')
    hdu.header['PCMAXERR'] = (max_error,
                              'Maximum compression error')

    return hdu

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
                 'polynomial': compress_and_encode_poly}

    try:
        compr_and_encode_fn = compr_fns[parser.get(table, 'compression')]
    except KeyError as e:
        log.error('unknown compression method "%s"', e.message)
        sys.exit(1)

    return compr_and_encode_fn(parser, table, samples_format, samples)

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
        samples = input_file[hdu].data.field(column)

        # Check for endianness
        if str(samples.dtype)[0] in ('<', '>'):
            # Convert to native order
            new_dtype = '=' + str(samples.dtype)[1:]
            samples = samples.astype(new_dtype)

        original_samples_dtype = samples.dtype
        if parser.has_option(table, 'datatype'):
            samples = np.array(samples, dtype=parser.get(table, 'datatype'))

        cur_hdu = compress_to_FITS_table(parser, table,
                                         samples_format,
                                         samples)
        cur_hdu.header['PCNUMSA'] = (len(samples),
                                     'Number of uncompressed samples')
        cur_hdu.header['PCCOMPR'] = (compression,
                                     'Polycomp compression algorithm')
        cur_hdu.header['PCSRCTP'] = (str(original_samples_dtype),
                                     'Original NumPy type of the data')
    return cur_hdu

########################################################################

def do_compress(arguments):
    """This function is called when the user uses the command-line
    'compress' command."""

    schema_file_name = arguments['<schema_file>']
    output_file_name = arguments['<output_file>']
    default_conf = {'clobber': 'True',
                    'write_checksum': 'True'}
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
        pass
    elif arguments['optimize']:
        pass
    elif arguments['info']:
        pass
    else:
        print('Sorry, I do not know what to do!')

if __name__ == "__main__":
    main()
