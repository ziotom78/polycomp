#!/usr/bin/env python3
# -*- mode: python -*-

import numpy as np
from pypolycomp._bindings import *
from pypolycomp.optimization import *

from pypolycomp.version import __version__

class PolycompError(Exception):
    "Generic Polycomp error"
    def __init__(self, msg):
        super(PolycompError, self).__init__()
        self.message = msg

class PolycompErrorInvalidFile(PolycompError):
    "Error raised when a .pcomp file is not valid"

    def __init__(self, file_name, msg):
        super(PolycompErrorInvalidFile, self).__init__(msg)
        self.file_name = file_name

class PolycompErrorInvalidCompression(PolycompError):
    "Error raised when an unknown compression algorithm is found in a file"

    def __init__(self, compression, msg):
        super(PolycompErrorInvalidCompression, self).__init__(msg)
        self.compression = compression

class PolycompErrorUnsupportedType(PolycompError):
    """Raised when "numpy_type_to_fits_type" receives an unsupported type
    argument
    """

    def __init__(self, dtype, msg):
        super(PolycompErrorUnsupportedType, self).__init__(msg)
        self.dtype = dtype

def numpy_type_equivalent(x, y):
    """Return True if x and y are essentially the same type.

    The comparison ignores endianness."""

    return np.issubdtype(x, y) and \
        (np.dtype(x).itemsize == np.dtype(y).itemsize)

def numpy_type_to_fits_type(nptype):
    """Convert a string representing a NumPy type (e.g., 'int8') into a
    FITS format character.

    Return the character and the value for BZERO."""

    if numpy_type_equivalent(nptype, 'int8'):
        result = 'B'
    elif numpy_type_equivalent(nptype, 'int16'):
        result = 'I'
    elif numpy_type_equivalent(nptype, 'int32'):
        result = 'J'
    elif numpy_type_equivalent(nptype, 'int64'):
        result = 'K'
    elif numpy_type_equivalent(nptype, 'uint8'):
        result = 'B'
    elif numpy_type_equivalent(nptype, 'float32'):
        result = 'E'
    elif numpy_type_equivalent(nptype, 'float64'):
        result = 'D'
    elif numpy_type_equivalent(nptype, 'int16'):
        result = 'I'
    elif numpy_type_equivalent(nptype, 'int32'):
        result = 'J'
    elif numpy_type_equivalent(nptype, 'int64'):
        result = 'K'
    else:
        msg = 'unable to save data of type {0} into a FITS table' \
          .format(str(nptype))
        raise PolycompErrorUnsupportedType(nptype, msg)

    return result

def to_native_endianness(samples):
    "Convert the endianness of a NumPy array into native."

    # Check for endianness
    if str(samples.dtype)[0] in ('<', '>'):
        # Convert to native order
        new_dtype = '=' + str(samples.dtype)[1:]
        return samples.astype(new_dtype)
    else:
        return samples

def decompress_none_from_hdu(hdu):
    "Restore uncompressed data from a PyFits HDU."
    return hdu.data.field(0)

def decompress_rle_from_hdu(hdu):
    "Decompress RLE data in a PyFits HDU object."
    compr_samples = np.asarray(to_native_endianness(hdu.data.field(0)),
                               dtype=hdu.header['PCSRCTP'])
    return ppc.rle_decompress(compr_samples)

def decompress_diffrle_from_hdu(hdu):
    "Decompress RLE data in a PyFits HDU object."
    compr_samples = np.asarray(to_native_endianness(hdu.data.field(0)),
                               dtype=hdu.header['PCSRCTP'])
    return ppc.diffrle_decompress(compr_samples)

def decompress_quant_from_hdu(hdu):
    "Decompress quantized data in a PyFits HDU object."

    quant = ppc.QuantParams(element_size=hdu.header['PCELEMSZ'],
                            bits_per_sample=hdu.header['PCBITSPS'])
    quant.set_normalization(normalization=hdu.header['PCNORM'],
                            offset=hdu.header['PCOFS'])

    compr_samples = to_native_endianness(hdu.data.field(0))
    return quant.decompress(compr_samples, hdu.header['PCNUMSA'])

def decompress_polynomial_from_hdu(hdu):
    """Decompress data in a PyFits HDU object using polynomial (de)compression.
    """

    raw_bytes = to_native_endianness(hdu.data.field(0))
    chunk_array = ppc.decode_chunk_array(raw_bytes)
    return ppc.decompress_polycomp(chunk_array)

def decompress_zlib_from_hdu(hdu):
    """Decompress data in a PyFits HDU object using libz.
    """

    import zlib

    data = hdu.data.field(0)
    source_type = np.dtype(hdu.header['PCSRCTP'])
    return (np.fromstring(zlib.decompress(data.tostring()),
                          dtype=source_type),
            numpy_type_to_fits_type(source_type))

def decompress_bzip2_from_hdu(hdu):
    """Decompress data in a PyFits HDU object using libz2.
    """

    import bz2

    data = hdu.data.field(0)
    source_type = np.dtype(hdu.header['PCSRCTP'])
    return (np.fromstring(bz2.decompress(data.tostring()),
                          dtype=source_type),
            numpy_type_to_fits_type(source_type))

def decompress_hdu(hdu):
    """Decompress the data in a PyFits HDU object.

    """

    if 'PCCOMPR' not in hdu.header:
        file_name = hdu.fileinfo()['file'].name
        raise PolycompErrorInvalidFile(file_name,
                                       '"{0}" is not a polycomp file'
                                       .format(file_name))

    decompr_table = {'none': decompress_none_from_hdu,
                     'rle': decompress_rle_from_hdu,
                     'diffrle': decompress_diffrle_from_hdu,
                     'quantization': decompress_quant_from_hdu,
                     'polynomial': decompress_polynomial_from_hdu,
                     'zlib': decompress_zlib_from_hdu,
                     'bzip2': decompress_bzip2_from_hdu}

    compression = hdu.header['PCCOMPR']
    if compression not in decompr_table:
        msg = '"{0}" is not a recognized compression algorithm' \
          .format(compression)
        raise PolycompErrorInvalidCompression(compression, msg)

    decompr_fn = decompr_table[compression]
    return decompr_fn(hdu)
