#!/usr/bin/env python
# -*- mode: cython -*-

cimport cpolycomp
from cpython cimport array
import array

import numpy as np
cimport numpy as np

# Direction of the Chebyshev transform
PCOMP_TD_DIRECT = 0
PCOMP_TD_INVERSE = 1

# Algorithm to use for polynomial compression
PCOMP_ALG_USE_CHEBYSHEV = 0
PCOMP_ALG_NO_CHEBYSHEV = 1

PCOMP_STAT_SUCCESS = 0 # All ok
PCOMP_STAT_INVALID_ENCODING = 1 # Decompression error
PCOMP_STAT_INVALID_BUFFER = 2 # Output buffer too small
PCOMP_STAT_INVALID_FIT = 3 # Least-square fit error

def __version__():
    cdef int major
    cdef int minor
    cpolycomp.pcomp_version(&major, &minor)

    return "{0}.{1}".format(major, minor)

################################################################################
# RLE compression

def rle_bufsize(input_size):
    return cpolycomp.pcomp_rle_bufsize(input_size)

def rle_compress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * rle_bufsize(values.size)
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    if values.dtype in (np.int8, np.uint8):
        result = cpolycomp.pcomp_compress_rle_int8(<np.int8_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int8_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int16, np.uint16):
        result = cpolycomp.pcomp_compress_rle_int16(<np.int16_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int16_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int32, np.uint32):
        result = cpolycomp.pcomp_compress_rle_int32(<np.int32_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int32_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int64, np.uint64):
        result = cpolycomp.pcomp_compress_rle_int64(<np.int64_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int64_t *> &values.data[0],
                                                   values.size)
    else:
        raise ValueError("invalid type {0} for RLE compression"
                         .format(str(values.dtype)))

    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("libpolycomp error (code: {0}) while doing RLE compression"
                         .format(result))

    output = np.resize(output, real_size * values.dtype.itemsize)
    return np.fromstring(output.tobytes(), values.dtype)

def rle_decompress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * np.sum(values[::2])
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    if values.dtype in (np.int8, np.uint8):
        result = cpolycomp.pcomp_decompress_rle_int8(<np.int8_t *> &output.data[0],
                                                     num_of_bytes,
                                                     <np.int8_t *> &values.data[0],
                                                     values.size)
    elif values.dtype in (np.int16, np.uint16):
        result = cpolycomp.pcomp_decompress_rle_int16(<np.int16_t *> &output.data[0],
                                                      num_of_bytes,
                                                      <np.int16_t *> &values.data[0],
                                                      values.size)
    elif values.dtype in (np.int32, np.uint32):
        result = cpolycomp.pcomp_decompress_rle_int32(<np.int32_t *> &output.data[0],
                                                      num_of_bytes,
                                                      <np.int32_t *> &values.data[0],
                                                      values.size)
    elif values.dtype in (np.int64, np.uint64):
        result = cpolycomp.pcomp_decompress_rle_int64(<np.int64_t *> &output.data[0],
                                                      num_of_bytes,
                                                      <np.int64_t *> &values.data[0],
                                                      values.size)
    else:
        raise ValueError("invalid type {0} for DIFFRLE compression"
                         .format(str(values.dtype)))

    return np.fromstring(output.tobytes(), values.dtype)

################################################################################
# Differenced RLE compression

def diffrle_bufsize(input_size):
    return cpolycomp.pcomp_diffrle_bufsize(input_size)

def diffrle_compress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * diffrle_bufsize(values.size)
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    if values.dtype in (np.int8, np.uint8):
        result = cpolycomp.pcomp_compress_diffrle_int8(<np.int8_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int8_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int16, np.uint16):
        result = cpolycomp.pcomp_compress_diffrle_int16(<np.int16_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int16_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int32, np.uint32):
        result = cpolycomp.pcomp_compress_diffrle_int32(<np.int32_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int32_t *> &values.data[0],
                                                   values.size)
    elif values.dtype in (np.int64, np.uint64):
        result = cpolycomp.pcomp_compress_diffrle_int64(<np.int64_t *> &output.data[0],
                                                   &real_size,
                                                   <np.int64_t *> &values.data[0],
                                                   values.size)
    else:
        raise ValueError("invalid type {0} for DIFFRLE compression"
                         .format(str(values.dtype)))

    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("libpolycomp error (code: {0}) while doing DIFFRLE compression"
                         .format(result))

    output = np.resize(output, real_size * values.dtype.itemsize)
    return np.fromstring(output.tobytes(), values.dtype)

def diffrle_decompress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * (1 + np.sum(values[1::2]))
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    if values.dtype in (np.int8, np.uint8):
        result = cpolycomp.pcomp_decompress_diffrle_int8(<np.int8_t *> &output.data[0],
                                                         num_of_bytes,
                                                         <np.int8_t *> &values.data[0],
                                                         values.size)
    elif values.dtype in (np.int16, np.uint16):
        result = cpolycomp.pcomp_decompress_diffrle_int16(<np.int16_t *> &output.data[0],
                                                          num_of_bytes,
                                                          <np.int16_t *> &values.data[0],
                                                          values.size)
    elif values.dtype in (np.int32, np.uint32):
        result = cpolycomp.pcomp_decompress_diffrle_int32(<np.int32_t *> &output.data[0],
                                                          num_of_bytes,
                                                          <np.int32_t *> &values.data[0],
                                                          values.size)
    elif values.dtype in (np.int64, np.uint64):
        result = cpolycomp.pcomp_decompress_diffrle_int64(<np.int64_t *> &output.data[0],
                                                          num_of_bytes,
                                                          <np.int64_t *> &values.data[0],
                                                          values.size)
    else:
        raise ValueError("invalid type {0} for DIFFRLE compression"
                         .format(str(values.dtype)))

    return np.fromstring(output.tobytes(), values.dtype)

################################################################################
# Quantization

cdef class QuantParams:
    cdef cpolycomp.pcomp_quant_params_t* _c_params

    def __cinit__(self, bits_per_sample, element_size):
        self._c_params = cpolycomp.pcomp_init_quant_params(element_size, bits_per_sample)

    def __dealloc__(self):
        cpolycomp.pcomp_free_quant_params(self._c_params)

    def element_size(self):
        return cpolycomp.pcomp_quant_element_size(self._c_params)

    def bits_per_sample(self):
        return cpolycomp.pcomp_quant_bits_per_sample(self._c_params)

    def buf_size(self, input_size):
        return cpolycomp.pcomp_quant_bufsize(input_size, self._c_params)

    def compress(self, np.ndarray input_values not None):
        # If the size of the elements in "input_values" does not match with the
        # self._c_params object, throw it away and create another one
        if input_values.dtype.itemsize != self.element_size():
            bits = self.bits_per_sample()
            cpolycomp.pcomp_free_quant_params(self._c_params)
            self._c_params = cpolycomp.pcomp_init_quant_params(input_values.dtype.itemsize,
                                                               bits)

        cdef num_of_bytes = self.buf_size(input_values.size)
        cdef size_t real_size = num_of_bytes
        cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                             dtype='int8')

        if input_values.dtype == np.float32:
            cpolycomp.pcomp_compress_quant_float(<np.float32_t *> &output.data[0],
                                                 &real_size,
                                                 <np.float32_t *> &input_values.data[0],
                                                 input_values.size,
                                                 self._c_params)
        elif input_values.dtype == np.float64:
            cpolycomp.pcomp_compress_quant_double(<np.float64_t *> &output.data[0],
                                                  &real_size,
                                                  <np.float64_t *> &input_values.data[0],
                                                  input_values.size,
                                                  self._c_params)
        else:
            raise ValueError("invalid type {0} for quantization"
                             .format(str(input_values.dtype)))

        return np.resize(output, real_size)

    def decompress(self, np.ndarray input_values not None,
                   num_of_elements,
                   output_dtype=np.float64):
        """Decompress a stream of quantized values.

        The number of values to decompress must be specified in the
        "num_of_elements" parameter, as it cannot be deduced reliably
        by "input_values".
        """
        
        cdef size_t itemsize = self.element_size()
        cdef size_t num_of_bits = len(input_values) * 8
        cdef size_t num_of_bytes = (len(input_values) * 8) / self.bits_per_sample() + 1
        cdef np.ndarray output = np.empty(num_of_elements, dtype=output_dtype)

        if output_dtype == np.float32:
            cpolycomp.pcomp_decompress_quant_float(<np.float32_t *> &output.data[0],
                                                   num_of_elements,
                                                   <np.int8_t *> &input_values.data[0],
                                                   len(input_values),
                                                   self._c_params)
        elif output_dtype == np.float64:
            cpolycomp.pcomp_decompress_quant_double(<np.float64_t *> &output.data[0],
                                                    num_of_elements,
                                                    <np.int8_t *> &input_values.data[0],
                                                    len(input_values),
                                                    self._c_params)
        else:
            raise ValueError("invalid type size ({0} bytes) for \
quantized data decompression".format(itemsize))

        return output
