#!/usr/bin/env python
# -*- mode: cython -*-

cimport cpolycomp

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

def rle_compress_int8(np.ndarray[np.int8_t, ndim=1] values):
    cdef size_t real_size
    cdef int result
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(rle_bufsize(values.size),
                                                         dtype='int8')

    result = cpolycomp.pcomp_compress_rle_int8(<np.int8_t *> output.data, &real_size,
                                               <np.int8_t *> values.data, values.size)
    return np.resize(output, real_size)

################################################################################
# Quantization

cdef class QuantParams:
    cdef cpolycomp.pcomp_quant_params_t* _c_params
    
    def __cinit__(self, element_size, bits_per_sample):
        self._c_params = cpolycomp.pcomp_init_quant_params(element_size, bits_per_sample)

    def element_size(self):
        return cpolycomp.pcomp_quant_element_size(self._c_params)

    def bits_per_sample(self):
        return cpolycomp.pcomp_quant_bits_per_sample(self._c_params)    

    def buf_size(self, input_size):
        return cpolycomp.pcomp_quant_bufsize(input_size, self._c_params)
