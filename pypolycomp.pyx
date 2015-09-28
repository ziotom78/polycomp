#!/usr/bin/env python
# -*- mode: cython -*-

cimport cpolycomp

import numpy as np
cimport numpy as np

def __version__():
    cdef int major
    cdef int minor
    cpolycomp.pcomp_version(&major, &minor)

    return "{0}.{1}".format(major, minor)

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
