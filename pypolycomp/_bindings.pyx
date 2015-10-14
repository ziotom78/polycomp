#!/usr/bin/env python
# -*- mode: cython -*-

cimport cpolycomp as cpc
from cpython cimport array
cimport libc.stdint
from libc.stdlib cimport malloc

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
    cpc.pcomp_version(&major, &minor)

    return "{0}.{1}".format(major, minor)

################################################################################
# RLE compression

def rle_bufsize(input_size):
    return cpc.pcomp_rle_bufsize(input_size)

def rle_compress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * rle_bufsize(values.size)
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    dtype = values.dtype
    if np.issubdtype(dtype, np.int8) or np.issubdtype(dtype, np.uint8):
        result = cpc.pcomp_compress_rle_int8(<np.int8_t *> &output.data[0],
                                             &real_size,
                                             <np.int8_t *> &values.data[0],
                                             values.size)
    elif np.issubdtype(dtype, np.int16) or np.issubdtype(dtype, np.uint16):
        result = cpc.pcomp_compress_rle_int16(<np.int16_t *> &output.data[0],
                                              &real_size,
                                              <np.int16_t *> &values.data[0],
                                              values.size)
    elif np.issubdtype(dtype, np.int32) or np.issubdtype(dtype, np.uint32):
        result = cpc.pcomp_compress_rle_int32(<np.int32_t *> &output.data[0],
                                              &real_size,
                                              <np.int32_t *> &values.data[0],
                                              values.size)
    elif np.issubdtype(dtype, np.int64) or np.issubdtype(dtype, np.uint64):
        result = cpc.pcomp_compress_rle_int64(<np.int64_t *> &output.data[0],
                                              &real_size,
                                              <np.int64_t *> &values.data[0],
                                              values.size)
    else:
        raise ValueError('invalid type "{0}" for RLE compression'
                         .format(str(values.dtype)))

    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("libpolycomp error (code: {0}) while doing RLE compression"
                         .format(result))

    output = np.resize(output, real_size * values.dtype.itemsize)
    return np.fromstring(output.tostring(), values.dtype)

def rle_decompress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * np.sum(values[::2])
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    dtype = values.dtype
    if np.issubdtype(dtype, np.int8) or np.issubdtype(dtype, np.uint8):
        result = cpc.pcomp_decompress_rle_int8(<np.int8_t *> &output.data[0],
                                               num_of_bytes,
                                               <np.int8_t *> &values.data[0],
                                               values.size)
    elif np.issubdtype(dtype, np.int16) or np.issubdtype(dtype, np.uint16):
        result = cpc.pcomp_decompress_rle_int16(<np.int16_t *> &output.data[0],
                                                num_of_bytes,
                                                <np.int16_t *> &values.data[0],
                                                values.size)
    elif np.issubdtype(dtype, np.int32) or np.issubdtype(dtype, np.uint32):
        result = cpc.pcomp_decompress_rle_int32(<np.int32_t *> &output.data[0],
                                                num_of_bytes,
                                                <np.int32_t *> &values.data[0],
                                                values.size)
    elif np.issubdtype(dtype, np.int64) or np.issubdtype(dtype, np.uint64):
        result = cpc.pcomp_decompress_rle_int64(<np.int64_t *> &output.data[0],
                                                num_of_bytes,
                                                <np.int64_t *> &values.data[0],
                                                values.size)
    else:
        raise ValueError("invalid type {0} for DIFFRLE decompression"
                         .format(str(values.dtype)))

    return np.fromstring(output.tostring(), values.dtype)

################################################################################
# Differenced RLE compression

def diffrle_bufsize(input_size):
    return cpc.pcomp_diffrle_bufsize(input_size)

def diffrle_compress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * diffrle_bufsize(values.size)
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    dtype = values.dtype
    if np.issubdtype(dtype, np.int8) or np.issubdtype(dtype, np.uint8):
        result = cpc.pcomp_compress_diffrle_int8(<np.int8_t *> &output.data[0],
                                                 &real_size,
                                                 <np.int8_t *> &values.data[0],
                                                 values.size)
    elif np.issubdtype(dtype, np.int16) or np.issubdtype(dtype, np.uint16):
        result = cpc.pcomp_compress_diffrle_int16(<np.int16_t *> &output.data[0],
                                                  &real_size,
                                                  <np.int16_t *> &values.data[0],
                                                  values.size)
    elif np.issubdtype(dtype, np.int32) or np.issubdtype(dtype, np.uint32):
        result = cpc.pcomp_compress_diffrle_int32(<np.int32_t *> &output.data[0],
                                                  &real_size,
                                                  <np.int32_t *> &values.data[0],
                                                  values.size)
    elif np.issubdtype(dtype, np.int64) or np.issubdtype(dtype, np.uint64):
        result = cpc.pcomp_compress_diffrle_int64(<np.int64_t *> &output.data[0],
                                                  &real_size,
                                                  <np.int64_t *> &values.data[0],
                                                  values.size)
    else:
        raise ValueError('invalid type "{0}" for DIFFRLE compression'
                         .format(str(values.dtype)))

    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("libpolycomp error (code: {0}) while doing DIFFRLE compression"
                         .format(result))

    output = np.resize(output, real_size * values.dtype.itemsize)
    return np.fromstring(output.tostring(), values.dtype)

def diffrle_decompress(np.ndarray values not None):
    cdef int result
    cdef size_t num_of_bytes = values.dtype.itemsize * (1 + np.sum(values[1::2]))
    cdef size_t real_size = num_of_bytes
    cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                         dtype='int8')

    dtype = values.dtype
    if np.issubdtype(dtype, np.int8) or np.issubdtype(dtype, np.uint8):
        result = cpc.pcomp_decompress_diffrle_int8(<np.int8_t *> &output.data[0],
                                                   num_of_bytes,
                                                   <np.int8_t *> &values.data[0],
                                                   values.size)
    elif np.issubdtype(dtype, np.int16) or np.issubdtype(dtype, np.uint16):
        result = cpc.pcomp_decompress_diffrle_int16(<np.int16_t *> &output.data[0],
                                                    num_of_bytes,
                                                    <np.int16_t *> &values.data[0],
                                                    values.size)
    elif np.issubdtype(dtype, np.int32) or np.issubdtype(dtype, np.uint32):
        result = cpc.pcomp_decompress_diffrle_int32(<np.int32_t *> &output.data[0],
                                                    num_of_bytes,
                                                    <np.int32_t *> &values.data[0],
                                                    values.size)
    elif np.issubdtype(dtype, np.int64) or np.issubdtype(dtype, np.uint64):
        result = cpc.pcomp_decompress_diffrle_int64(<np.int64_t *> &output.data[0],
                                                    num_of_bytes,
                                                    <np.int64_t *> &values.data[0],
                                                    values.size)
    else:
        raise ValueError("invalid type {0} for DIFFRLE decompression"
                         .format(str(values.dtype)))

    return np.fromstring(output.tostring(), values.dtype)

################################################################################
# Quantization

cdef class QuantParams:
    cdef cpc.pcomp_quant_params_t* _c_params

    def __cinit__(self, bits_per_sample, element_size):
        self._c_params = cpc.pcomp_init_quant_params(element_size, bits_per_sample)

    def __dealloc__(self):
        cpc.pcomp_free_quant_params(self._c_params)

    def element_size(self):
        return cpc.pcomp_quant_element_size(self._c_params)

    def bits_per_sample(self):
        return cpc.pcomp_quant_bits_per_sample(self._c_params)

    def normalization(self):
        return cpc.pcomp_quant_normalization(self._c_params)

    def offset(self):
        return cpc.pcomp_quant_offset(self._c_params)

    def set_normalization(self, normalization, offset):
        cpc.pcomp_quant_set_normalization(self._c_params,
                                          normalization,
                                          offset)

    def buf_size(self, input_size):
        return cpc.pcomp_quant_bufsize(input_size, self._c_params)

    def compress(self, np.ndarray input_values not None):
        # If the size of the elements in "input_values" does not match with the
        # self._c_params object, throw it away and create another one
        if input_values.dtype.itemsize != self.element_size():
            bits = self.bits_per_sample()
            cpc.pcomp_free_quant_params(self._c_params)
            self._c_params = cpc.pcomp_init_quant_params(input_values.dtype.itemsize,
                                                               bits)

        cdef num_of_bytes = self.buf_size(input_values.size)
        cdef size_t real_size = num_of_bytes
        cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                             dtype='int8')

        if np.issubdtype(input_values.dtype, np.float32):
            cpc.pcomp_compress_quant_float(<np.float32_t *> &output.data[0],
                                           &real_size,
                                           <np.float32_t *> &input_values.data[0],
                                           input_values.size,
                                           self._c_params)
        elif np.issubdtype(input_values.dtype, np.float64):
            cpc.pcomp_compress_quant_double(<np.float64_t *> &output.data[0],
                                            &real_size,
                                            <np.float64_t *> &input_values.data[0],
                                            input_values.size,
                                            self._c_params)
        else:
            raise ValueError("invalid type {0} for quantization"
                             .format(str(input_values.dtype)))

        return np.resize(output, real_size)

    def decompress(self, np.ndarray input_values not None,
                   num_of_samples,
                   output_dtype=np.float64):
        """Decompress a stream of quantized values.

        The number of values to decompress must be specified in the
        "num_of_samples" parameter, as it cannot be deduced reliably
        by "input_values".
        """

        cdef size_t itemsize = self.element_size()
        cdef size_t num_of_bits = len(input_values) * 8
        cdef size_t num_of_bytes = (len(input_values) * 8) / self.bits_per_sample() + 1
        cdef np.ndarray output = np.empty(num_of_samples, dtype=output_dtype)

        if output_dtype == np.float32:
            cpc.pcomp_decompress_quant_float(<np.float32_t *> &output.data[0],
                                             num_of_samples,
                                             <np.int8_t *> &input_values.data[0],
                                             len(input_values),
                                             self._c_params)
        elif output_dtype == np.float64:
            cpc.pcomp_decompress_quant_double(<np.float64_t *> &output.data[0],
                                              num_of_samples,
                                              <np.int8_t *> &input_values.data[0],
                                              len(input_values),
                                              self._c_params)
        else:
            raise ValueError("invalid type size ({0} bytes) for \
quantized data decompression".format(itemsize))

        return output

################################################################################
# Polynomial fit

cdef class PolyFit:
    cdef cpc.pcomp_poly_fit_data_t* _c_fit

    def __cinit__(self, num_of_samples, num_of_coeffs):
        self._c_fit = cpc.pcomp_init_poly_fit(num_of_samples, num_of_coeffs)

    def __dealloc__(self):
        cpc.pcomp_free_poly_fit(self._c_fit)

    def num_of_samples(self):
        return cpc.pcomp_poly_fit_num_of_samples(self._c_fit)

    def num_of_coeffs(self):
        return cpc.pcomp_poly_fit_num_of_coeffs(self._c_fit)

    def run(self, np.ndarray[np.float64_t, ndim=1] samples not None):
        cdef np.ndarray[np.float64_t, ndim=1] coeffs = np.empty(self.num_of_coeffs(),
                                                                dtype='float64')
        cdef int result
        result = cpc.pcomp_run_poly_fit(self._c_fit,
                                        <np.float64_t *> &coeffs.data[0],
                                        <np.float64_t *> &samples.data[0])

        if result != PCOMP_STAT_SUCCESS:
            raise ValueError("pcomp_run_poly_fit returned code {0}"
                             .format(result))

        return coeffs

################################################################################
# Chebyshev decomposition

cdef class Chebyshev:
    cdef cpc.pcomp_chebyshev_t* _c_cheb

    def __cinit__(self, num_of_samples, direction):
        self._c_cheb = cpc.pcomp_init_chebyshev(num_of_samples, direction)

    def __dealloc__(self):
        cpc.pcomp_free_chebyshev(self._c_cheb)

    def num_of_samples(self):
        return cpc.pcomp_chebyshev_num_of_samples(self._c_cheb)

    def direction(self):
        return cpc.pcomp_chebyshev_direction(self._c_cheb)

    def run(self, np.ndarray samples not None, dir=None):
        cdef np.ndarray[np.float64_t, ndim=1] output = np.empty(self.num_of_samples())
        cdef int result

        if dir is None:
            dir = self.direction()

        result = cpc.pcomp_run_chebyshev(self._c_cheb, dir,
                                         <np.float64_t *> &output.data[0],
                                         <np.float64_t *> &samples.data[0])

        if result != PCOMP_STAT_SUCCESS:
            raise ValueError("pcomp_run_chebyshev returned code {0}"
                             .format(result))

        return output

    cdef cpc.pcomp_chebyshev_t* __ptr(self):
        return self._c_cheb

################################################################################
# Polynomial compression

def straighten(np.ndarray[np.float64_t, ndim=1] samples not None, period):
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(samples.size)

    cpc.pcomp_straighten(<np.float64_t *> &result.data[0],
                               <np.float64_t *> &samples.data[0],
                               samples.size, period)

    return result

cdef class Polycomp:
    cdef cpc.pcomp_polycomp_t* _c_params

    def __cinit__(self, num_of_samples, num_of_coeffs,
                  max_allowable_error, algorithm=PCOMP_ALG_USE_CHEBYSHEV):
        self._c_params = cpc.pcomp_init_polycomp(num_of_samples,
                                                 num_of_coeffs,
                                                 max_allowable_error,
                                                 algorithm)
    cdef cpc.pcomp_polycomp_t* __ptr(Polycomp self):
        return self._c_params

    def __dispose__(self):
        cpc.pcomp_free_polycomp(self._c_params)

    def samples_per_chunk(Polycomp self):
        return cpc.pcomp_polycomp_samples_per_chunk(self._c_params)

    def num_of_poly_coeffs(Polycomp self):
        return cpc.pcomp_polycomp_num_of_poly_coeffs(self._c_params)

    def max_error(Polycomp self):
        return cpc.pcomp_polycomp_max_error(self._c_params)

    def algorithm(Polycomp self):
        return cpc.pcomp_polycomp_algorithm(self._c_params)

cdef extern from "stdlib.h":
    void free(void *) nogil

cdef class PolycompChunk:
    cdef cpc.pcomp_polycomp_chunk_t* _c_chunk;
    cdef int free_flag

    def __cinit__(PolycompChunk self):
        self._c_chunk = NULL
        self.free_flag = 0

    def init_with_num_of_samples(self, num_of_samples):

        self._c_chunk = cpc.pcomp_init_chunk(num_of_samples)
        self.free_flag = 1

    def init_with_ptr(self, ptr, free_at_end=True):

        self._c_chunk = <cpc.pcomp_polycomp_chunk_t*> ptr
        if free_at_end:
            self.free_flag = 1
        else:
            self.free_flag = 0

    def init_uncompressed(self,
                          np.ndarray[dtype=np.float64_t, ndim=1] samples not None):

        self._c_chunk = cpc.pcomp_init_uncompressed_chunk(<size_t> samples.size,
                                                          <np.float64_t *> &samples.data[0])
        self.free_flag = 1

    def init_compressed(self,
                        num_of_samples,
                        np.ndarray[dtype=np.float64_t, ndim=1] poly not None,
                        np.ndarray[dtype=np.float64_t, ndim=1] cheby not None):

        self._c_chunk = cpc.pcomp_init_compressed_chunk(num_of_samples,
                                                        poly.size,
                                                        <np.float64_t *> &poly.data[0],
                                                        cheby.size,
                                                        <np.float64_t *> &cheby.data[0])
        self.free_flag = 1

    def __dealloc__(self):
        if self.free_flag != 0:
            cpc.pcomp_free_chunk(self._c_chunk)
            self._c_chunk = NULL

    def num_of_samples(self):
        return cpc.pcomp_chunk_num_of_samples(self._c_chunk)

    def num_of_samples(self):
        return cpc.pcomp_chunk_num_of_bytes(self._c_chunk)

    def is_compressed(self):
        return cpc.pcomp_chunk_is_compressed(self._c_chunk) != 0

    def uncompressed_samples(self):
        if self.is_compressed():
            return np.array([])

        cdef size_t num = self.num_of_samples()
        cdef const double* ptr = cpc.pcomp_chunk_uncompressed_data(self._c_chunk)
        cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(num)
        cdef size_t i

        # Copy the memory, as "ptr" is still owned by libpolycomp
        for i in range(num):
            result[i] = ptr[i]

        return result

    def num_of_poly_coeffs(self):
        return cpc.pcomp_chunk_num_of_poly_coeffs(self._c_chunk)

    def poly_coeffs(self):
        if self.num_of_poly_coeffs() == 0:
            return np.array([])

        cdef size_t num = self.num_of_poly_coeffs()
        cdef const double* ptr = cpc.pcomp_chunk_poly_coeffs(self._c_chunk)
        cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(num)
        cdef size_t i

        for i in range(num):
            result[i] = ptr[i]

        return result

    def num_of_cheby_coeffs(self):
        return cpc.pcomp_chunk_num_of_cheby_coeffs(self._c_chunk)

    def cheby_coeffs(self):
        if self.num_of_cheby_coeffs() == 0:
            return np.array([])

        cdef size_t num = self.num_of_cheby_coeffs()
        cdef const double* ptr = cpc.pcomp_chunk_cheby_coeffs(self._c_chunk)
        cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(num)
        cdef size_t i

        for i in range(num):
            result[i] = ptr[i]

        return result

    def compress(self, np.ndarray[np.float64_t, ndim=1] values not None,
                 Polycomp params):
        cdef double max_error
        cdef int result

        result = cpc.pcomp_run_polycomp_on_chunk(params.__ptr(),
                                                 <np.float64_t *> &values.data[0],
                                                 values.size,
                                                 self._c_chunk,
                                                 &max_error)
        if result != PCOMP_STAT_SUCCESS:
            raise ValueError("pcomp_run_polycomp_on_chunk returned code {0}"
                             .format(result))

        return max_error

    def decompress(self, Chebyshev inv_chebyshev):
        cdef np.ndarray[np.float64_t, ndim=1] output = np.empty(self.num_of_samples())
        cdef int result
        result = cpc.pcomp_decompress_polycomp_chunk(<np.float64_t *> &output.data[0],
                                                     self._c_chunk,
                                                     inv_chebyshev.__ptr())
        if result != PCOMP_STAT_SUCCESS:
            raise ValueError("pcomp_decompress_polycomp_chunk returned code {0}"
                             .format(result))

        return output

cdef class PolycompChunkArray:
    cdef cpc.pcomp_polycomp_chunk_t** _c_array
    cdef size_t num_of_chunks

    def __cinit__(PolycompChunkArray self,
                  libc.stdint.uintptr_t ptr,
                  size_t num_of_chunks):
        self._c_array = <cpc.pcomp_polycomp_chunk_t**> ptr
        self.num_of_chunks = num_of_chunks

    def __dealloc__(PolycompChunkArray self):
        cpc.pcomp_free_chunks(self._c_array, self.num_of_chunks)

    def __len__(PolycompChunkArray self):
        return self.num_of_chunks

    def __getitem__(PolycompChunkArray self, int idx):
        chunk = PolycompChunk()
        chunk.init_with_ptr(<object> self._c_array[idx],
                            free_at_end=False)

        return chunk

    def total_num_of_samples(PolycompChunkArray self):
        return cpc.pcomp_total_num_of_samples(self._c_array,
                                              self.num_of_chunks)

    def num_of_bytes(self):
        return cpc.pcomp_chunks_num_of_bytes(self._c_array,
                                             self.num_of_chunks)

    def encode(PolycompChunkArray self):
        cdef num_of_bytes = self.num_of_bytes()
        cdef size_t real_size = num_of_bytes
        cdef np.ndarray[np.int8_t, ndim=1] output = np.empty(num_of_bytes,
                                                             dtype='int8')

        cpc.pcomp_encode_chunks(<np.int8_t *> &output.data[0],
                                &real_size,
                                self._c_array,
                                self.num_of_chunks)

        return np.resize(output, real_size)

    cdef cpc.pcomp_polycomp_chunk_t** __ptr(PolycompChunkArray self):
        return self._c_array

def compress_polycomp(np.ndarray[np.float64_t, ndim=1] samples not None,
                      Polycomp params):
    cdef cpc.pcomp_polycomp_chunk_t** chunk_array
    cdef size_t num_of_chunks
    cdef int result

    result = cpc.pcomp_compress_polycomp(&chunk_array, &num_of_chunks,
                                         <np.float64_t *> &samples.data[0],
                                         samples.size,
                                         params.__ptr())
    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("pcomp_compress_polycomp returned code {0}"
                         .format(result))

    return PolycompChunkArray(<libc.stdint.uintptr_t> chunk_array, num_of_chunks)

def build_chunk_array(np.ndarray[dtype=np.uint8_t, ndim=1] is_compressed,
                      np.ndarray[dtype=np.uint64_t, ndim=1] chunk_len,
                      np.ndarray[dtype=np.float64_t, ndim=1] uncompr,
                      np.ndarray[dtype=np.uint64_t, ndim=1] poly_size,
                      np.ndarray[dtype=np.float64_t, ndim=1] poly,
                      np.ndarray[dtype=np.uint64_t, ndim=1] cheby_size,
                      np.ndarray[dtype=np.float64_t, ndim=1] cheby):
    cdef size_t num_of_chunks = is_compressed.size
    cdef cpc.pcomp_polycomp_chunk_t** ptr = <cpc.pcomp_polycomp_chunk_t**> malloc(sizeof(void *) * num_of_chunks)
    cdef size_t uncompr_idx = 0
    cdef size_t poly_idx = 0
    cdef size_t cheby_idx = 0

    for idx in range(num_of_chunks):
        if is_compressed[idx] == 0:
            ptr[idx] = cpc.pcomp_init_uncompressed_chunk(chunk_len[idx],
                                                         (<double *> &uncompr.data[0]) + uncompr_idx)
            uncompr_idx += chunk_len[idx]
        else:
            ptr[idx] = cpc.pcomp_init_compressed_chunk(chunk_len[idx],
                                                       poly_size[idx],
                                                       (<double *> &poly.data[0]) + poly_idx,
                                                       cheby_size[idx],
                                                       (<double *> &cheby.data[0]) + cheby_idx)

            poly_idx += poly_size[idx]
            cheby_idx += cheby_size[idx]

    return PolycompChunkArray(<libc.stdint.uintptr_t> ptr, num_of_chunks)

def decompress_polycomp(PolycompChunkArray array):
    cdef size_t num_of_samples = array.total_num_of_samples()
    cdef np.ndarray[np.float64_t, ndim=1] samples = np.empty(num_of_samples)
    cdef int result

    result = cpc.pcomp_decompress_polycomp(<np.float64_t *> &samples.data[0],
                                           array.__ptr(), len(array))
    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("pcomp_decompress_polycomp returned code {0}"
                         .format(result))

    return samples

def decode_chunk_array(np.ndarray raw_bytes not None):
    cdef cpc.pcomp_polycomp_chunk_t** chunk_array
    cdef size_t num_of_chunks
    cdef int result

    result = cpc.pcomp_decode_chunks(&chunk_array, &num_of_chunks,
                                     <np.int8_t *> &raw_bytes.data[0])
    if result != PCOMP_STAT_SUCCESS:
        raise ValueError("pcomp_decode_chunks returned code {0}"
                         .format(result))

    return PolycompChunkArray(<libc.stdint.uintptr_t> chunk_array, num_of_chunks)
