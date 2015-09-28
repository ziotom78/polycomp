#!/usr/bin/env python
# -*- mode: cython -*-

from libc.stdint cimport *

cdef extern from "libpolycomp.h":
    void pcomp_version(int* major, int* minor)
 
    # RLE
    
    size_t pcomp_rle_bufsize(size_t input_size)
 
    int pcomp_compress_rle_int8(int8_t* output_buf, size_t* output_size,
                                const int8_t* input_buf, size_t input_size)
    int pcomp_compress_rle_int16(int16_t* output_buf, size_t* output_size,
                                 const int16_t* input_buf,
                                 size_t input_size)
    int pcomp_compress_rle_int32(int32_t* output_buf, size_t* output_size,
                                 const int32_t* input_buf,
                                 size_t input_size)
    int pcomp_compress_rle_int64(int64_t* output_buf, size_t* output_size,
                                 const int64_t* input_buf,
                                 size_t input_size)
 
    int pcomp_compress_rle_uint8(uint8_t* output_buf, size_t* output_size,
                                 const uint8_t* input_buf,
                                 size_t input_size)
    int pcomp_compress_rle_uint16(uint16_t* output_buf, size_t* output_size,
                                  const uint16_t* input_buf,
                                  size_t input_size)
    int pcomp_compress_rle_uint32(uint32_t* output_buf, size_t* output_size,
                                  const uint32_t* input_buf,
                                  size_t input_size)
    int pcomp_compress_rle_uint64(uint64_t* output_buf, size_t* output_size,
                                  const uint64_t* input_buf,
                                  size_t input_size)
 
    int pcomp_decompress_rle_int8(int8_t* output_buf, size_t output_size,
                                  const int8_t* input_buf,
                                  size_t input_size)
    int pcomp_decompress_rle_int16(int16_t* output_buf, size_t output_size,
                                   const int16_t* input_buf,
                                   size_t input_size)
    int pcomp_decompress_rle_int32(int32_t* output_buf, size_t output_size,
                                   const int32_t* input_buf,
                                   size_t input_size)
    int pcomp_decompress_rle_int64(int64_t* output_buf, size_t output_size,
                                   const int64_t* input_buf,
                                   size_t input_size)
 
    int pcomp_decompress_rle_uint8(uint8_t* output_buf, size_t output_size,
                                   const uint8_t* input_buf,
                                   size_t input_size)
    int pcomp_decompress_rle_uint16(uint16_t* output_buf,
                                    size_t output_size,
                                    const uint16_t* input_buf,
                                    size_t input_size)
    int pcomp_decompress_rle_uint32(uint32_t* output_buf,
                                    size_t output_size,
                                    const uint32_t* input_buf,
                                    size_t input_size)
    int pcomp_decompress_rle_uint64(uint64_t* output_buf,
                                    size_t output_size,
                                    const uint64_t* input_buf,
                                    size_t input_size)
     
    # Differenced RLE
    size_t pcomp_diffrle_bufsize(size_t input_size)
    
    int pcomp_compress_diffrle_int8(int8_t* output_buf, size_t* output_size,
                                    const int8_t* input_buf,
                                    size_t input_size)
    int pcomp_compress_diffrle_int16(int16_t* output_buf,
                                     size_t* output_size,
                                     const int16_t* input_buf,
                                     size_t input_size)
    int pcomp_compress_diffrle_int32(int32_t* output_buf,
                                     size_t* output_size,
                                     const int32_t* input_buf,
                                     size_t input_size)
    int pcomp_compress_diffrle_int64(int64_t* output_buf,
                                     size_t* output_size,
                                     const int64_t* input_buf,
                                     size_t input_size)
    
    int pcomp_compress_diffrle_uint8(uint8_t* output_buf,
                                     size_t* output_size,
                                     const uint8_t* input_buf,
                                     size_t input_size)
    int pcomp_compress_diffrle_uint16(uint16_t* output_buf,
                                      size_t* output_size,
                                      const uint16_t* input_buf,
                                      size_t input_size)
    int pcomp_compress_diffrle_uint32(uint32_t* output_buf,
                                      size_t* output_size,
                                      const uint32_t* input_buf,
                                      size_t input_size)
    int pcomp_compress_diffrle_uint64(uint64_t* output_buf,
                                      size_t* output_size,
                                      const uint64_t* input_buf,
                                      size_t input_size)
    
    int pcomp_decompress_diffrle_int8(int8_t* output_buf,
                                      size_t output_size,
                                      const int8_t* input_buf,
                                      size_t input_size)
    int pcomp_decompress_diffrle_int16(int16_t* output_buf,
                                       size_t output_size,
                                       const int16_t* input_buf,
                                       size_t input_size)
    int pcomp_decompress_diffrle_int32(int32_t* output_buf,
                                       size_t output_size,
                                       const int32_t* input_buf,
                                       size_t input_size)
    int pcomp_decompress_diffrle_int64(int64_t* output_buf,
                                       size_t output_size,
                                       const int64_t* input_buf,
                                       size_t input_size)
    
    int pcomp_decompress_diffrle_uint8(uint8_t* output_buf,
                                       size_t output_size,
                                       const uint8_t* input_buf,
                                       size_t input_size)
    int pcomp_decompress_diffrle_uint16(uint16_t* output_buf,
                                        size_t output_size,
                                        const uint16_t* input_buf,
                                        size_t input_size)
    int pcomp_decompress_diffrle_uint32(uint32_t* output_buf,
                                        size_t output_size,
                                        const uint32_t* input_buf,
                                        size_t input_size)
    int pcomp_decompress_diffrle_uint64(uint64_t* output_buf,
                                        size_t output_size,
                                        const uint64_t* input_buf,
                                        size_t input_size)

    # Quantization

    ctypedef struct pcomp_quant_params_t:
        pass

    pcomp_quant_params_t* pcomp_init_quant_params(size_t element_size,
                                                  size_t bits_per_sample)
    void pcomp_free_quant_params(pcomp_quant_params_t* params)

    size_t pcomp_quant_element_size(const pcomp_quant_params_t* params)
    size_t pcomp_quant_bits_per_sample(const pcomp_quant_params_t* params)

    size_t pcomp_quant_bufsize(size_t input_size,
                               const pcomp_quant_params_t* params)

    int pcomp_compress_quant_float(void* output_buf, size_t* output_size,
                                   const float* input_buf,
                                   size_t input_size,
                                   pcomp_quant_params_t* params)
    int pcomp_compress_quant_double(void* output_buf, size_t* output_size,
                                    const double* input_buf,
                                    size_t input_size,
                                    pcomp_quant_params_t* params)

    int pcomp_decompress_quant_float(float* output_buf, size_t output_size,
                                     const void* input_buf,
                                     size_t input_size,
                                     const pcomp_quant_params_t* params)
    int pcomp_decompress_quant_double(double* output_buf,
                                      size_t output_size,
                                      const void* input_buf,
                                      size_t input_size,
                                      const pcomp_quant_params_t* params)
    
