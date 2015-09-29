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

    # Polynomial fitting

    ctypedef struct pcomp_poly_fit_data_t:
        pass

    pcomp_poly_fit_data_t* pcomp_init_poly_fit(size_t num_of_points,
                                               size_t num_of_coeffs)
    void pcomp_free_poly_fit(pcomp_poly_fit_data_t* poly_fit)
    int pcomp_run_poly_fit(pcomp_poly_fit_data_t* poly_fit, double* coeffs,
                           const double* points)

    # Chebyshev transform

    ctypedef struct pcomp_chebyshev_t:
        pass

    pcomp_chebyshev_t*pcomp_init_chebyshev(size_t num_of_elements,
                                           int dir)
    void pcomp_free_chebyshev(pcomp_chebyshev_t* plan)
    int pcomp_run_chebyshev(pcomp_chebyshev_t* plan,
                            int dir, double* output,
                            const double* input)

    # Polynomial compression (low-level functions)

    ctypedef struct pcomp_polycomp_t:
        pass

    ctypedef struct pcomp_polycomp_chunk_t:
        pass

    pcomp_polycomp_chunk_t* pcomp_init_chunk(size_t num_of_elements);
    void pcomp_free_chunk(pcomp_polycomp_chunk_t* chunk);

    size_t pcomp_chunk_num_of_elements(const pcomp_polycomp_chunk_t* chunk)
    int pcomp_chunk_is_compressed(const pcomp_polycomp_chunk_t* chunk)
    const double* pcomp_chunk_uncompressed_data(const pcomp_polycomp_chunk_t* chunk)
    size_t pcomp_chunk_num_of_poly_coeffs(const pcomp_polycomp_chunk_t* chunk)
    const double* pcomp_chunk_poly_coeffs(const pcomp_polycomp_chunk_t* chunk)
    size_t pcomp_chunk_num_of_cheby_coeffs(const pcomp_polycomp_chunk_t* chunk)
    const double* pcomp_chunk_cheby_coeffs(const pcomp_polycomp_chunk_t* chunk)

    void pcomp_straighten(double* output, const double* input,
                          size_t num_of_elements, double period)

    pcomp_polycomp_t* pcomp_init_polycomp(size_t num_of_samples, size_t num_of_coeffs,
                                          double max_allowable_error,
                                          int algorithm)
    void pcomp_free_polycomp(pcomp_polycomp_t* params)

    int pcomp_run_polycomp_on_chunk(pcomp_polycomp_t* params,
                                    const double* input,
                                    size_t num_of_elements,
                                    pcomp_polycomp_chunk_t* chunk,
                                    double* max_error)

    int pcomp_decompress_polycomp_chunk(double* output,
                                        const pcomp_polycomp_chunk_t* chunk,
                                        pcomp_chebyshev_t* inv_chebyshev)

    # Polynomial compression (high-level functions)

    ctypedef struct pcomp_poly_parameters_t:
        pass

    pcomp_poly_parameters_t* pcomp_init_poly_parameters(size_t chunk_size,
                                                        size_t num_of_poly_coeffs,
                                                        double max_error,
                                                        int algorithm)
    void pcomp_free_poly_parameters(pcomp_poly_parameters_t* params)

    size_t pcomp_poly_parameters_chunk_size(const pcomp_poly_parameters_t* params)
    size_t pcomp_poly_parameters_num_of_poly_coeffs(
        const pcomp_poly_parameters_t* params)
    double pcomp_poly_parameters_max_error(const pcomp_poly_parameters_t* params)
    int pcomp_poly_parameters_algorithm(const pcomp_poly_parameters_t* params)

    int pcomp_compress_polycomp(pcomp_polycomp_chunk_t** chunk_array[],
                                size_t* num_of_chunks,
                                const double* input_buf, size_t input_size,
                                const pcomp_poly_parameters_t* params)

    size_t pcomp_total_num_of_samples(pcomp_polycomp_chunk_t* const chunk_array[],
                                      size_t num_of_chunks)

    int pcomp_decompress_polycomp(
        double* output_buf, pcomp_polycomp_chunk_t* const chunk_array[],
        size_t num_of_chunks)

    void pcomp_free_chunks(pcomp_polycomp_chunk_t* chunk_array[],
                           size_t num_of_chunks)
