import pypolycomp
import numpy as np

def test_compression():
    for cur_type in (np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64):
        compressed = pypolycomp.diffrle_compress(np.array([1, 1, 1, 2, 3],
                                                          dtype=cur_type))
        assert np.all(compressed == np.array([1, 2, 0, 2, 1],
                                             dtype=cur_type))

def test_decompression():
    for cur_type in (np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64):
        input_values = np.array(np.random.randint(100, size=1000),
                                dtype=cur_type)
        compressed = pypolycomp.diffrle_compress(input_values)
        output_values = pypolycomp.diffrle_decompress(compressed)
        assert np.all(input_values == output_values)
