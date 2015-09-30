import pypolycomp
import numpy as np

def test_compression():
    for cur_type in (np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64):
        compressed = pypolycomp.rle_compress(np.array([1, 1, 1, 2, 3], dtype=cur_type))
        assert np.all(compressed == np.array([3, 1, 1, 2, 1, 3], dtype=cur_type))

