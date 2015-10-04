import pypolycomp
import numpy as np

def test_quantization():
    for cur_dtype in (np.float32, np.float64):
        values = np.array([3.06, 5.31, 2.25, 7.92, 4.86], dtype=cur_dtype)
        quant = pypolycomp.QuantParams(element_size=values.dtype.itemsize,
                                       bits_per_sample=5)
        assert np.all(np.fromstring(quant.compress(values), dtype='uint8') ==
                      np.array([36, 65, 247, 0], dtype='uint8'))

def test_decompression():
    for cur_dtype in (np.float32, np.float64):
        values = np.array([3.06, 5.31, 2.25, 7.92, 4.86], dtype=cur_dtype)
        quant = pypolycomp.QuantParams(element_size=values.dtype.itemsize,
                                       bits_per_sample=5)
        compr = quant.compress(values)
        decompr = quant.decompress(compr, len(values))

        assert np.all(np.abs(values - decompr) <= 0.186)
