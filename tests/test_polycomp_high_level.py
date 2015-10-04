import numpy as np
import pypolycomp

def test_high_level_polycom():
    max_error = 0.1

    values = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 6.0, 7.0, 9.0],
                      dtype='float64')
    params = pypolycomp.Polycomp(4, 2, max_error,
                                 pypolycomp.PCOMP_ALG_USE_CHEBYSHEV)
    chunks = pypolycomp.compress_polycomp(values, params)
    assert len(chunks) == 3

    decompr = pypolycomp.decompress_polycomp(chunks)
    assert np.all(np.abs(decompr - values) < max_error)
