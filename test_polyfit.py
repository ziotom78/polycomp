import numpy as np
import pypolycomp

def test_fit():
    val = np.array([1, 2, 3, 4], dtype='float64')
    pf = pypolycomp.PolyFit(val.size, 2)
    coeffs = pf.run(np.array(val))

    assert coeffs.size == 2
    assert np.abs(coeffs[0] - 0.0) < 1.0e-7
    assert np.abs(coeffs[1] - 1.0) < 1.0e-7
