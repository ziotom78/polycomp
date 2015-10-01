import numpy as np
import pypolycomp

def test_chebyshev():
    val = np.array([0.0, 1.0, 3.0], dtype='float64')
    ch = pypolycomp.Chebyshev(val.size, pypolycomp.PCOMP_TD_DIRECT)
    coeffs = ch.run(np.array(val))

    assert coeffs.size == 3
    assert np.abs(coeffs[0] - 2.5) < 1.0e-7
    assert np.abs(coeffs[1] - (-1.5)) < 1.0e-7
    assert np.abs(coeffs[2] - 0.5) < 1.0e-7
