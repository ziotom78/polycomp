import numpy as np
import pypolycomp

def test_straighten_noop():
    # Smooth series of values
    values = np.array([0, 1, 2, 3, 4, 5], dtype='float64')
    result = pypolycomp.straighten(values,
                                   period=3)
    # There should be no difference between input and output
    assert np.all(np.abs(result - values) < 1.0e-7)

def test_straighten():
    values = np.array([0, 1, 2, 0, 1, 2], dtype='float64')
    result = pypolycomp.straighten(values,
                                   period=3)

    assert np.all(np.abs(result - np.array([0, 1, 2, 3, 4, 5], dtype='float64')) < 1.0e-7)
