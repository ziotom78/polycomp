import numpy as np
import pypolycomp

def test_polycomp():
    inv_cheby = pypolycomp.Chebyshev(samples.size, pypolycomp.PCOMP_TD_INVERSE)
    
    for alg in (pypolycomp.PCOMP_ALG_USE_CHEBYSHEV,
                pypolycomp.PCOMP_ALG_NO_CHEBYSHEV):
        max_error = 0.1
        samples = np.array([1.0, 2.0, 3.0, 4.1])
        params = pypolycomp.Polycomp(samples.size, 2, max_error, alg)

        chunk = pypolycomp.PolycompChunk(params, samples.size)

        chunk.compress(samples)
        assert np.all(np.abs(chunk.decompress(inv_cheby) - samples) < max_error)
