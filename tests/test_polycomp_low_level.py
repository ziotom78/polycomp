import numpy as np
import pypolycomp

def test_polycomp():
    for alg in (pypolycomp.PCOMP_ALG_USE_CHEBYSHEV,
                pypolycomp.PCOMP_ALG_NO_CHEBYSHEV):
        max_error = 0.1
        samples = np.array([1.0, 2.0, 3.0, 4.1])
        params = pypolycomp.Polycomp(samples.size, 2, max_error, alg)
        inv_cheby = pypolycomp.Chebyshev(samples.size, pypolycomp.PCOMP_TD_INVERSE)

        chunk = pypolycomp.PolycompChunk()
        chunk.init_with_num_of_samples(samples.size)

        chunk.compress(samples, params)
        assert chunk.is_compressed()
        assert np.all(np.abs(chunk.decompress(inv_cheby) - samples) < max_error)

if __name__ == "__main__":
    test_polycomp()
