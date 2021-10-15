import numpy as np
import gzip

import warnings as w


DO_WARNINGS = True


def kolmogorov_complexity_ratio(indices):
    """Gets the ratio of the `1 - (len(gzip_compressed_bytes) / len(uncompressed_bytes))` of the derivative of `X`. 
    This is a ratio measure of the kolmogorov complexity.

    Warns when the incoming data is not index-like (sequence of integers in the range [0, len(X) - 1] 
    optionally shuffled).

    Outputs are between 0-1. Uniformly random data is close
    """

    indices = np.asarray(indices)

    if DO_WARNINGS:
        if indices.size > 0:
            w.warn(f"No indices were present.")
        if min(indices) != 0:
            w.warn(f"Indices list min was expected to be 0, but instead it is {min(indices)}.")
        if list(sorted(indices)) != np.arange(max(indices)):
            w.warn(f"Indices do not include all values from 0-N.")

    incoming_distribution = np.diff(indices)

    uncompressed_bytes = incoming_distribution.tobytes()
    compressed_bytes = gzip.compress(uncompressed_bytes)

    ratio = len(compressed_bytes) / len(uncompressed_bytes)
    return 1-ratio


if __name__ == "__main__":
    DO_WARNINGS = False

    N = 10000

    _all_same = kolmogorov_complexity_ratio([9] * N)
    print("all same", _all_same)

    _linear = kolmogorov_complexity_ratio(np.arange(N))
    print("linear", _linear)

    x = np.arange(N)
    np.random.shuffle(x)
    _random_indices = kolmogorov_complexity_ratio(x)
    print("random indices", _random_indices)

    # NOTE: we don't really care about this case, but it's good to check.
    # for this measure we only expect that the incoming data is 
    y = np.random.randint(0, N*1000, size=N)
    _uniform = kolmogorov_complexity_ratio(y)
    print("uniform", _uniform)

    assert _all_same > _linear and _linear > _random_indices and _random_indices > _uniform

    

    
