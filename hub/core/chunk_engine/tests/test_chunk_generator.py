import numpy as np

from hub.core.chunk_engine.generator import chunk_along_all_axes

from util import get_random_shaped_array


def test_perfect_fit():
    dtype = "int32"
    chunk_size = 4096

    # must be batched!! (if not batched, need to add an axis of size 1)
    batched_samples = get_random_shaped_array((100, 1000), dtype, fixed=True)

    # since batched_samples is of shape (N=100, D=1000), & chunk_size=5000:
    # the output chunks should be one-dimensional (D=5000)

    for chunk, chunk_position in chunk_along_all_axes(batched_samples, chunk_size):

        pass
