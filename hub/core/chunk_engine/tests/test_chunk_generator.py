import numpy as np

from util import get_random_shaped_array


def test_perfect_fit():
    dtype = "int32"

    # must be batched!! (if not batched, need to add an axis of size 1)
    batched_samples = get_random_shaped_array((100, 1000), dtype, fixed=True)

    num_chunk_axes = len(batched_samples.shape) - 1
    bytes_per_element = np.dtype(dtype).itemsize
    elements_per_axis = 5000 // num_chunk_axes  # 5000 elements per chunk
    bytes_per_axis = elements_per_axis * bytes_per_element
    chunk_shape = tuple([elements_per_axis] * num_chunk_axes)

    # since batched_samples is of shape (N=100, D=1000), & chunk_size=5000:
    # the output chunks should be one-dimensional (D=5000)

    print(batched_samples.shape)
    print(elements_per_axis, bytes_per_axis)
    print(chunk_shape)
