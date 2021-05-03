import numpy as np

from typing import Generator, Tuple


def chunk_along_all_axes(
    data: np.array, chunk_size: int, batched: bool = True
) -> Generator[Tuple[bytes, int], None, None]:
    if not batched:
        raise NotImplementedError(
            "Unbatched chunking along all axes is not supported yet."
        )

    num_chunk_axes = len(data.shape) - 1
    bytes_per_element = data.dtype.itemsize
    bytes_per_axis = chunk_size // num_chunk_axes
    elements_per_axis = bytes_per_axis // bytes_per_element
    chunk_shape = tuple([elements_per_axis] * num_chunk_axes)
    print(chunk_shape)

    yield None, None
