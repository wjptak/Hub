from copy import deepcopy

import numpy as np


def make_dummy_byte_array(length: int) -> bytearray:
    """Generate a random bytearray of the provided length."""
    content = bytearray()
    a = np.random.randint(128, size=length)
    content.extend(a.tolist())
    assert len(content) == length
    return content


def get_random_chunk_size() -> int:
    return np.random.choice([8, 256, 1024, 4096])


def get_random_num_samples() -> int:
    return np.random.randint(1, 300)


def get_random_partial(chunk_size: int) -> int:
    return np.random.randint(1, chunk_size - 1)


def assert_valid_chunk(chunk: bytes, chunk_size: int):
    assert len(chunk) > 0
    assert len(chunk) <= chunk_size


def get_random_shaped_array(random_max_shape: tuple, dtype: str, fixed: bool=False) -> np.array:
    """
    Helper function for generating numpy arrays.

    Args:
        random_max_shape(tuple): Shape for the array. If Fixed=False, this will be the max_shape (random between 1 & 
            each dim value). If Fixed=True, the array will always have the shape `random_max_shape`.
        dtype(str): Datatype for the created array.
        fixed(bool): If Fixed, the array will have deterministic shape (`random_max_shape`), else it will have a random
            shape, the max of which is `random_max_shape`.

    Returns:
        np.array: Created array with the provided specifications.
    """

    if fixed:
        dims = random_max_shape
    else:
        dims = []
        if type(random_max_shape) != int:
            for max_dim in random_max_shape:
                dims.append(np.random.randint(1, max_dim + 1))

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        a = np.random.randint(low=low, high=high, size=dims, dtype=dtype)
    elif "float" in dtype:
        a = np.random.random_sample(size=dims).astype(dtype)
    elif "bool" in dtype:
        a = np.random.uniform(size=dims)
        a = a > 0.5

    return a
