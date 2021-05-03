import numpy as np

from hub.core.chunk_engine.generator import chunk_along_all_axes

from util import get_random_shaped_array

import pytest


test_data = [
    ("int32", 4096, (100, 100), True),
]


@pytest.mark.parametrize(
    "dtype,chunk_size,data_max_shape,data_is_fixed_shape", test_data
)
def test_perfect_fit(dtype, chunk_size, data_max_shape, data_is_fixed_shape):
    assert len(data_max_shape) > 1, "Data should be batched for this test."

    batched_samples = get_random_shaped_array(
        data_max_shape, dtype, fixed=data_is_fixed_shape
    )

    for chunk, chunk_position in chunk_along_all_axes(batched_samples, chunk_size):
        break
