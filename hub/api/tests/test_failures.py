import numpy as np
import pytest

from hub import Dataset

from hub.tests.common import TENSOR_KEY
from hub.util.exceptions import (
    DynamicTensorNumpyError,
    TensorAlreadyExistsError,
    TensorInvalidSampleShapeError,
    TensorMetaInvalidHtypeOverwriteValue,
)


@pytest.mark.xfail(raises=TensorInvalidSampleShapeError, strict=True)
def test_shape_length_mismatch(memory_ds: Dataset):
    a1 = np.ones((5, 20))
    a2 = np.ones((5, 20, 2))

    tensor = memory_ds.create_tensor(TENSOR_KEY)
    tensor.append(a1)
    tensor.append(a2)


@pytest.mark.xfail(raises=TensorAlreadyExistsError, strict=True)
def test_tensor_already_exists(memory_ds: Dataset):
    memory_ds.create_tensor(TENSOR_KEY)
    memory_ds.create_tensor(TENSOR_KEY)


@pytest.mark.xfail(raises=TensorMetaInvalidHtypeOverwriteValue, strict=True)
@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_invalid_chunk_sizes(memory_ds: Dataset, chunk_size):
    memory_ds.create_tensor(TENSOR_KEY, chunk_size=chunk_size)


@pytest.mark.xfail(raises=TensorMetaInvalidHtypeOverwriteValue, strict=True)
@pytest.mark.parametrize("dtype", [1, False, "floatf", "intj", "foo", "bar"])
def test_invalid_dtypes(memory_ds: Dataset, dtype):
    memory_ds.create_tensor(TENSOR_KEY, dtype=dtype)


@pytest.mark.xfail(raises=DynamicTensorNumpyError, strict=True)
def test_dynamic_as_numpy(memory_ds: Dataset):
    a1 = np.ones((9, 23))
    a2 = np.ones((99, 2))

    tensor = memory_ds.create_tensor(TENSOR_KEY)
    tensor.append(a1)
    tensor.append(a2)

    # aslist=False, but a1 / a2 are not the same shape
    tensor.numpy(aslist=False)
