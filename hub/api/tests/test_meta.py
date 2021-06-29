import pytest
from hub import Dataset


def test_dataset_meta(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)

    assert ds.meta.tensors == []

    ds.create_tensor("image")

    assert ds.meta.tensors == ["image"]

    del ds

    ds_new = Dataset(local_storage.root)

    assert ds_new.meta.tensors == ["image"]

    ds_new.delete()
