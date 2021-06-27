import hub
from hub.core.storage.provider import StorageProvider
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.tests.common import parametrize_all_caches

from hub.util.keys import get_dataset_meta_key


def _assert_version(meta):
    assert meta.version == hub.__version__


@parametrize_all_caches
def test_dataset_meta(storage: StorageProvider):
    key = get_dataset_meta_key()

    ds_meta = DatasetMeta()

    assert ds_meta.is_valid
    assert ds_meta.tensors == []
    ds_meta.tensors.append("tensor")
    _assert_version(ds_meta)

    storage[key] = ds_meta

    assert type(storage[key]) == DatasetMeta

    storage.flush()

    assert ds_meta.is_valid
    assert type(storage[key]) == DatasetMeta
    assert type(storage.next_storage[key]) == bytes

    storage.clear_cache()

    assert not ds_meta.is_valid
    assert type(storage[key]) == bytes

    del ds_meta

    ds_meta = DatasetMeta(storage[key])

    assert ds_meta.tensors == ["tensor"]
