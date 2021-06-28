from hub.core.meta.meta import Meta
import hub
from hub.core.storage.provider import StorageProvider
from hub.core.tests.common import parametrize_all_caches


def _assert_version(meta):
    assert meta.version == hub.__version__


@parametrize_all_caches
def test_meta(storage: StorageProvider):
    key = "dummy_meta"

    meta = Meta()
    meta.literally_anything = []

    assert meta.is_valid
    assert meta.literally_anything == []
    meta.literally_anything.append("AAAAAAHHHHHH")
    _assert_version(meta)

    storage[key] = meta

    assert type(storage[key]) == Meta

    storage.flush()

    assert meta.is_valid
    assert type(storage[key]) == Meta
    assert type(storage.next_storage[key]) == bytes

    storage.clear_cache()

    assert not meta.is_valid
    assert key not in storage.cache_storage
    assert type(storage[key]) == bytes

    del meta

    meta = Meta(storage[key])

    assert meta.literally_anything == ["AAAAAAHHHHHH"]
    _assert_version(meta)
