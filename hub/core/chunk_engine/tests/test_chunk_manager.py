from hub.core.chunk_engine import ChunkManager
from hub.core.chunk_engine.tests.util import get_random_shaped_array


def test_write():
    # TODO: test configs
    cm = ChunkManager(
        "test_storage_provider", compressor="dummy_compressor", chunk_size=256
    )

    a = get_random_shaped_array((10, 10), dtype="int32", fixed=True)
    cm.write(a)
