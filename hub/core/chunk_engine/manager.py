import numpy as np

from hub.core.chunk_engine.generator import chunk

# TODO: add compression functions to _compressor_function_map & remove dummy
# each compressor is a function that takes in a chunk (bytes) & returns the compressed chunk (bytes).
_compressor_function_map = {
    None: lambda x: x,
    "dummy_compressor": lambda x: x[: (len(x) // 2)],
}


def _assert_valid_shape(a: np.array):
    assert np.prod(a.shape) > 0, "array shape must not contain any 0s"


# TODO: delete dummy storage provider
class DummyStorageProvider:
    def __init__(self, url: str):
        self.url = url
        self.storage = {}

    def get_available_bytes(self):
        """
        Check the amount of available bytes that this StorageProvider has left to store.
        """

        return 9999999999

    def write(self, key: str, b: bytes):
        """
        Store bytes `b` under `key`.
        """

        if key in self.storage:
            raise Exception("key %s already exists." % key)

        self.storage[key] = b


class ChunkManager:
    """
    Manage the chunks for a single tensor.

    Args:
        storage_url(str): String with a url & path component.
            Format: [storage_provider_url]/path
            Example: s3://some-bucket/store/
        compressor(str): String for the compressor to be used.
    """

    def __init__(
        self,
        storage_url: str,
        compressor: str = None,
        chunk_size: int = 2048,
        pickle: bool = False,
    ):
        # TODO: change default chunk_size
        self.chunk_size = chunk_size

        # TODO: compressor selection docstring
        self.compress = _compressor_function_map[compressor]

        # TODO: replace cache_chain/storage with actual StorageProviders
        # cache_chain is an ordered list of StorageProviders, the first of which is the preferred
        # cache method, & the last is the last resort cache method.
        self.storage_provider = DummyStorageProvider(storage_url)
        self.cache_chain = [self.storage_provider]

        # TODO: chunk & store `index_map` to a StorageProvider.
        self.index_map = []

        self.pickle = pickle
        if self.pickle:
            raise NotImplementedError("Pickle support is not available yet.")

    def get_cache(self, num_bytes: int):
        """
        Goes through `self.cache_chain` & gets the first one available with sufficient space.

        Args:
            num_bytes(int): Number of bytes needed to cache.

        Returns:
            cache(StorageProvider): StorageProvider that can be used to cache incomplete chunks.
        """

        for cache in self.cache_chain:
            if num_bytes <= cache.get_available_bytes():
                return cache

        # TODO: failure to get cache from cache_chain in exceptions.py
        raise Exception("No cache space available.")

    def write(self, data: np.array, batched: bool = True):
        """
        Chunk data & write to this ChunkManager's StorageProvider.

        Args:
            data(np.array): Numpy array to be chunked/stored.
            batched(bool): If True, `data`'s first axis is treated as a batch axis.
        """

        # TODO: chunkmanager support for list(np.array)
        _assert_valid_shape(data)

        # TODO: normalize data.shape before adding to chunks

        if self.pickle:
            # TODO: pickle support
            raise NotImplementedError("Pickle support is not available yet.")
        else:
            data_bytes = data.tobytes()

        # TODO: get previous chunk's num bytes
        previous_num_bytes = None

        for uncompressed_chunk_bytes, relative_chunk_index in chunk(
            data_bytes, previous_num_bytes, self.chunk_size
        ):
            uncompressed_length = len(uncompressed_chunk_bytes)

            if uncompressed_length < self.chunk_size:
                # if this chunk is incomplete, cache it
                cache = self.get_cache(uncompressed_length)

                cache.write("incomplete_chunk", uncompressed_chunk_bytes)
                break

            if relative_chunk_index == 0:
                raise NotImplementedError(
                    "Haven't handled writing to previous chunk yet."
                )

            # compress & store full chunk
            compressed_chunk_bytes = self.compress(uncompressed_chunk_bytes)
            key = "chunk_%i" % relative_chunk_index
            self.storage_provider.write(key, compressed_chunk_bytes)

            # TODO: update index_map
