import numpy as np

from hub.core.chunk_engine.generator import chunk

# TODO: add compression functions to _compressor_function_map
# each compressor is a function that takes in a chunk (bytes) & returns the compressed chunk (bytes).
_compressor_function_map = {None: lambda x: x}


def _assert_valid_shape(a: np.array):
    assert np.prod(a.shape) > 0, "array shape must not contain any 0s"


# TODO: delete dummy storage provider
class DummyStorageProvider:
    pass


class ChunkManager:
    """
    Manage the chunks for a single tensor.

    Args:
        compressor(str): String for the compressor to be used.
    """

    def __init__(self, compressor=None, chunk_size=2048, pickle=False):
        # TODO: change default chunk_size
        self.chunk_size = chunk_size

        # TODO: compressor selection docstring
        self.compressor = compressor

        # TODO: replace cache_chain with actual StorageProviders
        # cache_chain is an ordered list of StorageProviders, the first of which is the preferred
        # cache method, & the last is the last resort cache method.
        self.cache_chain = [DummyStorageProvider()]

        self.pickle = pickle
        if self.pickle:
            raise NotImplementedError("Pickle support is not available yet.")

    def write(self, data: np.array):
        # TODO: chunkmanager support for list(np.array)
        _assert_valid_shape(data)

        # TODO: normalize data.shape before adding to chunks

        if self.pickle:
            # TODO: pickle support
            raise NotImplementedError("Pickle support is not available yet.")
        else:
            data_bytes = data.tobytes()

        # TODO: get previous chunk's num bytes
        previous_num_bytes = 0
        for uncompressed_chunk_bytes, relative_chunk_index in chunk(
            data_bytes, previous_num_bytes, self.chunk_size
        ):
            print(data_bytes)
            exit()
