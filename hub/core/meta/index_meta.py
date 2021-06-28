from typing import Dict, List, Tuple
from hub.util.keys import get_index_meta_key
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _create_entry(
    chunk_names: List[str],
    start_byte: int,
    end_byte: int,
    shape: Tuple[int],
) -> dict:
    # TODO: replace with `SampleMeta` class

    entry = {
        "chunk_names": chunk_names,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "shape": shape,
    }

    return entry


class IndexMeta(Meta):
    def __init__(self, buffer: bytes = None):
        self.entries = []

        super().__init__(buffer)

    def add_entry(
        self,
        chunk_names: List[str],
        start_byte: int,
        end_byte: int,
        shape: Tuple[int],
    ):
        self.entries.append(_create_entry(chunk_names, start_byte, end_byte, shape))
