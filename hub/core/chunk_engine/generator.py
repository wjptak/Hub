from typing import Generator, Tuple

def chunk_along_primary_axis(
    content_bytes: bytes, previous_num_bytes: int, chunk_size: int
) -> Generator[Tuple[bytes, int], None, None]:
