from abc import ABC
import json
from typing import Any


class Cachable(ABC):
    is_valid = True

    def __init__(self, buffer: bytes = None):
        if buffer:
            self.frombuffer(buffer)

    def invalidate(self):
        if not self.is_valid:  # TODO: exception.py
            raise Exception
        self.is_valid = False

    def __len__(self):
        if not self.is_valid:
            raise Exception
        return len(self.tobytes())

    def tobytes(self) -> bytes:
        if not self.is_valid:
            raise Exception
        return bytes(json.dumps(self.__dict__), "utf-8")

    def frombuffer(self, buffer: bytes):
        if not self.is_valid:
            raise Exception
        self.__dict__.update(json.loads(buffer))
