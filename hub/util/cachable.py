from abc import ABC
import json


class Cachable(ABC):
    is_valid = True

    def __init__(self, buffer: bytes = None):
        if buffer:
            self.frombuffer(buffer)

    def invalidate(self):
        self.is_valid = False

    def __len__(self):
        return len(self.tobytes())

    def tobytes(self) -> bytes:
        return bytes(json.dumps(self.__dict__), "utf-8")

    def frombuffer(self, buffer: bytes):
        self.__dict__.update(json.loads(buffer))
