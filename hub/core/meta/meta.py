import hub
from hub.util.cachable import Cachable


class Meta(Cachable):
    def __init__(self, buffer: bytes = None):
        self.version = hub.__version__
        super().__init__(buffer)
