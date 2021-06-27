from hub.core.meta.meta import Meta


class IndexMeta(Meta):
    def __init__(self, buffer: bytes = None):
        self.tensors = []

        super().__init__(buffer)
