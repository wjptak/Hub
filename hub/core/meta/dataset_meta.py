from hub.core.meta.meta import Meta


class DatasetMeta(Meta):
    def __init__(self, buffer: bytes = None):
        self.tensors = []

        super().__init__(buffer)
