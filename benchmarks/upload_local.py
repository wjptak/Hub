import numpy as np
from time import time
import logging

logging.basicConfig(level=logging.ERROR)

from hub import Dataset
from hub import transform


class Timer:
    def __init__(self, msg="Time:"):
        self.msg = msg

    def __enter__(self, *args):
        self.start = time()

    def __exit__(self, *args):
        interval = time() - self.start
        print(self.msg, "%.2f" % interval)


DATASET = "coil100"

ds = Dataset(f"~/datasets/{DATASET}")

Dataset.delete(f"hub://benchislett/{DATASET}")
ds_out = Dataset(f"hub://benchislett/{DATASET}")
ds_out.create_tensor("image")

data = data2 = ds.image.numpy(aslist=True)

with Timer("Write: "):
    with ds_out:
        ds_out.image.extend(data)

ds_out.clear_cache()

with Timer("Read: "):
    with ds_out:
        data2 = ds_out.image.numpy(aslist=True)

for i in range(len(data)):
    np.testing.assert_array_equal(data[i], data2[i])
