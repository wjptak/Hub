import matplotlib.pyplot as plt
from hub import Dataset

from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain

dataset_name = "asl_alphabet"
s3 = S3Provider("s3://hub-2.0-datasets/%s" % dataset_name)
storage = get_cache_chain(
    [MemoryProvider(dataset_name), s3],
    [
        256 * 1024 * 1024,
    ],
)

ds = Dataset(mode="r", provider=storage)

print(ds.tensors.keys())
a = ds["O"][0].numpy()
print(a.shape)
plt.imshow(a[0])
plt.show()
