import tqdm
import matplotlib.pyplot as plt
from hub import Dataset

from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain

from hub.core.chunk_engine import read_tensor_meta

dataset_name = "asl_alphabet"
s3 = S3Provider("s3://hub-2.0-datasets/%s" % dataset_name)
storage = get_cache_chain(
    [MemoryProvider(dataset_name), s3],
    [
        256 * 1024 * 1024,
    ],
)

ds = Dataset(mode="r", provider=storage)
meta = read_tensor_meta("images", ds.provider)
label_names = meta["label_names"]  # TODO: API should do this, something like ds["labels"][0].name would be very convenient

for sample in tqdm.tqdm(ds, total=len(ds), desc="iterating through ds"):
    x, t = sample["images"].numpy(), sample["labels"].numpy()
    label_name = label_names[t[0][0]]
    break
