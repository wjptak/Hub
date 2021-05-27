# https://www.kaggle.com/grassknoted/asl-alphabet

import time
import os
import tqdm
import numpy as np
import glob
from PIL import Image
from hub import Dataset

from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain

# TODO: we shouldn't have to do this
dataset_name = "asl_alphabet"
s3 = S3Provider("s3://snark-test/%s" % dataset_name)
local = LocalProvider("./%s" % dataset_name)
storage = get_cache_chain(
    [MemoryProvider(dataset_name), s3],
    [
        256 * 1024 * 1024,
    ],
)

storage.clear()  # TODO: overwrite=True should clear
ds = Dataset(mode="w", provider=storage)
print(ds)

folder_paths = glob.glob("datasets/asl_alphabet_train/asl_alphabet_train/*")

# TODO: auto ingestion
for folder_path in folder_paths:
    label = folder_path.split("/")[-1]

    paths = glob.glob(os.path.join(folder_path, "*"))
    arrays = []
    for path in tqdm.tqdm(
        paths, total=len(paths), desc="loading %s images into arrays" % label
    ):
        img = Image.open(path)
        img = img.resize((500, 500))
        a = np.array(img)
        arrays.append(a)

    start = time.time()
    ds[label] = np.array(arrays)
    storage.flush()  # TODO: we shouldn't have to do this.
    end = time.time()
    print("time to upload label %s: %.2fs" % (label, end - start))
