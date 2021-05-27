# https://www.kaggle.com/grassknoted/asl-alphabet

import time
import os
import tqdm
import numpy as np
import glob
from PIL import Image
from hub import Dataset
from hub.core.chunk_engine import read_tensor_meta, write_tensor_meta

from hub.core.storage import MemoryProvider, S3Provider, LocalProvider
from hub.util.cache_chain import get_cache_chain

SMOKE_TEST = False
IMAGE_SIZE = (300, 300)  # all images are reshaped to this size
CHANNELS = 3

if SMOKE_TEST:
    print("\n\nWARNING! SMOKE TEST WILL ONLY USE 1 IMAGE PER CLASS!\n\n")

# TODO: we shouldn't have to do this
dataset_name = "asl_alphabet"
s3 = S3Provider("s3://hub-2.0-datasets/%s" % dataset_name)
storage = get_cache_chain(
    [MemoryProvider(dataset_name), s3],
    [
        256 * 1024 * 1024,
    ],
)

# s3.clear()  # TODO: overwrite=True should clear
storage["mnop"] = b"123"  # TODO: we should not need to do this
ds = Dataset(mode="w", provider=storage)

folder_paths = glob.glob("datasets/asl_alphabet_train/asl_alphabet_train/*")

def get_paths(parent):
    return glob.glob(os.path.join(parent, "*"))

if SMOKE_TEST:
    # only use one image per class
    NUM_IMAGES = len(folder_paths)
else:
    NUM_IMAGES = sum([len(get_paths(parent)) for parent in folder_paths])

print("num images: %i" % NUM_IMAGES)

# TODO: auto ingestion
images = np.zeros((NUM_IMAGES, *IMAGE_SIZE, CHANNELS), dtype=np.uint8)

print("total images tensor shape:", images.shape)

labels = []
label_names = []

image_idx = 0
for label, folder_path in enumerate(folder_paths):
    label_name = folder_path.split("/")[-1]
    label_names.append(label_name)
    labels.append(label)

    paths = get_paths(folder_path)

    for path in tqdm.tqdm(
        paths, total=len(paths), desc="(%i/%i) loading %s images into \"images\" array" % (label, len(folder_paths), label_name)
    ):
        img = Image.open(path)
        img = img.resize(IMAGE_SIZE)
        images[image_idx] = np.array(img, dtype=np.uint8)

        if SMOKE_TEST:
            break

print("begin data upload...")
start = time.time()

# TODO: do this incrementally (API needs append)
ds["images"] = np.array(images)
ds["labels"] = np.array(labels)

# TODO: API should handle `label_names` for us
meta = read_tensor_meta("images", storage)
meta["label_names"] = label_names
write_tensor_meta("images", storage, meta)

storage.flush()  # TODO: we shouldn't have to do this.

end = time.time()
print("time to upload data: %.2fs" % (end - start))
