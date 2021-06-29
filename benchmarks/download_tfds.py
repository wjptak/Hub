import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_datasets as tfds

import hub
from hub import Dataset
import numpy as np

DATASET = "coil100"

data = tfds.as_numpy(
    tfds.load(DATASET, split="train").batch(64).prefetch(tf.data.experimental.AUTOTUNE)
)
Dataset.delete(f"~/datasets/{DATASET}")
ds = Dataset(f"~/datasets/{DATASET}")

with ds:
    for sample in data:
        for col in sample:
            if col not in ds.tensors:
                ds.create_tensor(col)
            ds[col].append(sample[col])
