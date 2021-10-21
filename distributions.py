import os
from collections import defaultdict
from typing import List
import matplotlib.pyplot as plt
import hub
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import l1_loss as mae

# the target frequency is used to determine the perfect uniform batch of a batch.
target_frequency = 4

MAX_BATCHES = 500
EPOCHS = 5
WORKERS = 4
DATASET_URI = "./distributions_evaluation_dataset"
num_classes = 100
num_samples_per_class = 500
BATCH_SIZE = num_classes * target_frequency


def create_dataset():
    if os.path.isdir(DATASET_URI):
        return
    print("creating dataset", DATASET_URI)

    ds = hub.empty(DATASET_URI, overwrite=True)
    ds.create_tensor("images", dtype=np.float64)
    ds.create_tensor("labels", dtype=np.uint32)
    with ds:
        for label in range(num_classes):
            ds.images.extend(np.ones((num_samples_per_class, 32, 32, 3)))
            ds.labels.extend([label] * num_samples_per_class)
    print("dataset created")



def get_best_case_batch(num_classes: int):
    """Model performance is optimal when the average batch of batches is fully uniform."""
    values = []
    for c in range(num_classes):
        for _ in range(target_frequency):
            values.append(c)
    assert len(values) == BATCH_SIZE
    return np.array(values, dtype=int)


def get_normal_case_batch(num_classes: int):
    """This is what class batches will look like in the real world."""
    return np.random.randint(0, num_classes, size=BATCH_SIZE, dtype=int)


def get_worst_case_batch(num_classes: int, batch_idx: int):
    """Model performance is the worst when a batch contains only 1 class."""
    use_class = batch_idx % num_classes
    values = [use_class] * BATCH_SIZE
    return np.array(values, dtype=int)


def plot_batches(batches: List[np.ndarray], titles: List[str], num_classes: int):
    assert len(batches) == len(titles)

    fig, axs = plt.subplots(len(batches))

    for i in range(len(batches)):
        axs[i].title.set_text(titles[i])
        # axs[i].hist(actual_labels, bins=bins)
        axs[i].hist(batches[i], bins=num_classes)

    plt.show()


def calculate_frequencies(tensor: torch.Tensor):
    """Calculate the frequencies of each class in the tensor."""

    freq = torch.zeros(num_classes)
    for x in tensor.flatten():
        freq[x] += 1
    return freq



def quantify_batches(batches: List[np.ndarray], titles: List[str], target_batch: np.ndarray):
    """The mean absolute error of the frequencies between each `batch` in `batches` is calculated with `target_batch` as the target.
    
    Minimum loss for any given batch is 0.
    """

    losses = {}

    T = torch.tensor(target_batch)
    # freq_T = torch.unique(T, return_counts=True)[1].float()
    freq_T = calculate_frequencies(T)

    for batch, title in zip(batches, titles):
        #assert len(batch) == BATCH_SIZE, f"{title} batch length was {len(batch)} but expected {BATCH_SIZE}"

        # get frequencies of classes in the batch
        X = torch.tensor(batch)
        # freq_X = torch.unique(X, return_counts=True)[1].float()  # TODO: better way to get frequencies
        freq_X = calculate_frequencies(X)

        loss = mae(freq_X, freq_T).item()
        losses[title] = loss

    return losses


def get_hub_loss(buffer_size: int):
    shuffle = buffer_size > 0
    return 5



if __name__ == '__main__':
    create_dataset()

    buffer_sizes = [0, 1]
    hub_shuffled_losses = [get_hub_loss(buffer_size) for buffer_size in buffer_sizes]

    plt.plot(buffer_sizes, hub_shuffled_losses)
    plt.show()


