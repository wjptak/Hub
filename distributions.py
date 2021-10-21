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
WORKERS = 0
DATASET_URI = "hub://activeloop/cifar100-train"
ds = hub.load(DATASET_URI)
num_classes = len(ds.labels.info.class_names)
BATCH_SIZE = num_classes * target_frequency


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


def plot_batches_over_time(losses_per_batch: dict):
    for key, loss_per_batch in losses_per_batch.items():
        plt.plot(loss_per_batch, label=key)

    # set log scale
    plt.legend()
    plt.show()


def get_average_losses(losses_per_batch: dict):
    return {k: np.mean(v) for k, v in losses_per_batch.items()}


def plot_losses_over_buffer_size(hub_losses: List[float], numpy_losses: List[float], buffer_sizes: List[int]):
    assert len(hub_losses) == len(numpy_losses) and len(hub_losses) == len(buffer_sizes)

    plt.title(f"quality of random shuffling ({EPOCHS} epochs)")

    plt.plot(buffer_sizes, hub_losses, label="hub")
    plt.plot(buffer_sizes, numpy_losses, label="numpy")

    plt.ylabel("mae(batch_frequency, uniform_frequencies)\nlower=better")
    plt.xlabel("buffer size")

    plt.legend()
    plt.show()


def get_losses_for_buffer_size(buffer_size: int) -> dict:
    # best_case_batch = get_best_case_batch(num_classes)
    # normal_case_batch = get_normal_case_batch(num_classes)
    # worst_case_batch = get_worst_case_batch(num_classes, 90)
    # plot_batches(
    #     [best_case_batch, normal_case_batch, worst_case_batch], 
    #     ["best case", "normal case", "worst case"],
    #     num_classes
    # )

    losses_per_batch = defaultdict(list)

    ptds = ds.pytorch(num_workers=WORKERS, batch_size=BATCH_SIZE, tensors=["images", "labels"], buffer_size=buffer_size, shuffle=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(tqdm(ptds, total=min(len(ptds) - 1, MAX_BATCHES))):  
            if i == len(ptds) - 1 or i > MAX_BATCHES:
                # skip last batch (not full)
                break

            _, T = batch

            # generate our comparison batches
            current_batch = T.flatten().numpy()
            best_case_batch = get_best_case_batch(num_classes)
            normal_case_batch = get_normal_case_batch(num_classes)
            # worst_case_batch = get_worst_case_batch(num_classes, i)


            losses = quantify_batches(
                [current_batch, best_case_batch, normal_case_batch],# , worst_case_batch], 
                ["current batch", "best case", "normal case"],# , "worst case"], 
                best_case_batch,
            )
            # print(losses)

            for key, loss in losses.items():
                losses_per_batch[key].append(loss)

            # plot_batches(
            #     [current_batch, best_case_batch, normal_case_batch, worst_case_batch], 
            #     ["current batch", "best case", "normal case", "worst case"], 
            #     num_classes
            # )

    # plot_batches_over_time(losses_per_batch)
    return get_average_losses(losses_per_batch)

if __name__ == '__main__':
    hub_losses = []
    numpy_losses = []

    buffer_sizes = [1, 10, 100, 1000, 10000]
    for buffer_size in buffer_sizes:
        losses = get_losses_for_buffer_size(buffer_size)

        hub_losses.append(losses["current batch"])
        numpy_losses.append(losses["normal case"])

    print("hub", hub_losses)
    print("numpy", numpy_losses)
    plot_losses_over_buffer_size(hub_losses, numpy_losses, buffer_sizes)
