import numpy as np
from typing import List

import hub
from hub.constants import MB

from kolmogorov_complexity import kolmogorov_complexity_ratio

import matplotlib.pyplot as plt


EXISTING_LOADERS = {}


def _dummy_loader(num_samples: int, shuffle_cache_size: int) -> hub.Dataset:
    uri = f"./plot_uniformity_dataset_{num_samples}"

    if uri in EXISTING_LOADERS:
        return EXISTING_LOADERS[uri]

    if hub.exists(uri):
        ds = hub.load(uri)
    else:
        ds = hub.empty(uri)
        ds.create_tensor("zeros")
        ds.zeros.extend([0] * num_samples)

    loader = ds.pytorch(shuffle=True, buffer_size=shuffle_cache_size)
    EXISTING_LOADERS[uri] = loader
    return loader

def _get_indices(dataset_length: int, shuffle_cache_size: int, clear_history: bool) -> List[int]:
    loader = _dummy_loader(dataset_length, shuffle_cache_size)
    list(loader)
    cache = loader.dataset.cache
    history = cache.indices_history.copy()
    if clear_history:
        cache.indices_history.clear()
    return history

def plot_uniformity(dataset_lengths: List[int], shuffle_cache_sizes: List[int], epochs: int, verbose=False):
    
    fig, ax = plt.subplots()

    for shuffle_cache_size in shuffle_cache_sizes:
        ratios_for_shuffle_cache_size = []

        for num_samples in dataset_lengths:
            # NOTE: a good ratio is close to 0.5 (uniform)
            # a ratio closer to 1 means the data is non-random

            ratios = []
            for _ in range(epochs):
                indices_history = _get_indices(num_samples, shuffle_cache_size, clear_history=True)
                ratios.append(kolmogorov_complexity_ratio(indices_history))
            mean_ratio = np.mean(ratios)
            ratios_for_shuffle_cache_size.append(mean_ratio)

            if verbose:
                print(f"N={num_samples}, cache={shuffle_cache_size}, mean_ratio={mean_ratio}")

        ax.plot(dataset_lengths, ratios_for_shuffle_cache_size, label=f"cache_size={shuffle_cache_size}")

    # ax.hlines(0.5, 0, max(dataset_lengths), linestyles="dashed", label="target (uniformly random)")

    ax.set_ylabel("uniformity measure (0.5=uniform, 1=non-random)")
    ax.set_xlabel("# of samples")

    ax.legend()
    plt.show()

if __name__ == "__main__":
    epochs = 100
    epoch_averaging = False

    # cache_sizes = [16 * MB, 32 * MB, 64 * MB]
    cache_sizes = [16 * MB]

    plot_uniformity(
        np.linspace(100, 1000, num=10, dtype=int), 
        cache_sizes, 
        epochs, 
        verbose=True,
    )
