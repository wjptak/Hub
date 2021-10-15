import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from typing import List
from plot_hub_shuffle_uniformity import main


def _get_indices(dataset_length: int, *args, **kwargs) -> List[int]:
    data = np.arange(dataset_length)
    dataset = TensorDataset(torch.tensor(data))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    indices = []
    for batch in loader:
        x = batch[0]
        indices.extend(x.tolist())
    return indices


if __name__ == "__main__":
    main("NOT-hub pytorch shuffle=True uniformity", indices_func=_get_indices, is_hub=False)