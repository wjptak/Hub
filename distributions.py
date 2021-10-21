from typing import List
import matplotlib.pyplot as plt
import hub
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import l1_loss as mae

from scipy.fft import idct

target_frequency = 10

DATASET_URI = "hub://activeloop/cifar100-train"
ds = hub.load(DATASET_URI)
num_classes = len(ds.labels.info.class_names)
BATCH_SIZE = num_classes * target_frequency


def get_best_case_distribution(num_classes: int):
    """Model performance is optimal when the average distribution of batches is fully uniform."""
    values = []
    for c in range(num_classes):
        for _ in range(target_frequency):
            values.append(c)
    assert len(values) == BATCH_SIZE
    return np.array(values, dtype=int)


def get_normal_case_distribution(num_classes: int):
    """This is what class distributions will look like in the real world."""
    return np.random.randint(0, num_classes, size=BATCH_SIZE, dtype=int)


def get_worst_case_distribution(num_classes: int, batch_idx: int):
    """Model performance is the worst when a batch contains only 1 class."""
    use_class = batch_idx % num_classes
    values = [use_class] * BATCH_SIZE
    return np.array(values, dtype=int)


def plot_distributions(distributions: List[np.ndarray], titles: List[str], num_classes: int):
    assert len(distributions) == len(titles)

    fig, axs = plt.subplots(len(distributions))

    for i in range(len(distributions)):
        axs[i].title.set_text(titles[i])
        # axs[i].hist(actual_labels, bins=bins)
        axs[i].hist(distributions[i], bins=num_classes)

    plt.show()


if __name__ == '__main__':

    best_case_distribution = get_best_case_distribution(num_classes)
    normal_case_distribution = get_normal_case_distribution(num_classes)
    worst_case_distribution = get_worst_case_distribution(num_classes, 0)

    plot_distributions(
        [best_case_distribution, normal_case_distribution, worst_case_distribution], 
        ["best case", "normal case", "worst case"],
        num_classes
    )
    
    exit()
    

    # plot_dist(get_target_distribution(num_classes), "Target Distribution", num_classes)

    ptds = ds.pytorch(num_workers=2, batch_size=BATCH_SIZE, tensors=["images", "labels"])


    for i, batch in enumerate(tqdm(ptds)):  
        X, T = batch

        T = T.flatten().float()
        q = get_best_case_distribution(num_classes)
        loss = mae(T.flatten(), torch.tensor(q).float()).item()
        print(loss)

        # generate our comparison distributions
        best_case_distribution = get_best_case_distribution(num_classes)
        normal_case_distribution = get_normal_case_distribution(num_classes)
        worst_case_distribution = get_worst_case_distribution(num_classes, i)

        plot_distributions([best_case_distribution, normal_case_distribution, worst_case_distribution], ["best case", "normal case", "worst case"])

        if i == len(ptds) - 1:
            # skip last batch (not full)
            break

        

        break