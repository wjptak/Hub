import hub



def _dummy_dataset(N):
    ds = hub.empty("./plot_uniformity_dataset", overwrite=True)
    ds.create_tensor("zeros")
    ds.zeros.extend([0] * N)
    return ds

def _get_indices(dataset_length):
    ds = _dummy_dataset(dataset_length)
    loader = ds.pytorch(shuffle=True)
    list(loader)
    return loader.dataset.cache.indices_history

def plot_uniformity(dataset_lengths):
    pass

if __name__ == "__main__":
    _get_indices(100)
