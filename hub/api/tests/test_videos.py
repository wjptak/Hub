from hub.core.dataset import Dataset
import hub
from hub.constants import KB
import numpy as np
from hub.tests.dataset_fixtures import enabled_datasets


@enabled_datasets
def test_upload(ds: Dataset):
    ds.create_tensor("videos", htype="video", max_chunk_size=16 * KB)

    video1 = np.ones((100, 10, 10, 3), dtype=np.uint8)  # ~30KB
    ds.videos.append(video1)
    assert ds.videos.shape == (1, 100, 10, 10, 3)

    video2 = np.ones((80, 11, 10, 3), dtype=np.uint8)  # ~26KB
    ds.videos.append(video2)
    assert ds.videos.shape == (2, None, None, 10, 3)

    assert ds.videos.meta.sample_compression == "mp4"