# data.py
#================================================================================================================================================
# Data downloader, checker, converter.
#================================================================================================================================================


from __future__ import annotations
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from config import DATASET_PATH
except ImportError:
    DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "turbulent_radiative_layer_2D" / "data"










#==============================================================================================================
# PATH UTILITIES
#==============================================================================================================

def get_dataset_root(dataset_path: Optional[Union[str, Path]] = None) -> Path:
    if dataset_path is None:
        dataset_path = DATASET_PATH

    dataset_path = Path(dataset_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist:\n{dataset_path}"
        )

    return dataset_path





def get_split_dir(
    split: str,
    dataset_path: Optional[Union[str, Path]] = None,
) -> Path:
    dataset_root = get_dataset_root(dataset_path)

    split_dir = dataset_root / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory does not exist:\n{split_dir}"
        )

    return split_dir





def get_split_files(
    split: str,
    dataset_path: Optional[Union[str, Path]] = None,
    max_files: Optional[int] = None,
) -> List[Path]:
    split_dir = get_split_dir(
        split = split,
        dataset_path = dataset_path,
    )

    files = sorted(
        list(split_dir.glob("*.h5"))
        + list(split_dir.glob("*.hdf5"))
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No .h5 or .hdf5 files found in:\n{split_dir}"
        )

    if max_files is not None:
        files = files[:max_files]

    return files










#==============================================================================================================
# HDF5 INSPECTION
#==============================================================================================================

def list_h5_datasets(file_path: Union[str, Path]) -> Dict[str, Tuple[int, ...]]:
    file_path = Path(file_path)

    datasets = {}

    with h5py.File(file_path, "r") as h5_file:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = tuple(obj.shape)

        h5_file.visititems(visitor)

    return datasets





def print_h5_structure(file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)

    print(f"\nHDF5 structure for: {file_path}\n")

    datasets = list_h5_datasets(file_path)

    for key, shape in datasets.items():
        print(f"{key}: {shape}")










#==============================================================================================================
# ARRAY SHAPE STANDARDIZATION
#==============================================================================================================

def _is_numeric_dataset(dataset: h5py.Dataset) -> bool:
    return np.issubdtype(dataset.dtype, np.number)





def _collect_numeric_arrays(h5_file: h5py.File) -> Dict[str, np.ndarray]:
    arrays = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and _is_numeric_dataset(obj):
            arr = np.asarray(obj)

            if arr.ndim >= 3:
                arrays[name] = arr

    h5_file.visititems(visitor)

    return arrays





def _to_nt_hw_c(array: np.ndarray) -> np.ndarray:
    """
    Convert common simulation tensor layouts into:

        N x T x H x W x C

    Supported common layouts:

        T x H x W
        T x H x W x C
        T x C x H x W
        N x T x H x W
        N x T x H x W x C
        N x T x C x H x W
    """

    array = np.asarray(array)

    if array.ndim == 3:
        array = array[None, ..., None]
        return array

    if array.ndim == 4:
        if array.shape[-1] <= 32:
            array = array[None, ...]
            return array

        if array.shape[1] <= 32:
            array = np.transpose(array, (0, 2, 3, 1))
            array = array[None, ...]
            return array

        array = array[..., None]
        return array

    if array.ndim == 5:
        if array.shape[-1] <= 32:
            return array

        if array.shape[2] <= 32:
            array = np.transpose(array, (0, 1, 3, 4, 2))
            return array

    raise ValueError(
        f"Could not standardize array with shape {array.shape}."
    )





def _choose_best_array(arrays: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    """
    Select the most likely full state tensor from an HDF5 file.

    Preference:
        1. Largest numeric array with ndim >= 4
        2. Largest numeric array with ndim == 3
    """

    if len(arrays) == 0:
        raise ValueError("No usable numeric arrays found in HDF5 file.")

    candidates = sorted(
        arrays.items(),
        key = lambda item: (item[1].ndim, item[1].size),
        reverse = True,
    )

    return candidates[0]










#==============================================================================================================
# DATA LOADING
#==============================================================================================================

def load_h5_state(
    file_path: Union[str, Path],
    key: Optional[str] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load one HDF5 file and return the state tensor as:

        N x T x H x W x C
    """

    file_path = Path(file_path)

    with h5py.File(file_path, "r") as h5_file:
        if key is not None:
            if key not in h5_file:
                raise KeyError(
                    f"Key '{key}' not found in {file_path}."
                )

            array = np.asarray(h5_file[key])

        else:
            arrays = _collect_numeric_arrays(h5_file)
            _, array = _choose_best_array(arrays)

    array = _to_nt_hw_c(array)
    array = array.astype(dtype, copy = False)

    return array





def load_split(
    split: str,
    dataset_path: Optional[Union[str, Path]] = None,
    key: Optional[str] = None,
    max_files: Optional[int] = None,
    max_trajectories: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load a full split and return:

        N x T x H x W x C
    """

    files = get_split_files(
        split = split,
        dataset_path = dataset_path,
        max_files = max_files,
    )

    states = []

    for file_path in files:
        state = load_h5_state(
            file_path = file_path,
            key = key,
            dtype = dtype,
        )

        states.append(state)

    states = np.concatenate(states, axis = 0)

    if max_trajectories is not None:
        states = states[:max_trajectories]

    return states





def load_train_valid_test(
    dataset_path: Optional[Union[str, Path]] = None,
    key: Optional[str] = None,
    max_files: Optional[int] = None,
    max_trajectories: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = load_split(
        split = "train",
        dataset_path = dataset_path,
        key = key,
        max_files = max_files,
        max_trajectories = max_trajectories,
        dtype = dtype,
    )

    valid = load_split(
        split = "valid",
        dataset_path = dataset_path,
        key = key,
        max_files = max_files,
        max_trajectories = max_trajectories,
        dtype = dtype,
    )

    test = load_split(
        split = "test",
        dataset_path = dataset_path,
        key = key,
        max_files = max_files,
        max_trajectories = max_trajectories,
        dtype = dtype,
    )

    return train, valid, test










#==============================================================================================================
# NORMALIZATION
#==============================================================================================================

class ChannelNormalizer:
    """
    Channel-wise normalizer for tensors shaped:

        N x T x H x W x C
    """

    def __init__(self, eps: float = 1.0e-8):
        self.eps = eps
        self.mean = None
        self.std = None





    def fit(self, states: np.ndarray) -> None:
        self.mean = states.mean(axis = (0, 1, 2, 3), keepdims = True)
        self.std = states.std(axis = (0, 1, 2, 3), keepdims = True)





    def transform(self, states: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fit before transform.")

        return (states - self.mean) / (self.std + self.eps)





    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fit before inverse_transform.")

        return states * (self.std + self.eps) + self.mean










#==============================================================================================================
# PATCH EXTRACTION
#==============================================================================================================

def extract_patch(
    frame: np.ndarray,
    i: int,
    j: int,
    patch_size: int = 3,
    padding: str = "edge",
) -> np.ndarray:
    """
    Extract one local patch centered at pixel (i, j).

    Input:
        frame: H x W x C

    Output:
        patch: patch_size x patch_size x C
    """

    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd.")

    radius = patch_size // 2

    padded = np.pad(
        frame,
        pad_width = (
            (radius, radius),
            (radius, radius),
            (0, 0),
        ),
        mode = padding,
    )

    i_pad = i + radius
    j_pad = j + radius

    patch = padded[
        i_pad - radius:i_pad + radius + 1,
        j_pad - radius:j_pad + radius + 1,
        :,
    ]

    return patch





def flatten_patch(patch: np.ndarray) -> np.ndarray:
    return patch.reshape(-1)





def iter_patch_pairs(
    states: np.ndarray,
    patch_size: int = 3,
    padding: str = "edge",
) -> Generator[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]], None, None]:
    """
    Iterate over all local adjacent-time patch pairs.

    Input:
        states: N x T x H x W x C

    Yields:
        patch_t: flattened 3 x 3 x C patch
        patch_tp1: flattened 3 x 3 x C patch
        metadata: (trajectory_index, time_index, i, j)
    """

    if states.ndim != 5:
        raise ValueError(
            f"Expected states with shape N x T x H x W x C, got {states.shape}."
        )

    n_traj, n_time, height, width, _ = states.shape

    for n in range(n_traj):
        for t in range(n_time - 1):
            frame_t = states[n, t]
            frame_tp1 = states[n, t + 1]

            for i in range(height):
                for j in range(width):
                    patch_t = extract_patch(
                        frame = frame_t,
                        i = i,
                        j = j,
                        patch_size = patch_size,
                        padding = padding,
                    )

                    patch_tp1 = extract_patch(
                        frame = frame_tp1,
                        i = i,
                        j = j,
                        patch_size = patch_size,
                        padding = padding,
                    )

                    yield (
                        flatten_patch(patch_t),
                        flatten_patch(patch_tp1),
                        (n, t, i, j),
                    )










#==============================================================================================================
# TORCH DATASETS
#==============================================================================================================

class TemporalPairDataset(Dataset):
    """
    Dataset of full-state adjacent timestamp pairs.

    Each sample is:

        x_t, x_tp1

    where both are shaped:

        H x W x C
    """

    def __init__(self, states: np.ndarray):
        if states.ndim != 5:
            raise ValueError(
                f"Expected states with shape N x T x H x W x C, got {states.shape}."
            )

        self.states = states
        self.n_traj = states.shape[0]
        self.n_time = states.shape[1]

        self.indices = []

        for n in range(self.n_traj):
            for t in range(self.n_time - 1):
                self.indices.append((n, t))





    def __len__(self) -> int:
        return len(self.indices)





    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n, t = self.indices[index]

        x_t = torch.from_numpy(self.states[n, t]).float()
        x_tp1 = torch.from_numpy(self.states[n, t + 1]).float()

        return x_t, x_tp1





class PatchPairDataset(Dataset):
    """
    Dataset of local patch pairs.

    Each sample is:

        flattened_patch_t, flattened_patch_tp1, metadata

    where each flattened patch has dimension:

        D = patch_size * patch_size * C
    """

    def __init__(
        self,
        states: np.ndarray,
        patch_size: int = 3,
        padding: str = "edge",
    ):
        if states.ndim != 5:
            raise ValueError(
                f"Expected states with shape N x T x H x W x C, got {states.shape}."
            )

        self.states = states
        self.patch_size = patch_size
        self.padding = padding

        self.n_traj, self.n_time, self.height, self.width, self.channels = states.shape

        self.indices = []

        for n in range(self.n_traj):
            for t in range(self.n_time - 1):
                for i in range(self.height):
                    for j in range(self.width):
                        self.indices.append((n, t, i, j))





    def __len__(self) -> int:
        return len(self.indices)





    def __getitem__(self, index: int):
        n, t, i, j = self.indices[index]

        frame_t = self.states[n, t]
        frame_tp1 = self.states[n, t + 1]

        patch_t = extract_patch(
            frame = frame_t,
            i = i,
            j = j,
            patch_size = self.patch_size,
            padding = self.padding,
        )

        patch_tp1 = extract_patch(
            frame = frame_tp1,
            i = i,
            j = j,
            patch_size = self.patch_size,
            padding = self.padding,
        )

        patch_t = torch.from_numpy(flatten_patch(patch_t)).float()
        patch_tp1 = torch.from_numpy(flatten_patch(patch_tp1)).float()

        metadata = torch.tensor(
            [n, t, i, j],
            dtype = torch.long,
        )

        return patch_t, patch_tp1, metadata










#==============================================================================================================
# QUICK TEST
#==============================================================================================================

if __name__ == "__main__":
    dataset_root = get_dataset_root()

    print(f"Dataset root: {dataset_root}")

    train_files = get_split_files(
        split = "train",
        max_files = 1,
    )

    print(f"\nFirst training file:\n{train_files[0]}")

    print_h5_structure(train_files[0])

    train = load_split(
        split = "train",
        max_files = 1,
        max_trajectories = 1,
    )

    print(f"\nLoaded train shape: {train.shape}")
    print("Expected shape: N x T x H x W x C")

    dataset = PatchPairDataset(
        states = train,
        patch_size = 3,
    )

    patch_t, patch_tp1, metadata = dataset[0]

    print(f"\nOne flattened patch_t shape: {patch_t.shape}")
    print(f"One flattened patch_tp1 shape: {patch_tp1.shape}")
    print(f"Metadata: {metadata.tolist()}")
