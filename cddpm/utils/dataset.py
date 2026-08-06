# utils/dataset.py
import os, logging, numpy as np, torch
from torch.utils.data import Dataset
from .condition import extract_conditions_from_filename

class CustomMatrixDataset(Dataset):
    def __init__(self, root_dir,
                 data_max: float,
                 mean_val1: float = 0.0, std_val1: float = 1.0,
                 mean_val2: float = 0.0, std_val2: float = 1.0,
                 resize_to: int | None = None):
        """
        Loads ONLY 48x48 .npy matrices (2D) and applies:
            1) Anscombe transform
            2) global [-1,1] normalization (data_max must be in Anscombe domain)
        """

        self.data_max = float(data_max)
        if not (self.data_max > 0):
            raise ValueError("data_max must be > 0 and measured AFTER Anscombe transform.")

        self.mean1, self.std1 = float(mean_val1), float(std_val1)
        self.mean2, self.std2 = float(mean_val2), float(std_val2)
        self.resize_to = resize_to  # kept for API compatibility; unused now that data is already 48x48

        imgs, conds = [], []

        for fn in sorted(os.listdir(root_dir)):
            if not fn.lower().endswith(".npy"):
                continue

            path = os.path.join(root_dir, fn)

            try:
                # ---- parse conditioning values from file name ----
                z1, z2 = extract_conditions_from_filename(
                    fn, self.mean1, self.std1, self.mean2, self.std2
                )

                # ---- load file ----
                arr = np.load(path, allow_pickle=False)
                arr = np.squeeze(arr)
                arr = np.nan_to_num(arr, copy=False)

                # ✅ REQUIRE EXACTLY 48×48 (no crop needed anymore)
                if arr.ndim != 2 or arr.shape != (48, 48):
                    raise ValueError(
                        f"Expected matrix of size (48, 48), got {arr.shape} for file: {fn}"
                    )

                # ---- convert to torch [1, H, W] ----
                t = torch.from_numpy(arr.astype(np.float32, copy=False)).unsqueeze(0)

                # ---- Anscombe ----
                t = torch.clamp(t, min=0.0)
                t = 2.0 * torch.sqrt(t + 3.0 / 8.0)

                # ---- global normalization to [-1, 1] ----
                t = (t / self.data_max) * 2.0 - 1.0

                imgs.append(t)
                conds.append(torch.tensor((z1, z2), dtype=torch.float32))

            except Exception as e:
                # We do NOT skip silently — missing / wrong shaped arrays must stop dataset build.
                raise e

        if not imgs:
            raise RuntimeError(f"No valid 48x48 .npy arrays found in {root_dir}")

        self.images = torch.stack(imgs)
        self.conditions = torch.stack(conds)

        logging.info(
            f"Loaded {len(self.images)} matrices from {root_dir}. "
            f"Final size: {tuple(self.images.shape[1:])}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.conditions[idx]


def compute_global_stats(root_dir: str, *, apply_anscombe: bool = True, clamp_min: float = 0.0):
    """
    Computes global mean, std, and max over all .npy (NxN) matrices in a directory,
    streaming (doesn't load all at once).

    If apply_anscombe=True (default), stats are computed AFTER the Anscombe transform:
        x' = 2 * sqrt( max(x, clamp_min) + 3/8 )
    Return values are thus in the transformed domain and should be used consistently
    with your dataset normalization (i.e., (t / data_max) * 2 - 1 after Anscombe).

    Args:
        root_dir: directory containing .npy files (each a 2D matrix)
        apply_anscombe: compute stats in Anscombe domain if True
        clamp_min: lower bound before sqrt for safety (default 0.0)

    Returns:
        mean (float), std (float), global_max (float)  -- all in the chosen domain
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    global_max = -float("inf")

    npy_files = [fn for fn in os.listdir(root_dir) if fn.lower().endswith(".npy")]
    if not npy_files:
        raise RuntimeError(f"No .npy files found in: {root_dir}")

    for fn in sorted(npy_files):
        arr = np.load(os.path.join(root_dir, fn))
        arr = np.squeeze(arr).astype(np.float64)
        arr = np.nan_to_num(arr, copy=False)

        if apply_anscombe:
            arr = 2.0 * np.sqrt(np.maximum(arr, clamp_min) + 3.0/8.0)

        total_sum     += arr.sum()
        total_sq_sum  += np.square(arr).sum()
        total_count   += arr.size

        max_local = arr.max()
        if max_local > global_max:
            global_max = max_local

        del arr

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean ** 2)
    std = float(np.sqrt(max(variance, 1e-12)))

    return float(mean), std, float(global_max)