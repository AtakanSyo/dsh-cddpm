"""Reading RefleX matrices, the Anscombe transform and condition normalisation.

Files are named like dust_x_9700_0_nH_5_62e_22.npy, meaning a dust distance of
9700.0 pc (x = 0.97 for the 10 kpc source) and N_H = 5.62e22 cm^-2.
"""
import logging
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FILENAME = re.compile(r"^dust_x_(\d+)_(\d+)_nH_(\d+)_(\d+)e_(\d+)\.npy$")
EPS = 1e-8  # added to standard deviations before dividing


# --------------------------------------------------------------------------
# Anscombe transform: Poisson counts <-> approximately unit-variance values
# --------------------------------------------------------------------------
def anscombe(counts):
    return 2.0 * torch.sqrt(torch.clamp(counts, min=0.0) + 3.0 / 8.0)


def inverse_anscombe(values):
    """Algebraic inverse, used for individual realizations (not for means)."""
    return ((values / 2.0) ** 2 - 3.0 / 8.0).clamp_min(0.0)


def to_model_space(counts, data_max, use_anscombe=True):
    """Photon counts -> network input in [-1, 1].

    With use_anscombe the counts are Anscombe-transformed first; data_max is
    then the maximum of the transformed training pixels, otherwise of the raw
    counts (Model A).
    """
    values = anscombe(counts) if use_anscombe else counts
    return values / data_max * 2.0 - 1.0


def to_counts(model_values, data_max, use_anscombe=True):
    """Network output in [-1, 1] -> photon counts (clipped to the valid range)."""
    values = ((model_values + 1.0) * 0.5).clamp(0.0, 1.0) * data_max
    return inverse_anscombe(values) if use_anscombe else values


# --------------------------------------------------------------------------
# File names and conditions
# --------------------------------------------------------------------------
def parse_filename(name):
    """Return (dust distance in pc, N_H in cm^-2) from a matrix file name."""
    match = FILENAME.match(name)
    if match is None:
        raise ValueError(f"Unexpected matrix file name: {name}")
    x_int, x_dec, nh_int, nh_dec, nh_exp = match.groups()
    return float(f"{x_int}.{x_dec}"), float(f"{nh_int}.{nh_dec}e{nh_exp}")


def matrix_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix == ".npy")


def condition_stats(folder):
    """Mean and sample std of x (pc) and log10(N_H) over a training folder."""
    parsed = [parse_filename(p.name) for p in matrix_files(folder)]
    x = np.array([x_pc for x_pc, _ in parsed], dtype=np.float64)
    log_nh = np.array([math.log10(nh) for _, nh in parsed], dtype=np.float64)
    return {
        "mean_c1": float(x.mean()), "std_c1": float(x.std(ddof=1)),
        "mean_c2": float(log_nh.mean()), "std_c2": float(log_nh.std(ddof=1)),
    }


def normalize_condition(x_pc, nh, stats):
    """(x in pc, N_H) -> z-scored conditioning vector (c1, c2)."""
    return (
        (x_pc - stats["mean_c1"]) / (stats["std_c1"] + EPS),
        (math.log10(nh) - stats["mean_c2"]) / (stats["std_c2"] + EPS),
    )


def global_stats(folder, use_anscombe=True):
    """Mean, std and maximum of the training pixels (Anscombe-transformed or raw).

    Only the maximum is used: it maps every image to [-1, 1].
    """
    total = total_sq = count = 0.0
    maximum = -float("inf")
    for path in matrix_files(folder):
        values = np.nan_to_num(np.squeeze(np.load(path)).astype(np.float64))
        if use_anscombe:
            values = 2.0 * np.sqrt(np.maximum(values, 0.0) + 3.0 / 8.0)
        total += values.sum()
        total_sq += np.square(values).sum()
        count += values.size
        maximum = max(maximum, values.max())
    mean = total / count
    std = float(np.sqrt(max(total_sq / count - mean**2, 1e-12)))
    return float(mean), std, float(maximum)


# --------------------------------------------------------------------------
# Training dataset
# --------------------------------------------------------------------------
class HaloDataset(Dataset):
    """All matrices of one folder, in model space [-1, 1], with z-scored conditions."""

    def __init__(self, folder, data_max, stats, image_size=48, use_anscombe=True):
        images, conditions = [], []
        for path in matrix_files(folder):
            counts = np.nan_to_num(np.squeeze(np.load(path, allow_pickle=False)))
            if counts.shape != (image_size, image_size):
                raise ValueError(f"{path.name}: expected {image_size}x{image_size}, got {counts.shape}")
            counts = torch.from_numpy(counts.astype(np.float32)).unsqueeze(0)
            images.append(to_model_space(counts, data_max, use_anscombe))
            conditions.append(torch.tensor(normalize_condition(*parse_filename(path.name), stats),
                                           dtype=torch.float32))
        if not images:
            raise RuntimeError(f"No .npy matrices found in {folder}")
        self.images = torch.stack(images)
        self.conditions = torch.stack(conditions)
        logging.info("Loaded %d matrices from %s", len(self.images), folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.conditions[index]
