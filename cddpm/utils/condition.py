import os
import math
import numpy as np

# -----------------------------------------------------------------------------
# Compute dataset-wide mean and std for each condition
# -----------------------------------------------------------------------------
def get_condition_stats_from_filenames(data_dir):
    """
    Scans image filenames in a dataset folder and computes
    the mean and std of the two physical parameters:
        - cond1: e.g. x (grain size)
        - cond2: log10(nH)

    Returns:
        mean1, std1, mean2, std2
    """
    cond1_vals, cond2_vals = [], []
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith((".npy")):
            continue

        name, _ = os.path.splitext(fn)
        parts = name.split("_")
        if len(parts) < 6:
            continue

        try:
            c1_raw = float(parts[2])
            s = parts[-3] + "." + parts[-2] + parts[-1]
            c2_raw = math.log10(float(s))
        except Exception:
            continue

        cond1_vals.append(c1_raw)
        cond2_vals.append(c2_raw)

    if not cond1_vals:
        raise ValueError(f"No valid image files in {data_dir!r}")

    c1_arr = np.array(cond1_vals, dtype=np.float64)
    c2_arr = np.array(cond2_vals, dtype=np.float64)

    return (
        float(c1_arr.mean()), float(c1_arr.std(ddof=1)),
        float(c2_arr.mean()), float(c2_arr.std(ddof=1))
    )

# -----------------------------------------------------------------------------
# Normalize / unnormalize
# -----------------------------------------------------------------------------
def extract_conditions_from_filename(filename, mean1, std1, mean2, std2):
    """
    Reads the physical parameters encoded in the filename and
    applies Z-score normalization:

        z = (x - mean) / std

    Args:
        filename (str)
        mean1, std1, mean2, std2 (float): statistics computed from dataset

    Returns:
        tuple: (z1, z2) normalized conditions
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError("Filename must have at least 4 parts")

    cond1 = float(parts[2])
    s = parts[-3] + "." + parts[-2] + parts[-1]
    log_nH = math.log10(float(s))

    # Z-score normalization
    z1 = (cond1 - mean1) / (std1 + 1e-8)
    z2 = (log_nH - mean2) / (std2 + 1e-8)

    return z1, z2


def unnormalize_conditions(z1, z2, mean1, std1, mean2, std2):
    """
    Converts normalized (z1, z2) back to physical values.
    Useful for plotting or interpreting label predictions.
    """
    cond1 = z1 * std1 + mean1
    log_nH = z2 * std2 + mean2
    nH = 10 ** log_nH
    return cond1, nH