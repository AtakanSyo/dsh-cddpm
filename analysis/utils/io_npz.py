from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def inspect_variance_from_npz(
    npz_path: str | Path,
    x: float,
    nH: float,
) -> np.ndarray:
    """Print, plot, and return the ``Vs`` matrix for a stored coordinate."""
    import matplotlib.pyplot as plt

    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path) as data:
        required_keys = {"xs", "nHs", "Vs"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(
                f"NPZ is missing required keys: {sorted(missing_keys)}"
            )

        xs = np.asarray(data["xs"], dtype=np.float64).reshape(-1)
        nHs = np.asarray(data["nHs"], dtype=np.float64).reshape(-1)
        variances = np.asarray(data["Vs"], dtype=np.float64)

        if not (len(xs) == len(nHs) == len(variances)):
            raise ValueError("'xs', 'nHs', and 'Vs' must have equal lengths.")

        matches = np.flatnonzero(
            np.isclose(xs, float(x), rtol=1e-9, atol=1e-12)
            & np.isclose(nHs, float(nH), rtol=1e-9, atol=1e-12)
        )

        if len(matches) == 0:
            raise ValueError(f"No variance matrix found for x={x}, nH={nH}.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple variance matrices found for x={x}, nH={nH}."
            )

        index = int(matches[0])
        variance_matrix = np.squeeze(variances[index]).copy()
        matched_x = float(xs[index])
        matched_nH = float(nHs[index])

    if variance_matrix.ndim != 2:
        raise ValueError(
            "The matched variance matrix must be 2-D after squeezing; "
            f"got shape {variance_matrix.shape}."
        )
    if not np.all(np.isfinite(variance_matrix)):
        raise ValueError("The matched variance matrix contains non-finite values.")
    if np.any(variance_matrix < 0.0):
        raise ValueError("The matched variance matrix contains negative values.")

    print(f"NPZ: {npz_path}")
    print(f"Index: {index}")
    print(f"Matched x: {matched_x}")
    print(f"Matched nH: {matched_nH:.6e}")
    print(f"Shape: {variance_matrix.shape}")
    print(f"Minimum: {variance_matrix.min():.6g}")
    print(f"Maximum: {variance_matrix.max():.6g}")
    print(f"Mean: {variance_matrix.mean():.6g}")
    print("Variance matrix:")
    with np.printoptions(precision=6, suppress=False):
        print(variance_matrix)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(variance_matrix, origin="lower", cmap="magma")
    ax.set_title(f"Propagated variance at x={matched_x}, nH={matched_nH:.2e}")
    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Pixel y")
    fig.colorbar(image, ax=ax, label="Propagated variance")
    fig.tight_layout()
    plt.show()

    return variance_matrix


def random_x_nH_from_npz(
    npz_path: str | Path,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Return a random (x, nH) pair from an NPZ archive containing `xs` and `nHs`.

    Args:
        npz_path: Path to the NPZ file with `xs` and `nHs` arrays.
        rng: Optional numpy random generator; defaults to `np.random.default_rng()`.

    Raises:
        KeyError: If `xs` or `nHs` arrays are missing.
        ValueError: If the arrays differ in length or are empty.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    if "xs" not in data or "nHs" not in data:
        raise KeyError("NPZ must contain 'xs' and 'nHs' arrays.")

    xs = np.asarray(data["xs"], dtype=np.float64).ravel()
    nHs = np.asarray(data["nHs"], dtype=np.float64).ravel()
    if xs.shape != nHs.shape:
        raise ValueError("'xs' and 'nHs' arrays must have the same length.")
    if xs.size == 0:
        raise ValueError("NPZ archive contains no entries.")

    rng = rng or np.random.default_rng()
    idx = int(rng.integers(xs.size))
    return float(xs[idx]), float(nHs[idx])
