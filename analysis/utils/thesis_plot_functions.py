"""
Custom plotting utilities for thesis figures.

This module is intended to collect higher-level plotting helpers that
wrap around the core routines in `plotting.py`, `sbp.py`, and related
modules.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .plotting import _load_matrix
from .error_functions import PIXEL_SCALE_ARCSEC


def show_matrix(
    npz_path: str,
    x: float,
    nH: float,
    *,
    abs_tol: float = 1e-6,
    cmap: str = "gray",
    ax=None,
    show: bool = True,
):
    """
    Load and display the matrix corresponding to (x, nH) from an NPZ file.

    Parameters
    ----------
    npz_path:
        Path to an NPZ file created by ``save_npy_to_npz``.
    x, nH:
        Parameter pair identifying which matrix to extract.
    abs_tol:
        Absolute tolerance for matching ``x`` and ``nH`` inside the NPZ.
    cmap:
        Matplotlib colormap name for ``imshow``.
    ax:
        Optional Matplotlib Axes to draw on. If None, a new figure+axes
        are created.
    show:
        If True and a new figure is created, call ``plt.show()``.

    Returns
    -------
    arr : np.ndarray
        The 2D matrix that was displayed.
    """
    arr, actual_x, actual_nH = _load_matrix(npz_path, x, nH, abs_tol=abs_tol)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(arr, origin="lower", cmap=cmap)
    display_x = actual_x / 10000.0
    display_nH = str(actual_nH).replace("+", "")
    ax.set_title(f"x={display_x}, nH={display_nH}")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return arr


def show_matrix_anscombe(
    npz_path: str,
    x: float,
    nH: float,
    *,
    abs_tol: float = 1e-6,
    cmap: str = "viridis",
    ax=None,
    show: bool = True,
):
    """
    Load the matrix for (x, nH) from an NPZ file, apply the Anscombe
    transform, and display the result.

    Parameters
    ----------
    npz_path:
        Path to an NPZ file created by ``save_npy_to_npz``.
    x, nH:
        Parameter pair identifying which matrix to extract.
    abs_tol:
        Absolute tolerance for matching ``x`` and ``nH`` inside the NPZ.
    cmap:
        Matplotlib colormap name for ``imshow``.
    ax:
        Optional Matplotlib Axes to draw on. If None, a new figure+axes
        are created.
    show:
        If True and a new figure is created, call ``plt.show()``.

    Returns
    -------
    arr_ans : np.ndarray
        The Anscombe-transformed 2D matrix that was displayed.
    """
    arr, actual_x, actual_nH = _load_matrix(npz_path, x, nH, abs_tol=abs_tol)
    arr_ans = anscombe_transform(arr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(arr_ans, origin="lower", cmap=cmap)
    display_x = actual_x / 10000.0
    display_nH = str(actual_nH).replace("+", "")
    ax.set_title(f"Anscombe • x={display_x}, nH={display_nH}")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return arr_ans



def show_matrix_across_t(
    npz_path: str,
    x: float,
    nH: float,
    *,
    abs_tol: float = 1e-6,
    cmap: str = "gray",
    ax=None,
    show: bool = True,
):
    """
    Display the matrix for (x, nH) as it is progressively noised by a
    DDPM-style forward process with a linear beta schedule.
    """
    arr, actual_x, actual_nH = _load_matrix(npz_path, x, nH, abs_tol=abs_tol)

    # Display scaling: keep original image scale for all panels
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    # --- DDPM forward process params (Ho et al., linear betas) ---
    T = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    alphas = 1.0 - betas

    rng = np.random.default_rng(0)
    x_t = arr.astype(np.float64)

    snapshot_steps = [0, 200, 400, 600, 800, 1000]
    snapshots: dict[int, np.ndarray] = {0: x_t.copy()}

    for step in range(1, T + 1):
        alpha = alphas[step - 1]          # alpha_t = 1 - beta_t
        beta = 1.0 - alpha                # beta_t, same as betas[step - 1]
        noise = rng.normal(size=x_t.shape)
        x_t = np.sqrt(alpha) * x_t + np.sqrt(beta) * noise

        if step in snapshot_steps:
            snapshots[step] = x_t.copy()

    # Plot snapshots: t=0,200,400,600,800,1000
    if show:
        fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(18, 3))

        for ax_i, t_step in zip(axes, snapshot_steps):
            img = snapshots[t_step]

            # --- NEW: Per-image autoscaling ---
            vmin = float(np.nanmin(img))
            vmax = float(np.nanmax(img))

            ax_i.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax_i.set_title(f"t={t_step}")
            ax_i.set_axis_off()

        plt.tight_layout()
        plt.show()

    return arr

def plot_pixel_hist_from_npz(
    npz_path: str,
    x: float,
    nH: float,
    *,
    bins: int = 50,
    logy: bool = False,
    ax=None,
    show: bool = True,
):
    """
    Plot histograms of pixel values (raw and Anscombe-transformed)
    for a matrix stored in an NPZ.

    Parameters
    ----------
    npz_path:
        Path to an NPZ file created by ``save_npy_to_npz``.
    x, nH:
        Parameter pair identifying which matrix to extract.
    bins:
        Number of histogram bins.
    logy:
        If True, use a logarithmic y-axis.
    ax:
        Optional pair of Matplotlib Axes ``(ax_raw, ax_ans)`` to draw on.
        If None, a new figure with two subplots is created.
    show:
        If True and a new figure is created, call ``plt.show()``.

    Returns
    -------
    (hist_raw, bin_edges_raw, hist_ans, bin_edges_ans)
        Histogram counts and bin edges for the raw and Anscombe
        transforms, respectively.
    """
    arr, actual_x, actual_nH = _load_matrix(npz_path, x, nH)

    # Raw pixel values (ensure non-negative for Poisson interpretation)
    raw = np.asarray(arr, dtype=np.float64).ravel()
    raw = np.clip(raw, a_min=0.0, a_max=None)

    # Anscombe-transformed pixel values
    ans = 2.0 * np.sqrt(raw + 3.0 / 8.0)

    created_fig = False
    if ax is None:
        fig, (ax_raw, ax_ans) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        created_fig = True
    else:
        try:
            ax_raw, ax_ans = ax
        except (TypeError, ValueError):
            raise ValueError("ax must be a pair of axes (ax_raw, ax_ans) when provided.")
        fig = ax_raw.figure

    hist_raw, bin_edges_raw, _ = ax_raw.hist(
        raw,
        bins=bins,
        alpha=0.5,
        density=False,
    )
    hist_ans, bin_edges_ans, _ = ax_ans.hist(
        ans,
        bins=bins,
        alpha=0.5,
        density=False,
    )

    ax_raw.set_xlabel("Pixel value")
    ax_raw.set_ylabel("Count")
    ax_raw.set_title("Raw")

    ax_ans.set_xlabel("Pixel value")
    ax_ans.set_title("Anscombe")

    # Overlay Gaussian on Anscombe histogram for values > 0
    ans_pos = ans[ans > 0.0]
    if ans_pos.size > 0:
        mu = float(ans_pos.mean())
        sigma = float(ans_pos.std(ddof=0)) or 1.0
        # Use bin edges from the Anscombe histogram to set the domain
        x_min = max(bin_edges_ans[0], 0.0)
        x_max = bin_edges_ans[-1]
        xs = np.linspace(x_min, x_max, 512)
        # Gaussian PDF scaled to histogram counts
        bin_widths = np.diff(bin_edges_ans)
        bw = float(bin_widths[0]) if bin_widths.size > 0 else (x_max - x_min) / max(bins, 1)
        norm_factor = ans_pos.size * bw
        gauss = (
            norm_factor * 2.4
            * (1.0 / (1 * np.sqrt(2.0 * np.pi)))
            * np.exp(-0.5 * ((xs - 0) / sigma) ** 2)
        )
        ax_ans.plot(xs, gauss, color="C1", linewidth=2.0, label="Gaussian fit")
        ax_ans.legend()

    if logy:
        ax_raw.set_yscale("log")
        ax_ans.set_yscale("log")
    ax_raw.grid(True, alpha=0.3, linestyle="--")
    ax_ans.grid(True, alpha=0.3, linestyle="--")

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return hist_raw, bin_edges_raw, hist_ans, bin_edges_ans


import numpy as np
import matplotlib.pyplot as plt

def anscombe_transform(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return 2.0 * np.sqrt(arr + 3.0/8.0)



import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

def anscombe_transform(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    return 2.0 * np.sqrt(arr + 3.0 / 8.0)



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def anscombe_transform(x):
    return 2 * np.sqrt(x + 3.0 / 8.0)

def anscombe_inverse_biased(y):
    """Classical biased inverse of the Anscombe transform."""
    y = np.asarray(y, dtype=np.float64)
    return (y * 0.5) ** 2 - 3.0 / 8.0

def anscombe_inverse_unbiased(y):
    """
    Unbiased inverse Anscombe transform (Makitalo & Foi 2011 approximation).

    Parameters
    ----------
    y : array-like
        Anscombe-transformed values.

    Returns
    -------
    x_hat : array-like
        Unbiased estimate of the original Poisson mean.
    """
    y = np.asarray(y, dtype=np.float64)
    y = np.maximum(y, 1e-8)  # avoid division instability

    A = y  # A(y) in the formula

    return (
        0.25 * A**2
        + 0.25 * np.sqrt(3/2) * A**(-1)
        - (11/8) * A**(-2)
        + (5/8) * np.sqrt(3/2) * A**(-3)
        - 1/8
    )

def show_anscombe_pixel_stabilization(mean_value, N=5000):
    rng = np.random.default_rng(0)

    # Raw Poisson samples
    raw = rng.poisson(lam=mean_value, size=N)

    # Anscombe-transformed samples
    ans = anscombe_transform(raw)
    inv_biased = anscombe_inverse_biased(ans)
    inv_unbiased = anscombe_inverse_unbiased(ans)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

    # --------------------------------------------------
    # TOP LEFT: Raw Poisson (integer-centered bins)
    # --------------------------------------------------
    kmin, kmax = raw.min(), raw.max()
    bins_raw = np.arange(kmin - 0.5, kmax + 1.5, 1.0)

    counts_raw, bins_raw, _ = axes[0, 0].hist(
        raw,
        bins=bins_raw,
        color="steelblue",
        alpha=0.7,
        density=True,
        label="Poisson samples",
    )

    mu_raw = raw.mean()
    var_raw = raw.var()

    axes[0, 0].set_title(
        f"Raw Poisson samples\n"
        fr"$\mu={mu_raw:.2f},\ \sigma^2={var_raw:.2f}$"
    )
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Density")

    # --------------------------------------------------
    # TOP RIGHT: Anscombe (bins aligned with transformed integers)
    # --------------------------------------------------
    # Discrete support of raw values
    k_vals = np.arange(kmin, kmax + 1)
    y_vals = anscombe_transform(k_vals)  # possible Anscombe outputs

    # Build bin edges halfway between successive y_k
    edges = np.empty(len(y_vals) + 1, dtype=float)
    # interior edges = midpoints
    edges[1:-1] = 0.5 * (y_vals[:-1] + y_vals[1:])
    # extrapolate first and last edge
    edges[0] = y_vals[0] - 0.5 * (y_vals[1] - y_vals[0])
    edges[-1] = y_vals[-1] + 0.5 * (y_vals[-1] - y_vals[-2])

    counts_ans, bins_ans, _ = axes[0, 1].hist(
        ans,
        bins=edges,
        density=True,
        alpha=0.7,
        color="darkorange",
        label="Anscombe samples",
    )

    mu_ans = ans.mean()
    var_ans = ans.var()

    axes[0, 1].set_title(
        "Anscombe-transformed samples\n"
        fr"$\mu={mu_ans:.2f},\ \sigma^2={var_ans:.2f}$"
    )
    axes[0, 1].set_xlabel("Transformed value")
    axes[0, 1].set_ylabel("Density")

    # --------------------------------------------------
    # BOTTOM LEFT: Inverse Anscombe (biased)
    # --------------------------------------------------
    counts_inv_b, bins_inv_b, _ = axes[1, 0].hist(
        inv_biased,
        bins=bins_raw,
        color="forestgreen",
        alpha=0.7,
        density=True,
    )
    mu_inv_b = inv_biased.mean()
    var_inv_b = inv_biased.var()

    axes[1, 0].set_title(
        "Inverse Anscombe (biased)\n"
        fr"$\mu={mu_inv_b:.2f},\ \sigma^2={var_inv_b:.2f}$"
    )
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Density")

    # --------------------------------------------------
    # BOTTOM RIGHT: Unbiased inverse Anscombe
    # --------------------------------------------------
    counts_inv_u, bins_inv_u, _ = axes[1, 1].hist(
        inv_unbiased,
        bins=bins_raw,
        color="purple",
        alpha=0.7,
        density=True,
    )
    mu_inv_u = inv_unbiased.mean()
    var_inv_u = inv_unbiased.var()

    axes[1, 1].set_title(
        "Inverse Anscombe (unbiased approx)\n"
        fr"$\mu={mu_inv_u:.2f},\ \sigma^2={var_inv_u:.2f}$"
    )
    axes[1, 1].set_xlabel("Value")
    axes[1, 1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()
