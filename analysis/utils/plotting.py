import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
import math
from pathlib import Path
from typing import Optional
from matplotlib.lines import Line2D
from .error_functions import radial_reduced_chi2_area2_bin_filtered

def _load_matrix(npz_path: str, x: float, nH: float, *, abs_tol: float = 1e-6):
    data = np.load(npz_path)
    try:
        xs = data["xs"]
        nHs = data["nHs"]
        mats = data["mats"]

        for i, (xi, nHi) in enumerate(zip(xs, nHs)):
            if math.isclose(float(x), float(xi), abs_tol=abs_tol) and \
               math.isclose(float(nH), float(nHi), abs_tol=abs_tol):
                arr = np.squeeze(mats[i]).astype(np.float64)
                return arr, float(xi), float(nHi)
    finally:
        data.close()

    raise ValueError(f"Matrix with x={x}, nH={nH} not found in {npz_path}")






def plot_mean_error_from_heatmap(
    error_grid: np.ndarray,
    xs: np.ndarray,
    nHs: np.ndarray,
    *,
    over: str = "x",
    second_error_grid: np.ndarray | None = None,
    ax=None,
    show: bool = True,
    color: str = "tab:blue",
    marker: str = "o",
    second_color: str = "tab:orange",
    second_marker: str = "s",
    add_legend: bool = True,
    label: str | None = None,
    second_label: str | None = None,
    figsize=(6, 4),
    title: str | None = None,
):
    """
    Plot mean error collapsed over one axis of an error heatmap returned by
    error_heatmap_between_npz. Optionally overlay a second heatmap on the
    same axes (xs and nHs are assumed identical).

    Args:
        error_grid: 2D array shaped [len(nHs), len(xs)].
        second_error_grid: Optional second 2D array of the same shape to
            overlay on the same plot.
        xs: 1D array of x values (columns of error_grid).
        nHs: 1D array of nH values (rows of error_grid).
        over: "x" to average over nH (mean error vs x),
              "nH" to average over x (mean error vs nH).
    Returns:
        If second_error_grid is None, returns a 1D array of mean errors
        corresponding to the chosen axis. Otherwise returns a tuple
        (values1, values2).
    """
    E = np.asarray(error_grid, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    nHs = np.asarray(nHs, dtype=np.float64)

    if E.shape != (len(nHs), len(xs)):
        raise ValueError(f"error_grid shape {E.shape} does not match nHs ({len(nHs)}) and xs ({len(xs)})")
    if second_error_grid is not None:
        E2 = np.asarray(second_error_grid, dtype=np.float64)
        if E2.shape != E.shape:
            raise ValueError(
                f"second_error_grid shape {E2.shape} does not match first error_grid shape {E.shape}"
            )
    if over not in {"x", "nH"}:
        raise ValueError('over must be either "x" or "nH"')

    if over == "x":
        values = np.nanmean(E, axis=0)
        x_axis = np.round(xs / 10000.0, 2)  # plot x scaled down for readability
        xlabel = "x"
    else:
        values = np.nanmean(E, axis=1)
        x_axis = nHs
        # more math-like notation for nH
        xlabel = r"$n_{\mathrm{H}}$"

    values2 = None
    if second_error_grid is not None:
        if over == "x":
            values2 = np.nanmean(E2, axis=0)
        else:
            values2 = np.nanmean(E2, axis=1)

    plot_title = title if title is not None else f"Mean Error vs {xlabel}"

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x_axis, values, marker=marker, color=color)
    if values2 is not None:
        ax.plot(x_axis, values2, marker=second_marker, color=second_color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Mean radial $\chi^2_{\mathrm{red}}$ error")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if add_legend:
        legend_elements = []
        # Primary line
        primary_label = label if label is not None else "Heatmap 1"
        legend_elements.append(
            Line2D([0], [0], color=color, marker=marker, label=primary_label)
        )
        # Optional secondary line
        if second_error_grid is not None:
            secondary_label = second_label if second_label is not None else "Heatmap 2"
            legend_elements.append(
                Line2D([0], [0], color=second_color, marker=second_marker, label=secondary_label)
            )
        ax.legend(handles=legend_elements)

    if over == "x":
        # Use a subset of ticks to avoid overcrowding
        if len(x_axis) > 0:
            max_ticks = 5
            n_ticks = min(max_ticks, len(x_axis))
            tick_indices = np.linspace(0, len(x_axis) - 1, num=n_ticks, dtype=int)
            tick_positions = x_axis[tick_indices]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{v:.2f}" for v in tick_positions])
            ax.xaxis.set_minor_locator(NullLocator())
    else:
        if np.any(x_axis <= 0):
            raise ValueError("nH values must be positive for log scale.")
        ax.set_xscale("log")
        # Use a subset of ticks to avoid overcrowding
        if len(x_axis) > 0:
            max_ticks = 5
            n_ticks = min(max_ticks, len(x_axis))
            tick_indices = np.linspace(0, len(x_axis) - 1, num=n_ticks, dtype=int)
            tick_positions = x_axis[tick_indices]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{v:.2e}".replace("+", "") for v in tick_positions])
            ax.xaxis.set_minor_locator(NullLocator())

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return values


def plot_radial_bin_counts_from_heatmap(
    counts_grid: np.ndarray,
    xs: np.ndarray,
    nHs: np.ndarray,
    *,
    over: str = "x",
    ax=None,
    show: bool = True,
    color: str = "tab:blue",
    marker: str = "o",
    figsize=(6, 4),
    title: str | None = None,
):
    """
    Plot radial bin counts collapsed over one axis of a heatmap produced by
    radial_bin_threshold_heatmap.

    Args:
        counts_grid: 2D array shaped [len(nHs), len(xs)] with bin counts.
        xs: 1D array of x values (columns of counts_grid).
        nHs: 1D array of nH values (rows of counts_grid).
        over: "x" to average over nH (bin count vs x),
              "nH" to average over x (bin count vs nH).
    Returns:
        1D array of radial bin counts corresponding to the chosen axis.
    """
    C = np.asarray(counts_grid, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    nHs = np.asarray(nHs, dtype=np.float64)

    if C.shape != (len(nHs), len(xs)):
        raise ValueError(f"counts_grid shape {C.shape} does not match nHs ({len(nHs)}) and xs ({len(xs)})")
    if over not in {"x", "nH"}:
        raise ValueError('over must be either "x" or "nH"')

    if over == "x":
        values = np.nanmean(C, axis=0)
        x_axis = np.round(xs / 10000.0, 2)
        xlabel = "x"
    else:
        values = np.nanmean(C, axis=1)
        x_axis = nHs
        xlabel = r"$n_{\mathrm{H}}$"

    plot_title = title if title is not None else f"Radial bin count vs {xlabel}"

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x_axis, values, marker=marker, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Radial bin count")
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", alpha=0.4)

    if over == "x":
        if len(x_axis) > 0:
            max_ticks = 5
            n_ticks = min(max_ticks, len(x_axis))
            tick_indices = np.linspace(0, len(x_axis) - 1, num=n_ticks, dtype=int)
            tick_positions = x_axis[tick_indices]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{v:.2f}" for v in tick_positions])
            ax.xaxis.set_minor_locator(NullLocator())
    else:
        if np.any(x_axis <= 0):
            raise ValueError("nH values must be positive for log scale.")
        ax.set_xscale("log")
        if len(x_axis) > 0:
            max_ticks = 5
            n_ticks = min(max_ticks, len(x_axis))
            tick_indices = np.linspace(0, len(x_axis) - 1, num=n_ticks, dtype=int)
            tick_positions = x_axis[tick_indices]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{v:.2e}".replace("+", "") for v in tick_positions])
            ax.xaxis.set_minor_locator(NullLocator())

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return values


def normalize_and_log_scale(
    image,
    a: float = 1000.0,
    *,
    data_min: float | None = None,
    data_max: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Normalize image to [0,1], apply log(ax+1)/log(a), return scaled + min/max."""
    image = np.asarray(image, dtype=np.float64)
    img_min = float(np.min(image)) if data_min is None else float(data_min)
    img_max = float(np.max(image)) if data_max is None else float(data_max)

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image)

    log_scaled = np.log(a * normalized + 1.0) / np.log(a)
    return log_scaled, img_min, img_max


def plot_npz_matrices_by_params(
    npz_path: str | Path,
    x_values,
    nH_values,
    interpolant_x: float,
    interpolant_nH: float,
    skip_all_zero_pixels: bool = False,
):
    """Interpolate four NPZ matrices, print pixel calculations, and plot.

    When ``skip_all_zero_pixels`` is true, calculations are not printed for
    pixels whose values are zero in all four corner matrices.
    """
    x_values = list(x_values)
    nH_values = list(nH_values)

    if len(x_values) != len(nH_values):
        raise ValueError("x_values and nH_values must have equal lengths.")
    if len(x_values) != 4:
        raise ValueError("Provide exactly four (x, nH) corner coordinates.")
    if not np.isfinite(interpolant_x):
        raise ValueError("interpolant_x must be finite.")
    if not np.isfinite(interpolant_nH) or interpolant_nH <= 0.0:
        raise ValueError("interpolant_nH must be positive and finite.")

    matrix_by_coordinate = {}
    for x, nH in zip(x_values, nH_values):
        matrix, matched_x, matched_nH = _load_matrix(
            str(npz_path),
            float(x),
            float(nH),
        )
        if matrix.ndim != 2:
            raise ValueError(
                f"Matrix at x={matched_x}, nH={matched_nH} must be 2-D; "
                f"got shape {matrix.shape}."
            )
        key = (matched_x, matched_nH)
        if key in matrix_by_coordinate:
            raise ValueError(f"Duplicate corner coordinate: {key}.")
        matrix_by_coordinate[key] = matrix

    unique_x = np.array(
        sorted({coordinate[0] for coordinate in matrix_by_coordinate}),
        dtype=np.float64,
    )
    unique_nH = np.array(
        sorted({coordinate[1] for coordinate in matrix_by_coordinate}),
        dtype=np.float64,
    )
    if len(unique_x) != 2 or len(unique_nH) != 2:
        raise ValueError(
            "The four coordinates must contain exactly two distinct x values "
            "and two distinct nH values."
        )

    x0, x1 = map(float, unique_x)
    nH0, nH1 = map(float, unique_nH)
    corner_coordinates = [
        (x0, nH0),
        (x1, nH0),
        (x0, nH1),
        (x1, nH1),
    ]
    missing_corners = [
        coordinate
        for coordinate in corner_coordinates
        if coordinate not in matrix_by_coordinate
    ]
    if missing_corners:
        raise ValueError(
            f"The supplied coordinates do not form a complete cell; "
            f"missing corners: {missing_corners}."
        )

    matrices = [matrix_by_coordinate[coordinate] for coordinate in corner_coordinates]
    matrix_shape = matrices[0].shape
    for index, matrix in enumerate(matrices[1:], start=1):
        if matrix.shape != matrix_shape:
            raise ValueError(
                "All selected matrices must have the same shape; "
                f"matrix 0 has shape {matrix_shape}, while matrix {index} "
                f"has shape {matrix.shape}."
            )

    u = (float(interpolant_x) - x0) / (x1 - x0)
    log_nH0 = np.log10(nH0)
    log_nH1 = np.log10(nH1)
    v = (
        np.log10(float(interpolant_nH)) - log_nH0
    ) / (log_nH1 - log_nH0)

    weights = np.array(
        [
            (1.0 - u) * (1.0 - v),
            u * (1.0 - v),
            (1.0 - u) * v,
            u * v,
        ],
        dtype=np.float64,
    )
    interpolant_raw = sum(
        weight * matrix
        for weight, matrix in zip(weights, matrices)
    )
    interpolant = np.clip(interpolant_raw, 0.0, None)

    print("Bilinear interpolation setup:")
    print(f"  x bounds: {x0} to {x1}")
    print(f"  nH bounds: {nH0:.6e} to {nH1:.6e}")
    print(f"  target: x={float(interpolant_x)}, nH={float(interpolant_nH):.6e}")
    print(f"  u={u:.12g}")
    print(f"  v={v:.12g}  (linear in log10(nH))")
    for name, coordinate, weight in zip(
        ("w00", "w10", "w01", "w11"),
        corner_coordinates,
        weights,
    ):
        print(
            f"  {name}={weight:.12g} for "
            f"x={coordinate[0]}, nH={coordinate[1]:.6e}"
        )

    print("\nPer-pixel intensity calculations:")
    for row in range(matrix_shape[0]):
        for column in range(matrix_shape[1]):
            pixel_values = [matrix[row, column] for matrix in matrices]
            if skip_all_zero_pixels and all(value == 0.0 for value in pixel_values):
                continue
            contributions = [
                weight * value
                for weight, value in zip(weights, pixel_values)
            ]
            expression = " + ".join(
                f"({weight:.8g} * {value:.8g} = {contribution:.8g})"
                for weight, value, contribution in zip(
                    weights,
                    pixel_values,
                    contributions,
                )
            )
            raw_value = float(interpolant_raw[row, column])
            final_value = float(interpolant[row, column])
            clipping_note = (
                f" -> clipped to {final_value:.8g}"
                if raw_value < 0.0
                else ""
            )
            print(
                f"  I_out[{row}, {column}] = {expression} "
                f"= {raw_value:.8g}{clipping_note}"
            )

    displayed_matrices = [*matrices, interpolant]
    shared_min = float(min(np.nanmin(matrix) for matrix in displayed_matrices))
    shared_max = float(max(np.nanmax(matrix) for matrix in displayed_matrices))
    if not np.isfinite(shared_min) or not np.isfinite(shared_max):
        raise ValueError("Selected matrices contain no finite shared color range.")

    figure_width = 4.5 * len(displayed_matrices)
    fig, axes = plt.subplots(
        1,
        len(displayed_matrices),
        figsize=(figure_width, 4.5),
        squeeze=False,
    )
    scale_max = float(np.log(1001.0) / np.log(1000.0))

    display_coordinates = [
        *corner_coordinates,
        (float(interpolant_x), float(interpolant_nH)),
    ]
    for index, (ax, matrix, (matched_x, matched_nH)) in enumerate(zip(
        axes[0],
        displayed_matrices,
        display_coordinates,
    )):
        scaled, _, _ = normalize_and_log_scale(
            matrix,
            1000.0,
            data_min=shared_min,
            data_max=shared_max,
        )
        ax.imshow(
            scaled,
            cmap="magma",
            origin="lower",
            vmin=0.0,
            vmax=scale_max,
        )
        title_prefix = "Interpolant\n" if index == 4 else ""
        ax.set_title(
            f"{title_prefix}x={matched_x:g}\nnH={matched_nH:.2e}"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()
    return fig, axes[0], interpolant


def _log_scaled_to_original(values, img_min: float, img_max: float, a: float) -> np.ndarray:
    """Invert normalize_and_log_scale for tick labeling."""
    values = np.asarray(values, dtype=np.float64)
    normalized = (np.power(a, values) - 1.0) / a
    original = normalized * (img_max - img_min) + img_min
    return original

from matplotlib.colors import ListedColormap

def plot_radial_bins(
    n_annuli: int = 24,
    shape: tuple[int, int] = (48, 48),
    ax=None,
    show: bool = True,
    font_size: int = 12,
):
    """
    Visualize radial bin assignment with alternating black/white bin colors.
    """

    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)

    # Bin edges and assignment
    edges = np.linspace(0.0, rr.max(), n_annuli + 1)
    bins = np.digitize(rr.ravel(), edges) - 1
    bins = np.clip(bins, 0, n_annuli - 1).reshape(H, W)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    # --- BLACK / WHITE alternating colormap ---
    bw_colors = []
    for i in range(n_annuli):
        if i % 2 == 0:
            bw_colors.append((1, 1, 1, 1))   # white
        else:
            bw_colors.append((0, 0, 0, 1))   # black

    bw_cmap = ListedColormap(bw_colors)

    # show bins
    im = ax.imshow(bins, origin="lower", cmap=bw_cmap, vmin=-0.5, vmax=n_annuli - 0.5)
    ax.tick_params(axis="both", labelsize=font_size)

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return bins, edges


def plot_matrix_from_npz_log(
    npz_path,
    x,
    nH,
    abs_tol=1e-6,
    cmap="magma",
    log_base: float = 1000.0,
    nbins: int = 12,
    show_radial_grid: bool = False,
    ax=None,
    show: bool = True,
    add_colorbar: bool = True,
    show_axes: bool = True,
    *,
    matrix=None,
    data_min: float | None = None,
    data_max: float | None = None,
    actual_x: float | None = None,
    actual_nH: float | None = None,
):
    """Plot the matrix with additional normalize+log scaling for visibility."""
    if matrix is None:
        arr, actual_x_loaded, actual_nH_loaded = _load_matrix(npz_path, x, nH, abs_tol=abs_tol)
        actual_x = actual_x_loaded
        actual_nH = actual_nH_loaded
    else:
        arr = np.asarray(matrix, dtype=np.float64)
        if actual_x is None:
            actual_x = x
        if actual_nH is None:
            actual_nH = nH

    scaled, img_min, img_max = normalize_and_log_scale(
        arr,
        a=log_base,
        data_min=data_min,
        data_max=data_max,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(scaled, cmap=cmap, origin="lower")
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    if show_radial_grid:
        H, W = scaled.shape
        yy, xx = np.indices((H, W))
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = rr.max()
        radii = np.linspace(0.0, r_max, nbins + 1)[1:]
        theta = np.linspace(0, 2 * np.pi, 512)
        bin_centers = 0.5 * (np.linspace(0.0, r_max, nbins + 1)[:-1] + radii)
        for idx, (r, center_r) in enumerate(zip(radii, bin_centers), start=1):
            x_circle = np.clip(cx + r * np.cos(theta), 0, W - 1)
            y_circle = np.clip(cy + r * np.sin(theta), 0, H - 1)
            ax.plot(x_circle, y_circle, color="white", alpha=0.3, linewidth=0.8)
            label_x = np.clip(cx + center_r, 0, W - 1)
            label_y = np.clip(cy, 0, H - 1)
            ax.text(label_x, label_y, str(idx), color="white", fontsize=8, ha="center", va="center")

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        if img_max > img_min:
            ticks = np.linspace(0.0, 1.0, num=6)
            cbar.set_ticks(ticks)
            orig_vals = _log_scaled_to_original(ticks, img_min, img_max, log_base)
            cbar.set_ticklabels([f"{val:0.3g}" for val in orig_vals])
            cbar.set_label("Photon counts (log display)")
        else:
            cbar.set_label("Photon counts (constant)")

    ax.set_title(f"Photon Matrix (log-scale) x={actual_x}, nH={actual_nH}")

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return arr, scaled












def collect_topk_matches_bray_curtis(
    model_npz: str,
    data_npz: str,
    *,
    decimals: int = 6,
    top_k: int = 5,
    show: bool = True,
    err_kwargs=None,
    eps: float = 1.0,
):
    """
    Same interface as collect_topk_matches_bin_filtered but ranks pairs using the
    Bray-Curtis dissimilarity computed via bc_err (radial sum profiles).
    The eps argument stabilizes the denominator (default 1.0).
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    m = np.load(model_npz)
    d = np.load(data_npz)

    xs_m = np.round(m["xs"].astype(float), decimals)
    nHs_m = np.round(m["nHs"].astype(float), decimals)
    mats_m = m["mats"]

    xs_d = np.round(d["xs"].astype(float), decimals)
    nHs_d = np.round(d["nHs"].astype(float), decimals)
    mats_d = d["mats"]

    if mats_m.ndim != 3 or mats_d.ndim != 3:
        raise ValueError("Expected mats_* to be (N,H,W) arrays.")
    Hm, Wm = mats_m.shape[-2:]
    Hd, Wd = mats_d.shape[-2:]
    if (Hm, Wm) != (Hd, Wd):
        raise ValueError(f"Model and data matrix sizes differ: model {(Hm, Wm)} vs data {(Hd, Wd)}")

    metric_kwargs = dict(err_kwargs or {})
    metric_kwargs.setdefault("eps", eps)

    errs = np.empty((len(xs_d), len(xs_m)), dtype=np.float64)

    for j in range(len(xs_m)):
        model_mat = np.squeeze(mats_m[j]).astype(np.float64)
        for i in range(len(xs_d)):
            data_mat = np.squeeze(mats_d[i]).astype(np.float64)
            if data_mat.shape != model_mat.shape:
                raise ValueError(f"Shape mismatch at data idx {i}, model idx {j}")
            errs[i, j] = bc_err(
                data_mat,
                model_mat,
                **metric_kwargs,
            )

    key_to_idx_data = {(float(x), float(nH)): idx for idx, (x, nH) in enumerate(zip(xs_d, nHs_d))}
    true_ranks = np.full(len(xs_m), np.nan)
    order = np.argsort(errs, axis=0)

    result = []
    for j in range(len(xs_m)):
        top_data_idx = order[: min(top_k, order.shape[0]), j]
        matches = [
            {
                "x": float(xs_d[i]),
                "nH": float(nHs_d[i]),
                "error": float(errs[i, j]),
            }
            for i in top_data_idx
        ]

        key = (float(xs_m[j]), float(nHs_m[j]))
        if key in key_to_idx_data:
            true_i = key_to_idx_data[key]
            col_order = order[:, j]
            pos = np.where(col_order == true_i)[0]
            if pos.size:
                rank_pos = int(pos[0])
                true_ranks[j] = rank_pos + 1
                if rank_pos < len(matches):
                    matches[rank_pos]["is_correct_match"] = True

        result.append(
            {
                "model_x": float(xs_m[j]),
                "model_nH": float(nHs_m[j]),
                "matches": matches,
                "true_match_rank": None if not np.isfinite(true_ranks[j]) else int(true_ranks[j]),
            }
        )

    finite = np.isfinite(true_ranks)
    plot_data = None
    if finite.any():
        ranks = true_ranks[finite]
        max_rank = min(len(xs_d), top_k)
        ks = np.arange(1, max_rank + 1)
        cmc = np.array([(ranks <= k).mean() for k in ks])
        plot_data = {
            "ks": ks,
            "cmc": cmc,
            "max_rank": max_rank,
            "num_models": int(finite.sum()),
        }

        if show:
            print("\nIdentification rate for first 10 ranks (Bray-Curtis):")
            for idx in range(min(10, len(cmc))):
                print(f"  Rank ≤ {idx + 1}: {cmc[idx]:.4f}")

            plt.figure(figsize=(6, 4))
            plt.plot(ks, cmc, marker="o")
            plt.xlabel("Rank k")
            plt.ylabel("Identification rate")
            plt.title("CMC Curve • Bray-Curtis")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.xlim(1, max_rank)
            plt.tight_layout()
            plt.show()
    else:
        if show:
            print("No overlapping (x, nH) pairs to compute CMC curve.")

    return result, plot_data




def plot_bin_errors_for_pair(
    model_npz: str,
    data_npz: str,
    x: float,
    nH: float,
    n_annuli: int = 24,
    center: str = "geom",
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 0,
    abs_tol: float | None = None,
    decimals: int = 6,
    mask_empty_data_bins: bool = False,
    verbose: bool = True,
    figsize=(7, 4),
    show: bool = True,
    eps: float = 1e-12,
):
    """
    Plot the contribution to the total chi-square coming from each radial bin
    for a single (x, nH) matrix pair drawn from two NPZ archives.

    Set mask_empty_data_bins=True to drop bins whose data-side annular sum is
    zero, mirroring the behavior of radial_reduced_chi2_area2 when bins are
    masked out.

    Set verbose=False to suppress intermediate computation prints.
    Set show=False to skip rendering the bar plot.
    """
    if abs_tol is None:
        abs_tol = 10 ** (-decimals)

    model_mat, _, _ = _load_matrix(model_npz, x, nH, abs_tol=abs_tol)
    data_mat, _, _ = _load_matrix(data_npz, x, nH, abs_tol=abs_tol)

    # pretty_print_matrix(model_mat, fmt="{:8.0f}")
    # pretty_print_matrix(data_mat, fmt="{:8.0f}")

    if data_mat.shape != model_mat.shape:
        raise ValueError(f"Shape mismatch: data={data_mat.shape}, model={model_mat.shape}")

    # pretty_print_matrix(
    #     model_mat,
    #     title=f"Model matrix (x={model_x}, nH={model_nH})",
    # )
    # pretty_print_matrix(
    #     data_mat,
    #     title=f"Data matrix (x={data_x}, nH={data_nH})",
    # )

    # plot_matrix_from_npz(model_npz, x, nH, abs_tol=abs_tol, cmap="magma")
    # plot_matrix_from_npz(data_npz, x, nH, abs_tol=abs_tol, cmap="viridis")

    H, W = data_mat.shape
    yy, xx = np.indices((H, W))

    if center == "peak":
        cy, cx = np.unravel_index(np.nanargmax(data_mat), data_mat.shape)
    else:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = rr.max()
    edges = np.linspace(0.0, r_max, n_annuli + 1)
    bins = np.digitize(rr.ravel(), edges) - 1
    bins = np.clip(bins, 0, n_annuli - 1)

    S_data = np.bincount(bins, weights=np.nan_to_num(data_mat.ravel(), nan=0.0), minlength=n_annuli).astype(np.float64)
    S_model = np.bincount(bins, weights=np.nan_to_num(model_mat.ravel(), nan=0.0), minlength=n_annuli).astype(np.float64)

    areas_px2 = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    areas_arcsec2 = areas_px2 * (pixel_scale_arcsec ** 2)

    mu_ref = S_data / np.maximum(areas_arcsec2, eps)
    mu_model = S_model / np.maximum(areas_arcsec2, eps)

    weights = (areas_arcsec2 ** 2) / np.maximum(S_data, eps)

    valid_bins = S_data > bin_filter

    if not np.any(valid_bins):
        raise ValueError("All bins were masked out; no valid data bins remain.")

    num = np.sum((weights * mu_model * mu_ref)[valid_bins])
    den = np.sum((weights * mu_model * mu_model)[valid_bins])
    N_best = num / den

    residuals = mu_ref - N_best * mu_model
    bin_errors = weights * residuals * residuals
    bin_errors[~valid_bins] = 0.0  # masked bins contribute zero

    if verbose:
        print("\nPer-bin photon counts and contributions:")
        print("bin | Sa(data) | Sb(model) | mu_ref | mu_model | weight | residual | contribution")
        for idx in range(n_annuli):
            masked_note = "  (masked)" if not valid_bins[idx] else ""
            print(
                f"{idx+1} | "
                f"{S_data[idx]:10.4g} | "
                f"{S_model[idx]:10.4g} | "
                f"{mu_ref[idx]:8.4g} | "
                f"{mu_model[idx]:8.4g} | "
                f"{weights[idx]:8.4g} | "
                f"{residuals[idx]:8.4g} | "
                f"{bin_errors[idx]:10.4g}"
                f"{masked_note}"
            )
            print(
                f"    contribution = weight * (mu_ref - N_best * mu_model)^2 = "
                f"{weights[idx]:.4g} * ({mu_ref[idx]:.4g} - {N_best:.4g} * {mu_model[idx]:.4g})^2"
            )
    chi2 = np.sum(bin_errors[valid_bins])
    dof = max(int(valid_bins.sum()) - 1, 1)
    chi2_red = chi2 / dof

    if mask_empty_data_bins and verbose:
        print(f"\nBins used: {int(valid_bins.sum())} / {n_annuli}")

    radii_arcsec = 0.5 * (edges[:-1] + edges[1:]) * pixel_scale_arcsec

    plot_errors = bin_errors.copy()
    if mask_empty_data_bins:
        plot_errors[~valid_bins] = 0.0

    if show:
        plt.figure(figsize=figsize)
        indices = np.arange(n_annuli)
        plt.bar(indices, plot_errors, color="tab:orange", alpha=0.8)
        plt.xticks(indices, indices + 1)
        plt.xlabel("Bin index")
        plt.ylabel("Error contribution (w * residual^2)")
        plt.title(f"Per-bin Error • x={x}, nH={nH}\nχ²_red={chi2_red:.3f}, N={N_best:.3f}")
        plt.grid(True, linestyle="--", alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    return {
        "bin_errors": bin_errors,
        "radii_arcsec": radii_arcsec,
        "chi2": chi2,
        "chi2_red": chi2_red,
        "scale": N_best,
        "masked_bins": int((~valid_bins).sum()) if mask_empty_data_bins else 0,
    }

def plot_mean_bin_errors_across_npz(
    model_npz: str,
    data_npz: str,
    *,
    n_annuli: int = 24,
    center: str = "geom",
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 0,
    x_list: Optional[list[float]] = None,
    nH_list: Optional[list[float]] = None,
    abs_tol: float | None = None,
    decimals: int = 6,
    mask_empty_data_bins: bool = False,
    figsize=(7, 4),
    eps: float = 1e-12,
    show: bool = True,
):
    """
    Average per-bin reduced-chi2 contributions across every (x, nH) matrix pair
    shared by two NPZ archives and plot the mean profile.

    Uses plot_bin_errors_for_pair with verbose=False to reuse the per-bin logic
    while suppressing intermediate printing and sub-plots.

    If x_list and/or nH_list are provided, only entries whose x and/or nH
    match one of the provided values within abs_tol are included.
    """
    if abs_tol is None:
        abs_tol = 10 ** (-decimals)

    # Load shared coordinate grids to iterate all pairs.
    try:
        model_npz_data = np.load(model_npz)
        data_npz_data = np.load(data_npz)
        xs_model = np.asarray(model_npz_data["xs"], dtype=float)
        xs_data = np.asarray(data_npz_data["xs"], dtype=float)
        nHs_model = np.asarray(model_npz_data["nHs"], dtype=float)
        nHs_data = np.asarray(data_npz_data["nHs"], dtype=float)
    except KeyError as exc:
        raise ValueError("NPZ archives must contain 'xs' and 'nHs' arrays.") from exc
    finally:
        try:
            model_npz_data.close()
        except Exception:
            pass
        try:
            data_npz_data.close()
        except Exception:
            pass

    if xs_model.shape != xs_data.shape or nHs_model.shape != nHs_data.shape:
        raise ValueError("Model and data NPZ archives must align in xs/nHs lengths.")
    if xs_model.size == 0:
        raise ValueError("No matrices found in NPZ archives.")
    if not (np.allclose(xs_model, xs_data, atol=abs_tol) and np.allclose(nHs_model, nHs_data, atol=abs_tol)):
        raise ValueError("Model and data NPZ archives must have matching xs/nHs entries.")

    selected_xs = xs_model
    selected_nHs = nHs_model
    if x_list is not None:
        requested_xs = np.asarray(x_list, dtype=float).ravel()
        if requested_xs.size == 0:
            raise ValueError("x_list was provided but is empty.")
        keep_mask_x = np.array(
            [np.any(np.isclose(x_val, requested_xs, atol=abs_tol, rtol=0.0)) for x_val in selected_xs],
            dtype=bool,
        )
        selected_xs = selected_xs[keep_mask_x]
        selected_nHs = selected_nHs[keep_mask_x]
        if selected_xs.size == 0:
            unique_xs = np.unique(np.round(xs_model, decimals=decimals))
            raise ValueError(
                f"No matching x values found for x_list={requested_xs.tolist()} "
                f"(abs_tol={abs_tol}). Available x values (rounded): {unique_xs.tolist()}"
            )

    if nH_list is not None:
        requested_nHs = np.asarray(nH_list, dtype=float).ravel()
        if requested_nHs.size == 0:
            raise ValueError("nH_list was provided but is empty.")

        keep_mask = np.array(
            [np.any(np.isclose(nh_val, requested_nHs, atol=abs_tol, rtol=0.0)) for nh_val in selected_nHs],
            dtype=bool,
        )
        selected_xs = selected_xs[keep_mask]
        selected_nHs = selected_nHs[keep_mask]
        if selected_xs.size == 0:
            unique_nHs = np.unique(np.round(nHs_model, decimals=decimals))
            raise ValueError(
                f"No matching nH values found for nH_list={requested_nHs.tolist()} "
                f"(abs_tol={abs_tol}). Available nH values (rounded): {unique_nHs.tolist()}"
            )

    per_bin_reduced = []
    radii_arcsec = None

    for x_val, nH_val in zip(selected_xs, selected_nHs):
        # Suppress per-call plots while reusing computation logic.
        with plt.ioff():
            res = plot_bin_errors_for_pair(
                model_npz,
                data_npz,
                float(x_val),
                float(nH_val),
                n_annuli=n_annuli,
                center=center,
                pixel_scale_arcsec=pixel_scale_arcsec,
                bin_filter=bin_filter,
                abs_tol=abs_tol,
                decimals=decimals,
                mask_empty_data_bins=mask_empty_data_bins,
                figsize=figsize,
                eps=eps,
                verbose=False,
                show=False,
            )

        if radii_arcsec is None:
            radii_arcsec = res["radii_arcsec"]

        # Convert per-bin chi2 contributions to reduced-chi2 contributions.
        dof = max(res["chi2"] / res["chi2_red"], 1e-12)
        per_bin_reduced.append(np.asarray(res["bin_errors"], dtype=float) / dof)

    stack = np.vstack(per_bin_reduced)
    mean_errors = np.mean(stack, axis=0)
    std_errors = np.std(stack, axis=0)
    x_vals_print = np.asarray(radii_arcsec, dtype=float) if radii_arcsec is not None else np.arange(n_annuli, dtype=float)

    print("\nMean reduced χ² contribution by radial bin:")
    for idx, (radius_arcsec, mean_val) in enumerate(zip(x_vals_print, mean_errors), start=1):
        print(f"  Bin {idx:>2d} (r={radius_arcsec:>9.3f} arcsec): {mean_val:.6e}")
    print(f"Total mean reduced χ² contribution (sum over bins): {np.sum(mean_errors):.6e}")

    if show:
        x_vals = x_vals_print
        bin_idx = np.arange(1, n_annuli + 1)
        if x_vals.size > 1:
            dx = float(np.median(np.diff(x_vals)))
        else:
            dx = float(pixel_scale_arcsec)

        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.bar(x_vals, mean_errors, width=0.8 * dx, color="tab:green", alpha=0.8, align="center")
        ax.set_xlabel(r"Radius $\left[\mathrm{arcsec}\right]$")
        plt.ylabel("Mean reduced χ² contribution per bin")
        plt.title(f"Mean Radial Reduced χ² per Bin")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.grid(True, linestyle="--", alpha=0.3, axis="both")

        # Secondary top axis: annulus indices, matching compare_npy_matrices style.
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks(x_vals)
        ax_top.set_xticklabels([str(i) for i in bin_idx])
        ax_top.set_xlabel("Radial bin index")

        plt.tight_layout()
        plt.show()

    return {
        "mean_errors": mean_errors,
        "std_errors": std_errors,
        "radii_arcsec": radii_arcsec,
        "count": stack.shape[0],
    }








from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_radial_profiles_overlay(
    npz_path_Model: str,
    npz_path_Data: str,
    x: float,
    nH: float,
    *,
    n_annuli: int = 24,
    center: str = "geom",
    abs_tol: float = 1e-6,
    figsize=(6, 4),
    ax=None,
    show: bool = True,
    bin_filter: float = 0.0,
    reflex_bin_variances=None,
    mu_ref_profile=None,
    mu_model_profile=None,
    N_best: float = 1.0,
    pixscale_arcsec: float = 43.2,
    font_size: int = 12,
    model_label: str = "Model C",
    plot_style: str = "line",  # <--- NEW ARGUMENT: "line" or "step"
):
    """
    Overlay radial surface brightness profiles.
    
    Parameters
    ----------
    plot_style : str
        "line" for standard connected points (default), 
        "step" for step-mid histogram style.
    """

    # --- 1. Load one matrix just to get geometry ---
    arr_Model, _, _ = _load_matrix(npz_path_Model, x, nH, abs_tol=abs_tol)
    H, W = np.squeeze(arr_Model).shape

    # --- 2. Geometric Setup ---
    if center == "peak":
        cy, cx = np.unravel_index(np.nanargmax(arr_Model), (H, W))
    else:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    yy, xx = np.indices((H, W))
    rr_pix = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max_pix = rr_pix.max()

    edges_pix = np.linspace(0.0, r_max_pix, n_annuli + 1)
    bin_centers_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
    bin_centers_arcsec = bin_centers_pix * pixscale_arcsec

    # --- 3. Validate Inputs ---
    mu_ref_vals = np.asarray(mu_ref_profile, dtype=np.float64)
    mu_model_vals = np.asarray(mu_model_profile, dtype=np.float64)
    
    if len(mu_ref_vals) != n_annuli or len(mu_model_vals) != n_annuli:
        raise ValueError(f"Profile length mismatch! Expected {n_annuli} bins.")

    # --- 4. Filtering and Scaling ---
    valid_mask = (mu_ref_vals > 0) & np.isfinite(mu_ref_vals)
    
    if reflex_bin_variances is not None:
        reflex_bin_variances = np.asarray(reflex_bin_variances)
        valid_mask &= (reflex_bin_variances > 0)
        reflex_errors = np.sqrt(reflex_bin_variances)
    else:
        reflex_errors = np.zeros_like(mu_ref_vals)

    scaled_model_profile = mu_model_vals * N_best
    
    x_plot = bin_centers_arcsec[valid_mask]
    y_ref = mu_ref_vals[valid_mask]
    y_model = scaled_model_profile[valid_mask]
    y_err = reflex_errors[valid_mask]

    # --- 5. Plotting ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # --- STYLE SWITCH ---
    if plot_style == "step":
        # Step Plot (Histogram style)
        ax.step(x_plot, y_ref, where='mid', color="tab:orange", label="RefleX (Data)", linewidth=2)
        ax.step(x_plot, y_model, where='mid', color="tab:blue", label=f"{model_label} (Scaled)", linewidth=2, linestyle='--')
    else:
        # Line Plot (Default)
        ax.plot(x_plot, y_ref, "-", color="tab:orange", label="RefleX", linewidth=2)
        ax.plot(x_plot, y_model, "-", color="tab:blue", label=f"{model_label}", linewidth=2)

    # Error bars (Common to both)
    ax.errorbar(
        x_plot, 
        y_ref, 
        yerr=y_err, 
        fmt="none", 
        ecolor="tab:orange", 
        elinewidth=1.5, 
        capsize=3,
        alpha=0.6
    )

    # Styling
    ax.set_xlabel(r"Radius [arcsec]", fontsize=font_size)
    ax.set_ylabel(r"Surface Brightness [photons arcsec$^{-2}$]", fontsize=font_size)
    ax.set_title(f"Radial Profile (N_best={N_best:.2f})", fontsize=font_size)
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", labelsize=font_size)
    ax.legend(fontsize=font_size)
    
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    if show and created_fig:
        fig.tight_layout()
        plt.show()

    return y_ref, y_model




def plot_matrices_with_radial_overlay_log(
    npz_path_Model: str,  # model
    npz_path_Data: str,  # data
    x: float,
    nH: float,
    *,
    abs_tol: float = 1e-6,
    cmap_a: str = "magma",
    cmap_b: str = "viridis",
    n_annuli: int = 24,
    center: str = "geom",
    figsize=(10, 10),          # taller: extra row for the count bar plot
    log_base: float = 1000.0,
    nbins: int = 24,
    show_radial_grid: bool = False,
    bin_filter: float = 0.0,
    font_size: int = 12,
    model_label: str = "Model C",
    pixscale_arcsec: float = 43.2,
):
    """
    Same as before, plus a bottom bar plot of total photon counts per radial bin
    for both Data (RefleX) and Model, using the same binning as the chi2 metric.
    Returns the radial_reduced_chi2_area2_bin_filtered error.
    """

    # --- Layout: 3 rows. matrices | radial profile | per-bin counts ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.0, 0.7])

    ax_model = fig.add_subplot(gs[0, 0])   # top-left: model matrix
    ax_data  = fig.add_subplot(gs[0, 1])   # top-right: data matrix
    ax_prof  = fig.add_subplot(gs[1, :])   # middle: radial profile overlay
    ax_cnt   = fig.add_subplot(gs[2, :])   # bottom: per-bin photon counts

    arr_Model, actual_x_a, actual_nH_a = _load_matrix(npz_path_Model, x, nH, abs_tol=abs_tol)
    arr_Data, actual_x_b, actual_nH_b = _load_matrix(npz_path_Data, x, nH, abs_tol=abs_tol)
    shared_min = float(np.nanmin([np.nanmin(arr_Model), np.nanmin(arr_Data)]))
    shared_max = float(np.nanmax([np.nanmax(arr_Model), np.nanmax(arr_Data)]))

    # --- per-bin photon counts, SAME binning as the chi2 metric ---
    def _bin_counts(img):
        img = np.nan_to_num(np.squeeze(np.asarray(img, dtype=np.float64)))
        H, W = img.shape
        cy, cx = (H - 1) / 2, (W - 1) / 2
        yy, xx = np.indices((H, W))
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        edges = np.linspace(0.0, rr.max(), n_annuli + 1)
        bins = np.clip(np.digitize(rr.ravel(), edges) - 1, 0, n_annuli - 1)
        S = np.bincount(bins, weights=img.ravel(), minlength=n_annuli)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        return S, r_centers

    S_data, r_centers = _bin_counts(arr_Data)
    S_model, _        = _bin_counts(arr_Model)
    r_arcsec = r_centers * pixscale_arcsec

    chi2_err = None
    reflex_bin_variances = None
    mu_ref_profile = None
    mu_model_profile = None
    try:
        (
            chi2_err,
            reflex_bin_variances,
            mu_ref_profile,
            mu_model_profile,
            N_best,
        ) = radial_reduced_chi2_area2_bin_filtered(
            data=arr_Data,
            model=arr_Model,
            n_annuli=n_annuli,
            bin_filter=bin_filter,
            return_bin_errors=True,
            return_profiles=True,
        )
        print(f"N best found to be {N_best}")
        print(f"radial_reduced_chi2_area2_bin_filtered (Model vs RefleX): {chi2_err:.4f}")
    except ValueError as exc:
        print(f"radial_reduced_chi2_area2_bin_filtered unavailable: {exc}")
        reflex_bin_variances = None
        mu_ref_profile = None
        mu_model_profile = None
        N_best = 1.0

    x_label = x / 10000.0
    nH_mantissa, nH_exp = f"{nH:.2e}".split("e")
    nH_label = f"{float(nH_mantissa):.2f}e{int(nH_exp)}"

    # --- Top row: matrices ---
    plot_matrix_from_npz_log(
        npz_path_Model, x, nH, abs_tol=abs_tol, cmap=cmap_a, log_base=log_base,
        nbins=nbins, show_radial_grid=show_radial_grid, ax=ax_model, show=False,
        add_colorbar=False, show_axes=False, matrix=arr_Model,
        data_min=shared_min, data_max=shared_max,
        actual_x=actual_x_a, actual_nH=actual_nH_a,
    )
    ax_model.set_title(model_label)

    plot_matrix_from_npz_log(
        npz_path_Data, x, nH, abs_tol=abs_tol, cmap=cmap_b, log_base=log_base,
        nbins=nbins, show_radial_grid=show_radial_grid, ax=ax_data, show=False,
        add_colorbar=False, show_axes=False, matrix=arr_Data,
        data_min=shared_min, data_max=shared_max,
        actual_x=actual_x_b, actual_nH=actual_nH_b,
    )
    ax_data.set_title("RefleX")

    # --- Middle row: radial profiles ---
    plot_radial_profiles_overlay(
        npz_path_Data, npz_path_Model, x, nH,
        n_annuli=n_annuli, center=center, abs_tol=abs_tol, ax=ax_prof, show=False,
        bin_filter=bin_filter, reflex_bin_variances=reflex_bin_variances,
        mu_ref_profile=mu_ref_profile, mu_model_profile=mu_model_profile,
        N_best=N_best, pixscale_arcsec=pixscale_arcsec,
        font_size=font_size, model_label=model_label,
    )
    if chi2_err is not None:
        ax_prof.set_title(
            f"Reduced $\\chi^2$ = {chi2_err:.4f}\n"
            f"x: {x_label:.4f}, nH: {nH_label}"
        )
    else:
        ax_prof.set_title("Reduced $\\chi^2$ unavailable")

    # --- Bottom row: per-bin photon counts ---
    width = (r_arcsec[1] - r_arcsec[0]) * 0.4 if len(r_arcsec) > 1 else 1.0
    ax_cnt.bar(r_arcsec - width / 2, S_data,  width=width, color="C0",
               alpha=0.8, label="RefleX")
    ax_cnt.bar(r_arcsec + width / 2, S_model, width=width, color="C1",
               alpha=0.8, label=model_label)
    if bin_filter > 0:
        ax_cnt.axhline(bin_filter, color="red", ls="--", lw=1.0,
                       label=f"bin_filter = {bin_filter:g}")
    ax_cnt.set_xlabel(r"Radius [arcsec]", fontsize=font_size)
    ax_cnt.set_ylabel("Counts per bin", fontsize=font_size)
    ax_cnt.set_yscale("log")
    ax_cnt.grid(True, alpha=0.2)
    ax_cnt.legend(fontsize=font_size - 2)

    fig.tight_layout()
    plt.show()

    return chi2_err




from typing import Optional
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# assume 45.0 is defined globally, e.g.:
# 45.0 = 43.2



from pathlib import Path
from typing import List, Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import optimize
from typing import Optional, Tuple


def _radius_map(H: int, W: int) -> np.ndarray:
    """Per-pixel radius from the tile center, normalized to [0, 1] at the
    nearest edge. Used only to color points by where they sit in the halo,
    so the exact center convention is not critical."""
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return r / min(cy, cx)


def _binned_ridge(mu: np.ndarray, var: np.ndarray, n_bins: int = 24):
    """Median sigma^2 in log-spaced mu bins. The ridge tracks the BULK of the
    pixels, so a handful of high-mu hot pixels (huge leverage in a raw fit)
    can't drag it. Returns (bin_centers, median_var, count_per_bin).

    Self-defending: drops non-finite and non-positive pixels first, so callers
    can pass raw red['mean']/red['var'] arrays (which contain zero-count pixels)
    without producing NaN edges from log10(0)."""
    mu = np.asarray(mu, float)
    var = np.asarray(var, float)
    finite = np.isfinite(mu) & np.isfinite(var) & (mu > 0) & (var > 0)
    mu, var = mu[finite], var[finite]
    if mu.size == 0:
        return np.array([]), np.array([]), np.array([])
    lo, hi = mu.min(), mu.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    idx = np.digitize(mu, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers, med, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            centers.append(np.sqrt(edges[b] * edges[b + 1]))  # geometric center
            med.append(np.median(var[m]))
            cnt.append(int(m.sum()))
    return np.array(centers), np.array(med), np.array(cnt)


def _fit_fano(mu: np.ndarray, var: np.ndarray, mu_max: Optional[float] = None):
    """Fit the constant-Fano model sigma^2 = F * mu to the BINNED RIDGE.

    The plots show variance running parallel to the Poisson line (slope 1),
    offset by a constant multiplicative factor -- i.e. a constant Fano factor
    F = sigma^2 / mu, NOT a fractional floor (alpha*mu)^2 (which would bend up
    at slope 2). F is the quantity that maps directly onto the chi2_red floor:
    putting Poisson variance mu in the chi2 denominator while the true variance
    is F*mu inflates chi2_red by exactly F.

    mu_max : if given, only ridge bins with center < mu_max enter the fit. Use
    this to get the clean "halo floor" by excluding the bright core, where a
    separate fractional term (alpha*mu)^2 takes over.

    Fitting the ridge (binned medians) rather than raw points keeps high-mu
    outlier pixels from biasing F upward.
    Returns (F, ridge_mu, ridge_var, ridge_cnt)."""
    rmu, rvar, rcnt = _binned_ridge(mu, var)
    sel = np.ones_like(rmu, dtype=bool) if mu_max is None else (rmu < mu_max)
    # constant Fano: F is the robust median of var/mu along the (selected) ridge
    F = float(np.median(rvar[sel] / rmu[sel]))
    return F, rmu, rvar, rcnt


def _fit_fano_plus_quad(mu: np.ndarray, var: np.ndarray):
    """Optional richer model sigma^2 = F*mu + (alpha*mu)^2, fit to the ridge in
    log space. Lets you TEST whether a slope-2 term is even warranted: if the
    ridge is straight (constant Fano), alpha collapses toward 0 and F carries
    the fit. The crossover mu ~ 1/alpha^2 marks where the fractional term starts
    to dominate the constant-Fano term. Returns (F, alpha)."""
    rmu, rvar, _ = _binned_ridge(mu, var)

    def log_resid(p):
        F, log_a = p[0], p[1]
        a = np.exp(log_a)
        model = F * rmu + (a * rmu) ** 2
        return np.log(model) - np.log(rvar)

    res = optimize.least_squares(
        log_resid, x0=[np.median(rvar / rmu), np.log(0.05)],
        bounds=([0.1, np.log(1e-4)], [50.0, np.log(5.0)]), loss="soft_l1")
    return float(res.x[0]), float(np.exp(res.x[1]))


def plot_variance_vs_mean(
    mean_arr: np.ndarray,
    var_arr: np.ndarray,
    *,
    n_seeds: Optional[int] = None,
    mu_floor: float = 0.2,
    gridsize: int = 60,
    fit: bool = True,
    fit_quad: bool = False,
    fano_mu_max: Optional[float] = None,
    color_by_radius: bool = True,
    figsize: Tuple[float, float] = (8, 6.5),
    title: str = r"Per-pixel variance vs. mean",
    font_size: int = 12,
    mask_low_mean: Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Diagnostic companion to plot_quantity_grid: collapses the spatial layout
    and plots every pixel as a point in (mean, variance) space against the
    Poisson reference line var = mu.

    Excess over Poisson is the vertical gap from the dashed line. If the
    overdispersion is multiplicative (constant fractional dispersion from
    photon-trajectory sampling), points follow var = mu + (alpha*mu)^2 — flat
    at low mu, bending into a slope-2 power law above the crossover mu ~ 1/alpha^2.

    Parameters
    ----------
    mean_arr, var_arr : (N_cells, H, W) — MUST be in raw photon counts for the
        Poisson reference to be meaningful. If RefleX output is flux-normalized,
        var = mu is not your reference and alpha is not a pure sampling term.
    n_seeds : if given, draws the +/- 1 sigma band on the sample-variance
        estimate itself (fractional uncertainty sqrt(2/(N-1))), so speckle from
        a small seed count is not mistaken for structure.
    mask_low_mean : (mean_array, threshold) — same NaN-blanking convention as
        plot_quantity_grid; blanked pixels are dropped from the scatter.
    """
    if color_by_radius and gridsize:
        # hexbin can't also color by a third variable cleanly; if both are on,
        # radius coloring wins and we use a 2-bin scatter instead of hexbin.
        pass

    N, H, W = mean_arr.shape
    mu = mean_arr.astype(float).copy()
    var = var_arr.astype(float).copy()

    if mask_low_mean is not None:
        m_arr, thresh = mask_low_mean
        keep = m_arr >= thresh
        mu = np.where(keep, mu, np.nan)
        var = np.where(keep, var, np.nan)

    # radius proxy, broadcast across all cells
    rad = np.broadcast_to(_radius_map(H, W)[None, :, :], mu.shape)

    mu_f = mu.ravel()
    var_f = var.ravel()
    rad_f = rad.ravel()

    good = np.isfinite(mu_f) & np.isfinite(var_f) & (mu_f >= mu_floor) & (var_f > 0)
    mu_f, var_f, rad_f = mu_f[good], var_f[good], rad_f[good]

    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set log scale BEFORE drawing any artist. If scatter/fill_between are
        # drawn on a linear axis and the scale is flipped to log afterwards,
        # the re-autoscale can produce a degenerate y-range (and a multi-
        # thousand-inch figure under bbox_inches='tight'). The hexbin branch
        # avoids this by passing xscale/yscale at draw time; do the same here.
        ax.set_xscale("log")
        ax.set_yscale("log")

        if color_by_radius:
            inner = rad_f < 0.45
            ax.scatter(mu_f[inner], var_f[inner], s=6, alpha=0.35,
                       color="#185FA5", edgecolors="none", label="inner halo")
            ax.scatter(mu_f[~inner], var_f[~inner], s=6, alpha=0.35,
                       color="#1D9E75", edgecolors="none", label="outer halo")
        else:
            hb = ax.hexbin(mu_f, var_f, gridsize=gridsize, xscale="log",
                           yscale="log", cmap="inferno", mincnt=1, bins="log")
            cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("pixel count", fontsize=max(8, int(font_size * 0.85)))

        # reference lines span the data
        lo = max(mu_floor, np.nanmin(mu_f))
        hi = np.nanmax(mu_f)
        line_mu = np.logspace(np.log10(lo), np.log10(hi), 200)
        ax.plot(line_mu, line_mu, "--", color="0.35", lw=1.8, zorder=5,
                label=r"Poisson  $\sigma^2=\mu$")

        if fit:
            F, rmu, rvar, rcnt = _fit_fano(mu_f, var_f)
            # ridge points sized by how many pixels back each bin: large dots
            # are well-constrained, tiny dots (sparse bright core) are not.
            sizes = 12 + 60 * (rcnt / rcnt.max())
            ax.scatter(rmu, rvar, s=sizes, color="#D85A30", zorder=7,
                       edgecolors="white", linewidths=0.5,
                       label="binned-median ridge (size $\\propto$ N pix)")
            ax.plot(line_mu, F * line_mu, "-", color="#D85A30", lw=2.0, zorder=6,
                    label=rf"fit  $\sigma^2=F\,\mu$,  $F={F:.2f}$")

            # optional clean halo-only Fano: fit excluding the bright core
            F_halo = None
            if fano_mu_max is not None:
                F_halo, _, _, _ = _fit_fano(mu_f, var_f, mu_max=fano_mu_max)
                halo_mu = line_mu[line_mu < fano_mu_max]
                ax.plot(halo_mu, F_halo * halo_mu, "-", color="#0F6E56",
                        lw=2.0, zorder=6,
                        label=rf"halo-only ($\mu<{fano_mu_max:g}$): $F={F_halo:.2f}$")
                ax.axvline(fano_mu_max, color="#0F6E56", ls=":", lw=1.0, alpha=0.5)

            floor = F_halo if F_halo is not None else F
            ax.text(0.97, 0.04,
                    rf"$F=\sigma^2/\mu \approx {floor:.2f}$" "\n"
                    rf"$\Rightarrow \chi^2_{{\rm red}}$ floor $\approx {floor:.1f}$",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=max(9, int(font_size * 0.85)), color="#993C1D",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FAECE7",
                              ec="#D85A30", lw=0.8))

        if fit_quad:
            Fq, alpha = _fit_fano_plus_quad(mu_f, var_f)
            crossover = 1.0 / alpha ** 2
            ax.plot(line_mu, Fq * line_mu + (alpha * line_mu) ** 2,
                    ":", color="#534AB7", lw=1.6, zorder=6,
                    label=(rf"$F\mu+(\alpha\mu)^2$: $F={Fq:.2f}$, $\alpha={alpha:.3f}$"
                           rf"  (cross $\mu\approx{crossover:.0f}$)"))

        if n_seeds is not None and n_seeds > 1:
            frac = np.sqrt(2.0 / (n_seeds - 1))
            ax.fill_between(line_mu, line_mu * (1 - frac), line_mu * (1 + frac),
                            color="0.5", alpha=0.12, zorder=1,
                            label=rf"$\pm1\sigma$ on $\sigma^2$ ($N={n_seeds}$)")

        ax.set_xlabel(r"mean count  $\mu$  (photons / pixel)", fontsize=font_size)
        ax.set_ylabel(r"sample variance  $\sigma^2$", fontsize=font_size)
        ax.set_title(title, fontsize=font_size, pad=8)
        ax.legend(fontsize=max(8, int(font_size * 0.82)), framealpha=0.9,
                  loc="upper left")
        ax.grid(True, which="both", alpha=0.12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.12)

        # Guard: a degenerate log axis under bbox_inches='tight' can balloon the
        # rendered canvas to tens of thousands of px. Clamp the data limits to
        # the finite point cloud so the render can never exceed sane bounds.
        finite_y = var_f[np.isfinite(var_f) & (var_f > 0)]
        if finite_y.size:
            ax.set_ylim(finite_y.min() * 0.5, finite_y.max() * 2.0)
        finite_x = mu_f[np.isfinite(mu_f) & (mu_f > 0)]
        if finite_x.size:
            ax.set_xlim(finite_x.min() * 0.8, finite_x.max() * 1.25)

    return fig, ax



from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

def radial_reduce_stack(
    stack: np.ndarray,
    *,
    n_annuli: int = 24,
    r_max: Optional[float] = None,
) -> dict:
    """
    Compute per-seed radial profiles for every cell, then reduce across seeds.

    Works in TOTAL COUNTS per annulus (not mean intensity), so the Fano factor
    is simply Var_seed(counts) / mean_seed(counts) with no N_pix factor.

    Parameters
    ----------
    stack : (N_seeds, N_cells, H, W)
    n_annuli : number of radial bins
    r_max : optional cap on outer radius (in pixels). Defaults to corner distance.

    Returns
    -------
    dict with:
      r_centers     : (n_annuli,)            — bin centers in pixels
      r_edges       : (n_annuli + 1,)
      n_pix_per_bin : (n_annuli,)            — pixels per annulus
      counts        : (N_seeds, N_cells, n_annuli)  — total counts per annulus per seed
      mean_counts   : (N_cells, n_annuli)    — across-seed mean of counts
      var_counts    : (N_cells, n_annuli)    — across-seed variance of counts
      fano_profile  : (N_cells, n_annuli)    — var_counts / mean_counts
      mean_profile  : (N_cells, n_annuli)    — mean intensity (counts / N_pix), for masking/plotting
    """
    N_seeds, N_cells, H, W = stack.shape
    cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    if r_max is None:
        r_max = float(rr.max())
    edges = np.linspace(0.0, r_max, n_annuli + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    bins_flat = np.digitize(rr.ravel(), edges) - 1
    bins_flat = np.clip(bins_flat, 0, n_annuli - 1)   # (H*W,)

    n_pix_per_bin = np.bincount(bins_flat, minlength=n_annuli).astype(float)

    # Total counts per annulus per (seed, cell) via one-hot matmul.
    # onehot: (H*W, n_annuli), 1 where pixel belongs to that annulus.
    onehot = np.zeros((bins_flat.size, n_annuli), dtype=stack.dtype)
    onehot[np.arange(bins_flat.size), bins_flat] = 1.0

    # (N_seeds*N_cells, H*W) @ (H*W, n_annuli) -> (N_seeds*N_cells, n_annuli)
    flat = stack.reshape(N_seeds * N_cells, H * W)
    counts = (flat @ onehot).reshape(N_seeds, N_cells, n_annuli).astype(float)

    if N_seeds >= 2:
        mean_counts = counts.mean(axis=0)
        var_counts = counts.var(axis=0, ddof=1)
    else:
        mean_counts = counts[0]
        var_counts = np.full_like(mean_counts, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        fano_profile = var_counts / np.maximum(mean_counts, 1e-30)

    # Mean intensity, kept for masking / display only
    mean_profile = mean_counts / np.maximum(n_pix_per_bin, 1)[None, :]

    return {
        "r_centers": r_centers,
        "r_edges": edges,
        "n_pix_per_bin": n_pix_per_bin,
        "counts": counts,
        "mean_counts": mean_counts,
        "var_counts": var_counts,
        "fano_profile": fano_profile,
        "mean_profile": mean_profile,
    }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

def plot_model_fano_overlay(
    radial, x_raw, nH_raw,
    *,
    pixel_scale_arcsec=43.2,
    min_counts_per_bin=5.0,
    figsize=(9, 6),
    font_size=12,
    show=True,
):
    """All N_cond Fano(r) curves overlaid, colored by nH. Unweighted global mean."""
    r = radial["r_centers"] * pixel_scale_arcsec
    fano = radial["fano_profile"].copy()      # (N_cond, n_annuli)
    mean_counts = radial["mean_counts"]

    low = mean_counts < min_counts_per_bin
    fano[low] = np.nan

    nH = np.asarray(nH_raw, float)
    norm = LogNorm(vmin=nH.min(), vmax=nH.max())
    cmap = plt.cm.viridis

    with plt.rc_context({"font.size": font_size}):
        fig, ax = plt.subplots(figsize=figsize)
        for k in range(fano.shape[0]):
            ax.plot(r, fano[k], color=cmap(norm(nH[k])), lw=0.8, alpha=0.5)

        # median curve across conditions + Poisson + global mean
        med = np.nanmedian(fano, axis=0)
        ax.plot(r, med, color="k", lw=2.0, label="median over conditions")
        ax.axhline(1.0, color="0.5", ls="--", lw=1.0, label="Poisson")

        gmean = np.nanmean(fano)

        w = np.where(low, np.nan, mean_counts)          # counts as weights
        ok = np.isfinite(fano) & np.isfinite(w) & (w > 0)
        gmean_w = np.average(fano[ok], weights=w[ok])

        ax.axhline(gmean, color="red", ls=":", lw=1.5,
                   label=f"global mean (unweighted) = {gmean:.2f}")
        ax.axhline(gmean_w, color="darkred", ls="-.", lw=1.5,
                   label=f"global mean (weighted) = {gmean_w:.2f}")

        ax.set_xlabel("r [arcsec]")
        ax.set_ylabel(r"Fano  $\mathrm{Var}_{\rm seed}(C_r)/\langle C_r\rangle$")
        ax.set_title("Diffusion model radial Fano (64 conditions)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=font_size - 2)

        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r"$n_H$")
        fig.tight_layout()
        if show:
            plt.show()

    print(f"Global mean Fano  unweighted = {gmean:.4f}   weighted = {gmean_w:.4f}")
    return fig, ax
