from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PIXEL_SCALE_ARCSEC = 43.2


def _normalize_and_log_scale(
    image,
    a: float = 1000.0,
    *,
    data_min: float | None = None,
    data_max: float | None = None,
):
    """Normalize an image to [0, 1] and apply logarithmic scaling."""
    image = np.asarray(image, dtype=np.float64)
    img_min = float(np.min(image)) if data_min is None else float(data_min)
    img_max = float(np.max(image)) if data_max is None else float(data_max)

    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image)

    log_scaled = np.log(a * normalized + 1.0) / np.log(a)
    return log_scaled, img_min, img_max


def compare_npz_photon_totals_by_params(
    x: float,
    nH: float,
    npz_paths,
    *,
    labels=None,
    decimals: int = 6,
    cmaps=None,
    a: float = 1000.0,
    diff_cmap: str = "RdBu_r",
    figsize=None,
    font_size: int = 12,
):
    """
    Like compare_npz_by_params, but for exactly 2 .npz archives: pulls the
    (x, nH) matrix out of each archive, displays both images side by side
    (log scale, shared colour range) plus a linear difference image
    (npz_paths[0] - npz_paths[1], diverging colormap centered at 0), and
    prints the total photon count in each image along with their difference.

    x, nH : float
        Parameter values to look up in each archive.
    npz_paths : sequence of str/Path
        Exactly 2 .npz archive paths (as produced by save_npy_to_npz, keyed
        by "xs"/"nHs"/"mats" - the same archives compare_npz_by_params and
        error_heatmap_between_npz read).
    labels : list of str, optional
        Display labels. Defaults to each archive's filename stem.
    decimals : int
        Rounding applied to each archive's xs/nHs before matching against
        x, nH (guards against float rounding drift between archives).
    """
    npz_paths = list(npz_paths)
    if len(npz_paths) != 2:
        raise ValueError(
            f"compare_npz_photon_totals_by_params compares exactly 2 .npz files, got {len(npz_paths)}."
        )

    x_r = round(float(x), decimals)
    nH_r = round(float(nH), decimals)
    print(f"x={x} -> matched x={x_r}")
    print(f"nH={nH} -> matched nH={nH_r}")

    matrices = []
    fallback_labels = []
    for p in npz_paths:
        with np.load(p) as data:
            xs_arr = np.round(data["xs"].astype(float), decimals)
            nHs_arr = np.round(data["nHs"].astype(float), decimals)
            match = np.where((xs_arr == x_r) & (nHs_arr == nH_r))[0]
            if len(match) == 0:
                raise ValueError(
                    f"No matrix for x={x} (rounded {x_r}), nH={nH} (rounded {nH_r}) found in {p}"
                )
            arr = np.squeeze(np.asarray(data["mats"][match[0]], dtype=np.float64))
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected 2D matrix after squeeze, got {arr.shape}")
        matrices.append(arr)
        fallback_labels.append(Path(p).stem)

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < 2:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[0], base_labels[1]]

    return _photon_totals_diff_plot(
        matrices, labels,
        cmaps=cmaps, a=a, diff_cmap=diff_cmap, figsize=figsize, font_size=font_size,
    )


def compare_npy_photon_totals_by_params(
    x: float,
    nH: float,
    folders,
    subdir: str = "",
    *,
    labels=None,
    cmaps=None,
    a: float = 1000.0,
    diff_cmap: str = "RdBu_r",
    figsize=None,
    font_size: int = 12,
    title: str | None = None,
    show_sbp: bool = True,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    bin_filter: float = 10.0,
    center: str = "geom",
    x_lim=None,
    y_lim=None,
):
    """
    Like compare_npy_by_params, but for exactly 2 folders: finds the (x, nH)
    .npy file in each folder (via find_npy), displays both images side by
    side (log scale, shared colour range) plus a linear difference image
    (folders[0] - folders[1], diverging colormap centered at 0), and prints
    the total photon count in each image along with their difference.

    Also adds a radial surface-brightness profile panel below the images,
    computed the same way as compare_npy_by_params / compare_npz_by_params
    (per-annulus sums converted to counts/arcsec^2, with Poisson error bars),
    plus the SBP of the difference image itself (no chi2/N_best fit - just
    its own per-annulus sum, with error propagated in quadrature from the
    two input images' Poisson errors).

    folders : sequence of str/Path
        Exactly 2 folders, resolved relative to `subdir`. Each is searched
        (via find_npy, from io_npy.py) for a .npy file whose name contains
        "x_{x_str}" and "nH_{nH_str}". The LAST folder is treated as "data"
        for the radial-bin validity mask (bin_filter), same convention as
        compare_npy_by_params.
    subdir : str
        Common base directory prepended to every entry in `folders`.
    labels : list of str, optional
        Display labels. Defaults to each matched file's stem.
    title : str, optional
        Extra line shown above the totals summary in the figure title.
    show_sbp : bool
        If True (default), add the radial surface-brightness profile panel.
    n_annuli, pixel_scale_arcsec, bin_filter, center, x_lim, y_lim :
        Forwarded to the radial profile panel; see compare_npz_by_params.
    """
    from .io_npy import format_x, format_nH, find_npy

    folders = list(folders)
    if len(folders) != 2:
        raise ValueError(
            f"compare_npy_photon_totals_by_params compares exactly 2 folders, got {len(folders)}."
        )

    print(f"x={x} -> x_str={format_x(x)}")
    print(f"nH={nH} -> nH_str={format_nH(nH)}")

    paths = []
    for folder in folders:
        folder_path = str(Path(subdir) / folder)
        p = find_npy(folder_path, x, nH)
        if p is None:
            raise FileNotFoundError(
                f"No .npy file matching x={x} (x_str={format_x(x)}), "
                f"nH={nH} (nH_str={format_nH(nH)}) found under {folder_path}"
            )
        paths.append(p)

    matrices = []
    fallback_labels = []
    for p in paths:
        arr = np.squeeze(np.load(p).astype(np.float64))
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected 2D after squeeze, got {arr.shape}")
        matrices.append(arr)
        fallback_labels.append(Path(p).stem)

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < 2:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[0], base_labels[1]]

    return _photon_totals_diff_plot(
        matrices, labels,
        cmaps=cmaps, a=a, diff_cmap=diff_cmap, figsize=figsize, font_size=font_size,
        title=title,
        show_sbp=show_sbp, n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter, center=center, x_lim=x_lim, y_lim=y_lim,
    )


def _photon_totals_diff_plot(
    matrices,
    labels,
    *,
    cmaps=None,
    a: float = 1000.0,
    diff_cmap: str = "RdBu_r",
    figsize=None,
    font_size: int = 12,
    title: str | None = None,
    show_sbp: bool = False,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = PIXEL_SCALE_ARCSEC,
    bin_filter: float = 10.0,
    center: str = "geom",
    x_lim=None,
    y_lim=None,
):
    """
    Shared implementation behind compare_npz_photon_totals_by_params and
    compare_npy_photon_totals_by_params: display 2 already-loaded 2D matrices
    side by side (log scale, shared colour range) plus a linear difference
    image, print/return their total photon counts and difference, and
    optionally add a radial surface-brightness profile panel (data, other,
    and the difference image's own SBP - no chi2/N_best fit for the
    difference, just its own per-annulus sum with Poisson error propagated
    in quadrature).
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    mat_a, mat_b = matrices
    total_a = float(np.nansum(mat_a))
    total_b = float(np.nansum(mat_b))
    diff_total = total_a - total_b
    pct_diff = (diff_total / total_b * 100.0) if total_b != 0 else float("nan")

    print("\nTotal photon counts:")
    print(f"  {labels[0]}: {total_a:.4f}")
    print(f"  {labels[1]}: {total_b:.4f}")
    print(f"  Difference ({labels[0]} - {labels[1]}) = {diff_total:.4f}  ({pct_diff:+.2f}% of {labels[1]})")

    diff_img = mat_a - mat_b

    default_cmaps = ["magma", "magma"]
    base_cmaps = default_cmaps if cmaps is None else list(cmaps)
    if len(base_cmaps) < 2:
        base_cmaps = base_cmaps + default_cmaps[len(base_cmaps):]
    cmaps = base_cmaps[:2]

    shared_min = float(min(mat_a.min(), mat_b.min()))
    shared_max = float(max(mat_a.max(), mat_b.max()))

    if figsize is None:
        figsize = (15, 10) if show_sbp else (15, 5)

    if show_sbp:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_diff = fig.add_subplot(gs[0, 2])
        ax_sbp = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_a, ax_b, ax_diff) = plt.subplots(1, 3, figsize=figsize)
        ax_sbp = None

    for ax, arr, label, cmap in zip((ax_a, ax_b), (mat_a, mat_b), labels, cmaps):
        scaled, _, _ = _normalize_and_log_scale(
            arr,
            a,
            data_min=shared_min,
            data_max=shared_max,
        )
        ax.imshow(scaled, cmap=cmap, origin="lower")
        ax.set_title(label, fontsize=font_size)
        ax.set_xticks([])
        ax.set_yticks([])

    diff_min = float(diff_img.min())
    diff_max = float(diff_img.max())
    if diff_min >= 0.0:
        diff_min = -abs(diff_max) - 1e-9
    if diff_max <= 0.0:
        diff_max = abs(diff_min) + 1e-9
    norm = TwoSlopeNorm(vmin=diff_min, vcenter=0.0, vmax=diff_max)

    im = ax_diff.imshow(diff_img, cmap=diff_cmap, norm=norm, origin="lower")
    diff_label = f"{labels[0]} − {labels[1]}"
    ax_diff.set_title(diff_label, fontsize=font_size)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

    if show_sbp:
        # --- Radial geometry (mat_b = "data", same convention as
        # compare_npy_by_params / compare_npz_by_params) ---
        template = mat_b
        H, W = template.shape
        yy, xx = np.indices((H, W))
        if center == "peak":
            cy, cx = np.unravel_index(np.nanargmax(template), template.shape)
            cy, cx = float(cy), float(cx)
        else:
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = rr.max()
        edges_pix = np.linspace(0.0, r_max, n_annuli + 1)
        bins = np.clip(np.digitize(rr.ravel(), edges_pix) - 1, 0, n_annuli - 1)
        bin_centers_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
        bin_centers_arc = bin_centers_pix * pixel_scale_arcsec

        N_pixels = np.bincount(bins, minlength=n_annuli).astype(np.float64)
        area_bin = N_pixels * (pixel_scale_arcsec ** 2)

        def _radial_sum(arr):
            return np.bincount(bins, weights=np.nan_to_num(arr.ravel()), minlength=n_annuli)

        S_a = _radial_sum(mat_a)
        S_b = _radial_sum(mat_b)
        S_diff = S_a - S_b

        valid_bins = S_b > bin_filter
        xv = bin_centers_arc[valid_bins]

        with np.errstate(divide="ignore", invalid="ignore"):
            sb_a = S_a[valid_bins] / area_bin[valid_bins]
            sb_b = S_b[valid_bins] / area_bin[valid_bins]
            sb_diff = S_diff[valid_bins] / area_bin[valid_bins]
            err_a = np.sqrt(S_a[valid_bins]) / area_bin[valid_bins]
            err_b = np.sqrt(S_b[valid_bins]) / area_bin[valid_bins]
            # Poisson error propagated in quadrature - no chi2/N_best fit.
            err_diff = np.sqrt(S_a[valid_bins] + S_b[valid_bins]) / area_bin[valid_bins]

        tab10 = plt.colormaps.get_cmap("tab10")
        sbp_colors = [tab10(0), tab10(1), tab10(2)]
        sbp_profiles = [sb_a, sb_b, sb_diff]
        sbp_errors = [err_a, err_b, err_diff]
        sbp_labels = [labels[0], labels[1], diff_label]

        for profile, err, lbl, color in zip(sbp_profiles, sbp_errors, sbp_labels, sbp_colors):
            ax_sbp.plot(xv, profile, "-", color=color, label=lbl)
            ax_sbp.errorbar(
                xv, profile, yerr=err,
                fmt="none", ecolor=color, elinewidth=1.2, alpha=0.8, capsize=0,
            )

        ax_sbp.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
        ax_sbp.set_yscale("symlog")
        ax_sbp.set_xlabel(r"Radius $\left[\mathrm{arcsec}\right]$", fontsize=font_size)
        ax_sbp.set_ylabel(
            r"Surface Brightness $\left[\mathrm{photons}\,\mathrm{arcsec}^{-2}\right]$",
            fontsize=font_size,
        )
        ax_sbp.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_sbp.tick_params(axis="both", labelsize=font_size)
        ax_sbp.legend(fontsize=font_size)
        ax_sbp.grid(True, which="both", linestyle="--", alpha=0.3)
        if x_lim is not None:
            ax_sbp.set_xlim(x_lim)
        if y_lim is not None:
            ax_sbp.set_ylim(y_lim)

    totals_line = (
        f"Total: {labels[0]}={total_a:.1f}, {labels[1]}={total_b:.1f}, "
        f"Δ={diff_total:.1f} ({pct_diff:+.2f}%)"
    )
    fig.suptitle(
        f"{title}\n{totals_line}" if title else totals_line,
        fontsize=font_size,
    )
    fig.tight_layout()
    plt.show()

    return {
        "labels": labels,
        "totals": [total_a, total_b],
        "difference": diff_total,
        "percent_difference": pct_diff,
        "diff_image": diff_img,
    }
