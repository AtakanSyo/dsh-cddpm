from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

def radial_reduced_chi2_area2_bin_filtered(
    data,
    model,
    V=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 43.2,
    bin_filter: float = 10.0,
    return_bin_errors: bool = False,
    return_profiles: bool = False,
    return_chi2_contributions: bool = False,
    return_N_best: bool = False,
    use_N: bool = True,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
):
    """
    Calculate the radial reduced chi-squared between a data image and a
    normalized model image.

    The images are reduced to radial surface-brightness profiles. A
    multiplicative model normalization, N_best, is fitted by directly
    minimizing

        chi2(N) = sum_i [
            (mu_data,i - N * mu_model,i)^2
            /
            (
                Fano_Data * S_data,i / A_i^2
                + N^2 * Fano_Model * V_model,i / A_i^2
            )
        ].

    Equivalently, in count-space variance notation,

        chi2(N) = sum_i [
            A_i^2 * (mu_data,i - N * mu_model,i)^2
            /
            (
                Fano_Data * S_data,i
                + N^2 * Fano_Model * V_model,i
            )
        ].

    Parameters
    ----------
    data : array_like
        Observed count image.

    model : array_like
        Model count image.

    V : array_like, optional
        Per-pixel variance of the model before normalization.

        For a deterministic bilinear interpolant in
        (x, log10(nH)),

            V = sum_k w_k^2 I_k.

        If V is None, the model is treated as a Poisson realization and
        its per-pixel variance is taken to be equal to its counts.

    n_annuli : int
        Number of radial annuli.

    pixel_scale_arcsec : float
        Pixel width in arcseconds.

    bin_filter : float
        Only annuli with S_data > bin_filter are included.

    return_bin_errors : bool
        Return the total one-sigma surface-brightness uncertainty per
        annulus, evaluated at N_best.

    return_profiles : bool
        Return the data and unnormalized model radial profiles, along
        with N_best.

    return_chi2_contributions : bool
        Return each annulus's contribution to chi-squared.

    return_N_best : bool
        Return the fitted model normalization.

    Fano_Data : float
        Multiplicative variance factor for the data.

    Fano_Model : float
        Multiplicative variance factor for the model.

    Returns
    -------
    float or tuple
        Reduced chi-squared, optionally followed by N_best and requested
        diagnostic arrays.

    Notes
    -----
    The degrees of freedom are

        dof = number of valid annuli - 1,

    because one normalization parameter is fitted.

    This minimizes the explicitly defined chi-squared statistic. It is
    not a full Gaussian maximum-likelihood fit because the variance
    depends on N and no logarithmic variance term is included.
    """

    # ------------------------------------------------------------------
    # 2. Validate images.
    # ------------------------------------------------------------------
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    model = np.squeeze(np.asarray(model, dtype=np.float64))

    if data.shape != model.shape:
        raise ValueError(
            f"data shape {data.shape} must match model shape {model.shape}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or infinite values.")

    if not np.all(np.isfinite(model)):
        raise ValueError("model contains NaN or infinite values.")

    if np.any(data < 0.0):
        raise ValueError("data contains negative counts.")

    if np.any(model < 0.0):
        raise ValueError("model contains negative counts.")

    if V is not None:
        V = np.squeeze(np.asarray(V, dtype=np.float64))

        if V.ndim != 2:
            raise ValueError(
                f"V must be 2-D after squeezing; got shape {V.shape}."
            )

        if V.shape != model.shape:
            raise ValueError(
                f"V shape {V.shape} must match model shape {model.shape}."
            )

        if not np.all(np.isfinite(V)):
            raise ValueError("V contains NaN or infinite values.")

        if np.any(V < 0.0):
            raise ValueError("V contains negative variance values.")

    height, width = data.shape

    # ------------------------------------------------------------------
    # 3. Construct radial bins.
    # ------------------------------------------------------------------
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    yy, xx = np.indices((height, width))

    radius_pixels = np.sqrt(
        (yy - cy) ** 2
        + (xx - cx) ** 2
    )

    edges = np.linspace(
        0.0,
        radius_pixels.max(),
        n_annuli + 1,
    )

    bins = np.digitize(
        radius_pixels.ravel(),
        edges,
        right=False,
    ) - 1

    bins = np.clip(
        bins,
        0,
        n_annuli - 1,
    )

    # ------------------------------------------------------------------
    # 4. Sum counts and model variance within annuli.
    # ------------------------------------------------------------------
    S_data = np.bincount(
        bins,
        weights=data.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    S_model = np.bincount(
        bins,
        weights=model.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    N_pixels = np.bincount(
        bins,
        minlength=n_annuli,
    ).astype(np.float64)

    if V is None:
        V_model = S_model.copy()
    else:
        V_model = np.bincount(
            bins,
            weights=V.ravel(),
            minlength=n_annuli,
        ).astype(np.float64)
    # ------------------------------------------------------------------
    # 6. Convert counts to surface brightness.
    # ------------------------------------------------------------------
    area_per_pixel = pixel_scale_arcsec**2
    area_bin = N_pixels * area_per_pixel

    valid_bins = (
        (S_data > bin_filter)
        & (area_bin > 0.0)
        & np.isfinite(S_data)
        & np.isfinite(S_model)
        & np.isfinite(V_model)
    )

    valid_indices = np.flatnonzero(valid_bins)

    def empty_result():
        extras = []

        if return_bin_errors:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )

        if return_profiles:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )
            extras.append(float("nan"))

        if return_chi2_contributions:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )

        n_best_slot = (
            (float("nan"),)
            if return_N_best
            else ()
        )

        if extras or return_N_best:
            return (
                float("nan"),
                *n_best_slot,
                *extras,
            )

        return float("nan")

    if len(valid_indices) == 0:
        return empty_result()

    mu_data = np.zeros(n_annuli, dtype=np.float64)
    mu_model = np.zeros(n_annuli, dtype=np.float64)

    mu_data[valid_bins] = (
        S_data[valid_bins]
        / area_bin[valid_bins]
    )

    mu_model[valid_bins] = (
        S_model[valid_bins]
        / area_bin[valid_bins]
    )

    md = mu_data[valid_bins]
    mm = mu_model[valid_bins]

    Sd = S_data[valid_bins]
    Vm = V_model[valid_bins]
    Ab = area_bin[valid_bins]

# ------------------------------------------------------------------
    # 7. Determine model normalization.
    #    If use_N is True, fit N_best by minimizing the reported chi2.
    #    If use_N is False, fix N = 1 (no parameter fitted), which is
    #    appropriate when data and model share the same absolute photon
    #    normalization (e.g. cDDPM / interpolation / RefleX at the same
    #    photon budget). Fixing N avoids the small-nu bias that the
    #    fitted normalization introduces into the reduced chi2.
    # ------------------------------------------------------------------
    def chi2_for_normalization(
        normalization: float,
    ) -> float:
        if (
            not np.isfinite(normalization)
            or normalization < 0.0
        ):
            return float("inf")

        variance_counts = (
            Fano_Data * Sd
            + normalization**2 * Fano_Model * Vm
        )

        positive_variance = variance_counts > 0.0

        if not np.all(positive_variance):
            return float("inf")

        residuals = (
            md
            - normalization * mm
        )

        contributions = (
            Ab**2
            * residuals**2
            / variance_counts
        )

        value = np.sum(contributions)

        if not np.isfinite(value):
            return float("inf")

        return float(value)

    if use_N:
        # Obtain a stable scale estimate for the optimization bounds.
        model_sum = float(np.sum(S_model[valid_bins]))
        data_sum = float(np.sum(S_data[valid_bins]))

        if model_sum > 0.0:
            normalization_scale = data_sum / model_sum
        else:
            normalization_scale = 1.0

        normalization_scale = max(
            normalization_scale,
            1.0,
        )

        upper_bound = max(
            10.0,
            10.0 * normalization_scale,
        )

        optimization = minimize_scalar(
            chi2_for_normalization,
            bounds=(0.0, upper_bound),
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 1000,
            },
        )

        if (
            not optimization.success
            or not np.isfinite(optimization.fun)
        ):
            raise RuntimeError(
                "Could not determine N_best. "
                f"Optimizer message: {optimization.message}"
            )

        N_best = float(optimization.x)

        # Warn through an exception if the optimum appears constrained by
        # the automatically selected upper bound.
        if N_best >= 0.999 * upper_bound:
            raise RuntimeError(
                "The fitted normalization reached the optimization upper "
                f"bound ({upper_bound:.6g}). Increase the bound or inspect "
                "the data/model scaling."
            )
    else:
        # No normalization fitted: fix N = 1.
        N_best = 1.0

    # ------------------------------------------------------------------
    # 8. Calculate final chi-squared contributions.
    # ------------------------------------------------------------------
    sigma2_counts = (
        Fano_Data * Sd
        + N_best**2 * Fano_Model * Vm
    )

    if np.any(sigma2_counts <= 0.0):
        raise ValueError(
            "One or more valid annuli have non-positive total variance."
        )

    residuals = (
        md
        - N_best * mm
    )

    chi2_contributions_valid = (
        Ab**2
        * residuals**2
        / sigma2_counts
    )

    chi2 = float(
        np.sum(chi2_contributions_valid)
    )

    # Degrees of freedom: subtract one only if a normalization was fitted.
    if use_N:
        dof = len(valid_indices) - 1
    else:
        dof = len(valid_indices)

    if dof <= 0:
        chi2_red = float("nan")
    else:
        chi2_red = chi2 / dof

    # ------------------------------------------------------------------
    # 9. Assemble optional outputs.
    # ------------------------------------------------------------------
    extras = []

    if return_bin_errors:
        total_profile_variance = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        total_profile_variance[valid_bins] = (
            sigma2_counts
            / Ab**2
        )

        total_profile_errors = np.sqrt(
            total_profile_variance
        )

        extras.append(
            total_profile_errors.tolist()
        )

    if return_profiles:
        extras.append(mu_data.tolist())
        extras.append(mu_model.tolist())
        extras.append(float(N_best))

    if return_chi2_contributions:
        full_contributions = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        full_contributions[valid_bins] = (
            chi2_contributions_valid
        )

        extras.append(
            full_contributions.tolist()
        )

    n_best_slot = (
        (float(N_best),)
        if return_N_best
        else ()
    )

    if extras or return_N_best:
        return (
            float(chi2_red),
            *n_best_slot,
            *extras,
        )

    return float(chi2_red)


def _call_chi2_pair(mat_a, mat_b, *, V, n_annuli, pixel_scale_arcsec, bin_filter, need_contributions, Fano_Data=1.0, Fano_Model=1.0):
    """Call radial_reduced_chi2_area2_bin_filtered and return a normalized result tuple."""
    val, errors, mu_data, mu_model, N_best, *extra = radial_reduced_chi2_area2_bin_filtered(
        mat_a, mat_b,
        V=V,
        n_annuli=n_annuli,
        pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter,
        return_bin_errors=True,
        return_profiles=True,
        return_chi2_contributions=need_contributions,
        Fano_Data=Fano_Data,
        Fano_Model=Fano_Model
    )
    contrib = np.asarray(extra[0], dtype=np.float64) if need_contributions and extra else np.zeros(n_annuli, dtype=np.float64)
    valid = np.asarray(errors, dtype=np.float64) > 0.0
    dof = max(int(np.count_nonzero(valid)) - 1, 1)
    summary_str = f"N_best={N_best:.4f}"
    helper_data = (np.asarray(errors, dtype=np.float64), np.asarray(mu_data, dtype=np.float64), np.asarray(mu_model, dtype=np.float64), float(N_best))
    return val, contrib, dof, summary_str, helper_data

def compare_npz_by_params(
    x: float,
    nH: float,
    npz_paths,
    *,
    labels=None,
    decimals: int = 6,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
):
    """
    Like compare_npy_matrices, but pulls the (x, nH) matrix directly out of
    saved .npz archives (as produced by save_npy_to_npz, keyed by "xs"/"nHs"/
    "mats" - the same archives error_heatmap_between_npz reads) instead of
    loading individual .npy files.

    x, nH : float
        Parameter values to look up in every archive.
    npz_paths : sequence of str/Path
        One .npz archive per matrix to compare. As with compare_npy_matrices,
        the LAST archive's matrix is treated as the reference "data".
    labels : list of str, optional
        Display labels. Defaults to each archive's filename stem.
    decimals : int
        Rounding applied to each archive's xs/nHs before matching against
        x, nH (guards against float rounding drift between archives).
    """
    if len(npz_paths) < 1:
        raise ValueError("Provide at least 1 .npz file path.")

    n = len(npz_paths)
    x_r = round(float(x), decimals)
    nH_r = round(float(nH), decimals)
    print(f"x={x} -> matched x={x_r}")
    print(f"nH={nH} -> matched nH={nH_r}")

    matrices = []
    model_variances = []
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
            match_index = int(match[0])
            arr = np.squeeze(np.asarray(data["mats"][match_index], dtype=np.float64))
            variance = (
                np.squeeze(np.asarray(data["Vs"][match_index], dtype=np.float64))
                if "Vs" in data.files
                else None
            )
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected 2D matrix after squeeze, got {arr.shape}")
        matrices.append(arr)
        model_variances.append(variance)
        fallback_labels.append(Path(p).stem)

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < n:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[i] for i in range(n)]

    return _compare_matrices(
        matrices, labels,
        model_variances=model_variances,
        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter, Fano_Data=Fano_Data, Fano_Model=Fano_Model,
        cmaps=cmaps, a=a, center=center, figsize=figsize, font_size=font_size,
        x_lim=x_lim, y_lim=y_lim, show_bin_contributions=show_bin_contributions,
        show_mean_bin_contribution_plot=show_mean_bin_contribution_plot,
        show_chi2_label=False,
    )
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# Colourblind-safe palette (Wong/Okabe-Ito-adjacent); reference is black.
_CB_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"]
_LINESTYLES = ["-", "--", "-.", ":"]

_MNRAS_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _compare_matrices(
    matrices,
    labels,
    *,
    model_variances=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 9,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
    show_chi2_label: bool = True,
    # ---- NEW styling parameters ----
    journal_style: bool = True,      # apply MNRAS-like rc styling locally
    ref_as_points: bool = True,      # reference (last matrix) as black points
    log_x: bool = False,             # log-scale radius axis
    scalebar_arcsec: float | None = 500.0,  # scale bar on image panels; None disables
    panel_labels: bool = True,       # (a), (b), (c) corner labels
    save_path=None,                  # save figure (pdf recommended)
    show: bool = True,
):
    """
    Shared implementation behind compare_npy_matrices and compare_npz_by_params:
    display already-loaded 2D matrices in log scale with a shared radial
    surface-brightness profile panel below, and compute a pairwise
    goodness-of-fit metric for every pair.

    Styling follows MNRAS figure conventions when journal_style=True:
    serif/STIX fonts, inward ticks on all four sides with minor ticks,
    no grid, frameless legend, reference profile drawn as black data
    points with error bars and models as colourblind-safe lines, a
    shared logarithmic colour scale across image panels with a scale
    bar and (a)/(b)/(c) panel labels.
    """
    from .plotting import normalize_and_log_scale
    from matplotlib.ticker import MaxNLocator  # type: ignore[import]

    n = len(matrices)
    if model_variances is None:
        model_variances = [None] * n
    elif len(model_variances) != n:
        raise ValueError(
            "model_variances must have the same length as matrices."
        )

    validated_variances = []
    for index, (matrix, variance) in enumerate(zip(matrices, model_variances)):
        if variance is None:
            validated_variances.append(None)
            continue

        variance = np.squeeze(np.asarray(variance, dtype=np.float64))
        if variance.shape != matrix.shape:
            raise ValueError(
                f"Variance shape mismatch at index {index}: "
                f"V={variance.shape}, matrix={matrix.shape}."
            )
        if not np.all(np.isfinite(variance)):
            raise ValueError(
                f"Variance matrix at index {index} contains NaN or infinite values."
            )
        if np.any(variance < 0.0):
            raise ValueError(
                f"Variance matrix at index {index} contains negative values."
            )
        validated_variances.append(variance)

    model_variances = validated_variances

    # Image panels: one perceptually-uniform map shared by all panels is
    # the journal-preferred choice; per-panel maps remain available by
    # passing an explicit cmaps list.
    if cmaps is None:
        cmaps = ["magma"] * n
    else:
        base = list(cmaps) or ["magma"]
        cmaps = [base[i % len(base)] for i in range(n)]

    # --- Shared colour range ---
    shared_min = float(min(m.min() for m in matrices))
    shared_max = float(max(m.max() for m in matrices))

    rc = _MNRAS_RC if journal_style else {}

    with plt.rc_context(rc):
        # --- Figure layout ---
        if figsize is None:
            # Sized for a two-column (page-width) MNRAS figure; use
            # figsize=(3.32, ...) explicitly for single-column.
            figsize = (2.3 * n, 4.8)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, n, height_ratios=[1, 1.35], hspace=0.12)

        img_axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
        ax_prof = fig.add_subplot(gs[1, :])

        # --- Top row: log-scaled images ---
        for k, (ax, arr, label, cmap) in enumerate(
            zip(img_axes, matrices, labels, cmaps)
        ):
            scaled, _, _ = normalize_and_log_scale(
                arr,
                a,
                data_min=shared_min,
                data_max=shared_max,
            )
            ax.imshow(
                scaled,
                cmap=cmap,
                origin="lower",
                interpolation="nearest",   # show the 48x48 pixels as pixels
            )
            ax.set_title(label, fontsize=font_size)
            ax.set_xticks([])
            ax.set_yticks([])

            if panel_labels:
                ax.text(
                    0.04, 0.96, f"({chr(ord('a') + k)})",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=font_size, color="white",
                )

            # Scale bar on the last (reference) panel only, to avoid clutter.
            if scalebar_arcsec is not None and k == n - 1:
                bar_pix = scalebar_arcsec / pixel_scale_arcsec
                H_img, W_img = arr.shape
                x0 = 0.95 * W_img - bar_pix
                y0 = 0.06 * H_img
                ax.plot([x0, x0 + bar_pix], [y0, y0],
                        color="white", lw=1.5, solid_capstyle="butt")
                ax.text(x0 + bar_pix / 2.0, y0 + 0.03 * H_img,
                        rf"${scalebar_arcsec:.0f}''$",
                        ha="center", va="bottom",
                        fontsize=font_size - 1, color="white")

        # --- Radial geometry (last matrix as template) ---
        template = matrices[-1]
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
        area_per_pix = pixel_scale_arcsec ** 2
        area_bin = N_pixels * area_per_pix

        def _radial_sum(arr):
            return np.bincount(
                bins, weights=np.nan_to_num(arr.ravel()), minlength=n_annuli
            )

        sums = [_radial_sum(m) for m in matrices]
        variance_sums = [
            _radial_sum(variance) if variance is not None else sums[index]
            for index, variance in enumerate(model_variances)
        ]
        sb_profiles = [np.zeros(n_annuli, dtype=np.float64) for _ in matrices]
        for sb, s in zip(sb_profiles, sums):
            valid_sb = s > bin_filter
            sb[valid_sb] = s[valid_sb] / area_bin[valid_sb]

        # --- Pairwise metric computation (unchanged) ---
        results = {}

        ref_idx = n - 1
        ref_sum = sums[ref_idx]
        valid_ref = ref_sum > bin_filter

        err_ref = np.zeros(n_annuli, dtype=np.float64)
        err_ref[valid_ref] = (
            np.sqrt(Fano_Data * ref_sum[valid_ref])
            / area_bin[valid_ref]
        )

        helper_profiles = {}
        pairwise_red_contributions = []
        pairwise_mean_values = []
        ref_chi2_values = []

        metric_label = "chi2_red"
        print(f"\nPairwise {metric_label} (chi2):")
        print(
            f"  n_annuli={n_annuli}, bin_filter={bin_filter}, "
            f"Fano Data={Fano_Data}, Fano Model={Fano_Model}, "
            f"pixel_scale={pixel_scale_arcsec}\""
        )
        print("-" * 55)
        need_pairwise_contributions = (
            show_bin_contributions or show_mean_bin_contribution_plot
        )
        if n == 1:
            print(f"  Single input matrix provided; pairwise {metric_label} skipped.")
        else:
            for (i, j) in combinations(range(n), 2):
                val_ij, contrib_ij, dof_ij, summary_str_ij, hdata_ij = _call_chi2_pair(
                    matrices[i], matrices[j],
                    V=model_variances[j],
                    n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                    bin_filter=bin_filter,
                    need_contributions=need_pairwise_contributions,
                    Fano_Data=Fano_Data, Fano_Model=Fano_Model,
                )
                val_ji, contrib_ji, dof_ji, summary_str_ji, hdata_ji = _call_chi2_pair(
                    matrices[j], matrices[i],
                    V=model_variances[i],
                    n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                    bin_filter=bin_filter,
                    need_contributions=need_pairwise_contributions,
                    Fano_Data=Fano_Data, Fano_Model=Fano_Model,
                )
                helper_profiles[(i, j)] = hdata_ij
                helper_profiles[(j, i)] = hdata_ji

                pair_mean = 0.5 * (val_ij + val_ji)
                pairwise_mean_values.append(pair_mean)
                if j == ref_idx:
                    ref_chi2_values.append(val_ji)
                elif i == ref_idx:
                    ref_chi2_values.append(val_ij)
                results[(i, j)] = (val_ij, val_ji)
                print(f"  {labels[i]} vs {labels[j]}")
                print(f"    {metric_label} ({labels[i]} as data) = {val_ij:.4f}   {summary_str_ij}")
                print(f"    {metric_label} ({labels[j]} as data) = {val_ji:.4f}   {summary_str_ji}")

                if need_pairwise_contributions:
                    red_contrib_ij = contrib_ij / float(dof_ij)
                    red_contrib_ji = contrib_ji / float(dof_ji)
                    pairwise_red_contributions.append(red_contrib_ij)
                    pairwise_red_contributions.append(red_contrib_ji)

                    if show_bin_contributions:
                        label_i = str(labels[i])
                        label_j = str(labels[j])
                        print(f"    per-annulus reduced {metric_label} contributions:")
                        print(
                            "      bin | radius[arcsec] | area[arcsec^2] | "
                            f"counts({label_i}) | counts({label_j}) | "
                            f"{label_i}->{label_j} | {label_j}->{label_i}"
                        )
                        for b in range(n_annuli):
                            print(
                                f"      {b+1:>3d} | {bin_centers_arc[b]:>13.3f} | "
                                f"{area_bin[b]:>13.3f} | "
                                f"{float(sums[i][b]):>12.3f} | "
                                f"{float(sums[j][b]):>13.3f} | "
                                f"{red_contrib_ij[b]:>11.4e} | "
                                f"{red_contrib_ji[b]:>11.4e}"
                            )

            if pairwise_mean_values:
                mean_all = float(np.mean(pairwise_mean_values))
                std_all = float(np.std(pairwise_mean_values))
                print(f"  Mean {metric_label} (all pairs, both directions) = {mean_all:.4f} ± {std_all:.4f}")
            if ref_chi2_values:
                mean_ref = float(np.mean(ref_chi2_values))
                std_ref = float(np.std(ref_chi2_values))
                print(f"  Mean {metric_label} (model fit to reference '{labels[ref_idx]}' as data) = {mean_ref:.4f} ± {std_ref:.4f}")
            if results:
                print(f"  Pairwise {metric_label} summary (mean of both directions):")
                for (i, j), (v_ij, v_ji) in results.items():
                    print(f"    {labels[i]} x {labels[j]} = {0.5 * (v_ij + v_ji):.4f}")
        print("-" * 55)

        # --- Radial profile panel ---
        x_vals = bin_centers_arc
        valid_bins_plot = valid_ref
        profiles_to_plot = [np.asarray(p, dtype=np.float64) for p in sb_profiles]
        ref_errors_plot = np.asarray(err_ref, dtype=np.float64)
        errors_plot = [np.zeros(n_annuli, dtype=np.float64) for _ in range(n)]
        errors_plot[ref_idx] = ref_errors_plot

        if n >= 2:
            ref_profile = None
            display_profiles = [None] * n

            for idx in range(n - 1):
                helper_key = (ref_idx, idx)
                if helper_key not in helper_profiles:
                    continue

                _, mu_ref, mu_other, n_best = helper_profiles[helper_key]
                if ref_profile is None:
                    ref_profile = mu_ref

                model_profile = np.zeros_like(mu_ref, dtype=np.float64)
                valid_bins = ref_sum > bin_filter
                model_profile[valid_bins] = n_best * mu_other[valid_bins]
                display_profiles[idx] = model_profile

                model_error = np.zeros_like(mu_ref, dtype=np.float64)
                model_error[valid_bins] = (
                    n_best
                    * np.sqrt(Fano_Model * variance_sums[idx][valid_bins])
                    / area_bin[valid_bins]
                )
                errors_plot[idx] = model_error

            if ref_profile is not None:
                valid_bins = ref_sum > bin_filter
                display_profiles[ref_idx] = np.asarray(ref_profile, dtype=np.float64)
                valid_bins_plot = valid_bins
                profiles_to_plot = [
                    np.asarray(p, dtype=np.float64)
                    if p is not None
                    else np.zeros_like(ref_profile, dtype=np.float64)
                    for p in display_profiles
                ]

        xv = x_vals[valid_bins_plot]

        # Reference (ground truth) drawn as black data points; models as
        # colourblind-safe lines with distinct linestyles. Legend ordered
        # reference-first.
        handles, handle_labels = [], []

        if ref_as_points:
            h = ax_prof.errorbar(
                xv,
                profiles_to_plot[ref_idx][valid_bins_plot],
                yerr=errors_plot[ref_idx][valid_bins_plot],
                fmt="o", ms=3.0, color="black",
                elinewidth=0.8, capsize=0, zorder=5,
            )
            handles.append(h)
            handle_labels.append(labels[ref_idx])
        model_indices = [k for k in range(n) if k != ref_idx or not ref_as_points]

        for slot, k in enumerate(model_indices):
            color = "black" if (k == ref_idx) else _CB_COLORS[slot % len(_CB_COLORS)]
            ls = _LINESTYLES[slot % len(_LINESTYLES)]
            (h,) = ax_prof.plot(
                xv, profiles_to_plot[k][valid_bins_plot],
                ls, color=color, lw=1.2, label=labels[k],
            )
            ax_prof.errorbar(
                xv,
                profiles_to_plot[k][valid_bins_plot],
                yerr=errors_plot[k][valid_bins_plot],
                fmt="none", ecolor=color,
                elinewidth=0.8, alpha=0.8, capsize=0,
            )
            handles.append(h)
            handle_labels.append(labels[k])

        ax_prof.set_xlabel(r"Radius [arcsec]", fontsize=font_size)
        ax_prof.set_ylabel(
            r"Surface brightness [photons arcsec$^{-2}$]",
            fontsize=font_size,
        )
        ax_prof.set_yscale("log")
        if log_x:
            ax_prof.set_xscale("log")
        else:
            ax_prof.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_prof.tick_params(axis="both", labelsize=font_size)
        ax_prof.legend(handles, handle_labels, fontsize=font_size, frameon=False)

        if x_lim is not None:
            ax_prof.set_xlim(x_lim)
        if y_lim is not None:
            ax_prof.set_ylim(y_lim)

        if show_chi2_label and ref_chi2_values:
            mean_ref = float(np.mean(ref_chi2_values))
            ax_prof.text(
                0.97, 0.95,
                rf"$\langle\chi^2_{{\mathrm{{red}}}}\rangle = {mean_ref:.2f}$",
                transform=ax_prof.transAxes,
                ha="right", va="top",
                fontsize=font_size,
            )

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            print(f"Saved figure to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return results
""" 
def _compare_matrices(
    matrices,
    labels,
    *,
    model_variances=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
    show_chi2_label: bool = True,
):

    from .plotting import normalize_and_log_scale
    from matplotlib.ticker import MaxNLocator  # type: ignore[import]
    plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.grid': False, 'legend.frameon': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
    n = len(matrices)
    if model_variances is None:
        model_variances = [None] * n
    elif len(model_variances) != n:
        raise ValueError(
            "model_variances must have the same length as matrices."
        )

    validated_variances = []
    for index, (matrix, variance) in enumerate(zip(matrices, model_variances)):
        if variance is None:
            validated_variances.append(None)
            continue

        variance = np.squeeze(np.asarray(variance, dtype=np.float64))
        if variance.shape != matrix.shape:
            raise ValueError(
                f"Variance shape mismatch at index {index}: "
                f"V={variance.shape}, matrix={matrix.shape}."
            )
        if not np.all(np.isfinite(variance)):
            raise ValueError(
                f"Variance matrix at index {index} contains NaN or infinite values."
            )
        if np.any(variance < 0.0):
            raise ValueError(
                f"Variance matrix at index {index} contains negative values."
            )
        validated_variances.append(variance)

    model_variances = validated_variances
    default_cmaps = ["magma", "viridis", "plasma", "cividis", "inferno", "cubehelix"]

    base_cmaps = default_cmaps if cmaps is None else list(cmaps)
    if len(base_cmaps) == 0:
        base_cmaps = default_cmaps
    cmaps = [base_cmaps[i % len(base_cmaps)] for i in range(n)]

    # --- Shared colour range ---
    shared_min = float(min(m.min() for m in matrices))
    shared_max = float(max(m.max() for m in matrices))

    # --- Figure layout ---
    if figsize is None:
        figsize = (5 * n, 10)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.3])

    img_axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
    ax_prof = fig.add_subplot(gs[1, :])

    # --- Top row: log-scaled images, matching plot_matrices_with_radial_overlay_log_three ---
    for ax, arr, label, cmap in zip(img_axes, matrices, labels, cmaps):
        scaled, _, _ = normalize_and_log_scale(
            arr,
            a,
            data_min=shared_min,
            data_max=shared_max,
        )
        ax.imshow(scaled, cmap=cmap, origin="lower")
        ax.set_title(label, fontsize=font_size)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Radial geometry (use last matrix as template, same as original) ---
    template = matrices[-1]
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
    area_per_pix = pixel_scale_arcsec ** 2
    area_bin = N_pixels * area_per_pix

    def _radial_sum(arr):
        return np.bincount(bins, weights=np.nan_to_num(arr.ravel()), minlength=n_annuli)

    sums = [_radial_sum(m) for m in matrices]
    variance_sums = [
        _radial_sum(variance) if variance is not None else sums[index]
        for index, variance in enumerate(model_variances)
    ]
    sb_profiles = [np.zeros(n_annuli, dtype=np.float64) for _ in matrices]
    for sb, s in zip(sb_profiles, sums):
        valid_sb = s > bin_filter
        sb[valid_sb] = s[valid_sb] / area_bin[valid_sb]

    # --- Pairwise metric computation ---
    tab10 = plt.colormaps.get_cmap("tab10")
    profile_colors = [tab10(i % tab10.N) for i in range(n)]
    results = {}

    ref_idx = n - 1
    ref_sum = sums[ref_idx]
    valid_ref = ref_sum > bin_filter

    # One-sigma surface-brightness errors for the reference data.
    err_ref = np.zeros(n_annuli, dtype=np.float64)
    err_ref[valid_ref] = (
        np.sqrt(Fano_Data * ref_sum[valid_ref])
        / area_bin[valid_ref]
    )

    helper_profiles = {}
    pairwise_red_contributions = []
    pairwise_mean_values = []
    ref_chi2_values = []  # chi2 with the reference (last / "data") matrix as data, model fit to it

    metric_label = "χ²_red"
    print(f"\nPairwise {metric_label} (chi2):")
    print(f"  n_annuli={n_annuli}, bin_filter={bin_filter}, Fano Data ={Fano_Data}, Fano Model ={Fano_Model}, pixel_scale={pixel_scale_arcsec}\"")
    print("-" * 55)
    need_pairwise_contributions = show_bin_contributions or show_mean_bin_contribution_plot
    if n == 1:
        print(f"  Single input matrix provided; pairwise {metric_label} skipped.")
    else:
        for (i, j) in combinations(range(n), 2):
            val_ij, contrib_ij, dof_ij, summary_str_ij, hdata_ij = _call_chi2_pair(
                matrices[i], matrices[j],
                V=model_variances[j],
                n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                bin_filter=bin_filter, need_contributions=need_pairwise_contributions,
                Fano_Data=Fano_Data, Fano_Model=Fano_Model,
            )
            val_ji, contrib_ji, dof_ji, summary_str_ji, hdata_ji = _call_chi2_pair(
                matrices[j], matrices[i],
                V=model_variances[i],
                n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                bin_filter=bin_filter, need_contributions=need_pairwise_contributions,
                Fano_Data=Fano_Data, Fano_Model=Fano_Model,
            )
            helper_profiles[(i, j)] = hdata_ij
            helper_profiles[(j, i)] = hdata_ji

            pair_mean = 0.5 * (val_ij + val_ji)
            pairwise_mean_values.append(pair_mean)
            if j == ref_idx:
                ref_chi2_values.append(val_ji)  # data=matrices[j] (reference), model=matrices[i] fit to it
            elif i == ref_idx:
                ref_chi2_values.append(val_ij)  # data=matrices[i] (reference), model=matrices[j] fit to it
            results[(i, j)] = (val_ij, val_ji)
            print(f"  {labels[i]} vs {labels[j]}")
            print(f"    {metric_label} ({labels[i]} as data) = {val_ij:.4f}   {summary_str_ij}")
            print(f"    {metric_label} ({labels[j]} as data) = {val_ji:.4f}   {summary_str_ji}")

            if need_pairwise_contributions:
                red_contrib_ij = contrib_ij / float(dof_ij)
                red_contrib_ji = contrib_ji / float(dof_ji)
                pairwise_red_contributions.append(red_contrib_ij)
                pairwise_red_contributions.append(red_contrib_ji)

                if show_bin_contributions:
                    label_i = str(labels[i])
                    label_j = str(labels[j])
                    print(f"    per-annulus reduced {metric_label} contributions:")
                    print(
                        "      bin | radius[arcsec] | area[arcsec^2] | "
                        f"counts({label_i}) | counts({label_j}) | "
                        f"{label_i}->{label_j} | {label_j}->{label_i}"
                    )
                    for b in range(n_annuli):
                        print(
                            f"      {b+1:>3d} | {bin_centers_arc[b]:>13.3f} | {area_bin[b]:>13.3f} | "
                            f"{float(sums[i][b]):>12.3f} | {float(sums[j][b]):>13.3f} | "
                            f"{red_contrib_ij[b]:>11.4e} | {red_contrib_ji[b]:>11.4e}"
                        )

        if pairwise_mean_values:
            mean_all = float(np.mean(pairwise_mean_values))
            std_all = float(np.std(pairwise_mean_values))
            print(f"  Mean {metric_label} (all pairs, both directions) = {mean_all:.4f} ± {std_all:.4f}")
        if ref_chi2_values:
            mean_ref = float(np.mean(ref_chi2_values))
            std_ref = float(np.std(ref_chi2_values))
            print(f"  Mean {metric_label} (model fit to reference '{labels[ref_idx]}' as data) = {mean_ref:.4f} ± {std_ref:.4f}")
        if results:
            print(f"  Pairwise {metric_label} summary (mean of both directions):")
            for (i, j), (v_ij, v_ji) in results.items():
                print(f"    {labels[i]} x {labels[j]} = {0.5 * (v_ij + v_ji):.4f}")
    print("-" * 55)

    # --- Radial profile plots (log-line + linear-bars) ---
    x_vals = bin_centers_arc
    valid_bins_plot = valid_ref
    profiles_to_plot = [np.asarray(p, dtype=np.float64) for p in sb_profiles]
    ref_errors_plot = np.asarray(err_ref, dtype=np.float64)
    errors_plot = [np.zeros(n_annuli, dtype=np.float64) for _ in range(n)]
    errors_plot[ref_idx] = ref_errors_plot

    if n >= 2:
        ref_profile = None
        display_profiles = [None] * n

        for idx in range(n - 1):
            helper_key = (ref_idx, idx)
            if helper_key not in helper_profiles:
                continue

            _, mu_ref, mu_other, n_best = helper_profiles[helper_key]
            if ref_profile is None:
                ref_profile = mu_ref

            model_profile = np.zeros_like(mu_ref, dtype=np.float64)
            valid_bins = ref_sum > bin_filter
            model_profile[valid_bins] = n_best * mu_other[valid_bins]
            display_profiles[idx] = model_profile

            # One-sigma model surface-brightness errors use the propagated
            # variance when supplied, or Poisson variance as a fallback.
            model_error = np.zeros_like(mu_ref, dtype=np.float64)
            model_error[valid_bins] = (
                n_best
                * np.sqrt(Fano_Model * variance_sums[idx][valid_bins])
                / area_bin[valid_bins]
            )
            errors_plot[idx] = model_error

        if ref_profile is not None:
            valid_bins = ref_sum > bin_filter
            display_profiles[ref_idx] = np.asarray(ref_profile, dtype=np.float64)
            valid_bins_plot = valid_bins
            profiles_to_plot = [
                np.asarray(p, dtype=np.float64) if p is not None else np.zeros_like(ref_profile, dtype=np.float64)
                for p in display_profiles
            ]

    xv = x_vals[valid_bins_plot]
    for profile, err, label, color in zip(profiles_to_plot, errors_plot, labels, profile_colors):
        ax_prof.plot(xv, profile[valid_bins_plot], "-", color=color, label=label)
        ax_prof.errorbar(
            xv,
            profile[valid_bins_plot],
            yerr=err[valid_bins_plot],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            alpha=0.8,
            capsize=0,
        )

    ax_prof.set_xlabel(r"Radius $\left[\mathrm{arcsec}\right]$", fontsize=font_size)
    ax_prof.set_ylabel(
        r"Surface Brightness $\left[\mathrm{photons}\,\mathrm{arcsec}^{-2}\right]$",
        fontsize=font_size,
    )
    ax_prof.set_yscale("log")
    ax_prof.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_prof.tick_params(axis="both", labelsize=font_size)
    ax_prof.legend(fontsize=font_size)
    ax_prof.grid(True, which="both", linestyle="--", alpha=0.3)

    if x_lim is not None:
        ax_prof.set_xlim(x_lim)
    if y_lim is not None:
        ax_prof.set_ylim(y_lim)

    if show_chi2_label and ref_chi2_values:
        mean_ref = float(np.mean(ref_chi2_values))
        ax_prof.text(
            0.5, -0.12,
            rf"Mean $\chi^2_{{\mathrm{{red}}}}$ (model fit to data) = {mean_ref:.4f}",
            transform=ax_prof.transAxes,
            ha="center", va="top",
            fontsize=font_size,
        )

    fig.tight_layout()
    plt.show()

    return results

 """