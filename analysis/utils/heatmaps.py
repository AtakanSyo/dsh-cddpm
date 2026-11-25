import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from .error_functions import radial_reduced_chi2_area2_bin_filtered, radial_reduced_poisson_deviance_abs

# --- shared journal styling helpers (module level) -----------------

JOURNAL_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _latex_sci(value: float, decimals: int = 2) -> str:
    """Format 1.08e21 -> $1.08\\times10^{21}$."""
    mantissa, exponent = f"{value:.{decimals}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Small statistical helpers.
# --------------------------------------------------------------------------
def _trimmed_mean(values: np.ndarray, frac: float) -> float:
    """Symmetrically trimmed mean; robust to the few large outliers that
    dominate the arithmetic mean of a chi^2 grid."""
    if frac <= 0.0:
        return float(np.mean(values))

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    k = int(np.floor(ordered.size * frac))

    if ordered.size - 2 * k <= 0:
        return float("nan")

    return float(ordered[k: ordered.size - k].mean())


def _resolve_colour_limits(
    E,
    finite,
    *,
    vmin,
    vmax,
    diverging,
    log_norm,
    vcenter,
):
    """Resolve the vmin/vmax that the figure will actually use.

    Resolved up front (rather than inside the plotting block) so that the
    printed summary can report exactly which cells the colour scale clips.
    """
    if log_norm:
        positive = finite & (E > 0.0)
        reference = E[positive] if positive.any() else E[finite]
    else:
        reference = E[finite]

    if diverging:
        res_vmin = (
            float(np.nanmin(reference)) if vmin is None else float(vmin)
        )
        res_vmax = (
            float(np.nanpercentile(reference, 99))
            if vmax is None
            else float(vmax)
        )

        # TwoSlopeNorm requires vmin < vcenter < vmax.
        if res_vmin >= vcenter:
            res_vmin = vcenter - (res_vmax - vcenter) * 0.05
        if res_vmax <= vcenter:
            res_vmax = vcenter + (vcenter - res_vmin) * 0.05

    elif log_norm:
        res_vmin = (
            float(np.nanmin(reference)) if vmin is None else float(vmin)
        )
        res_vmax = (
            float(np.nanmax(reference)) if vmax is None else float(vmax)
        )

    else:
        res_vmin = (
            float(np.nanmin(reference)) if vmin is None else float(vmin)
        )
        res_vmax = (
            float(np.nanpercentile(reference, 99))
            if vmax is None
            else float(vmax)
        )

    return res_vmin, res_vmax


def error_heatmap_between_npz(
    npz_model_path: str,
    npz_data_path: str,
    *,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    use_N: bool = True,
    decimals: int = 6,
    abs_tol: float = 1e-6,
    cmap: str = "magma",
    diverging: bool = False,
    vcenter: float = 1.0,
    title: str = "",
    log_norm: bool = False,
    tick_stride: int = 1,
    figsize=(7, 5),
    show: bool = True,
    vmin=None,
    vmax=None,
    annotate_above: float | None = None,
    annotate_fontsize: float = 13.0,
    x_display_scale: float = 1e-4,
    over_color: str | None = None,
    under_color: str | None = None,
    consistency_levels=(1.0, 2.0, 3.0),
    print_stats: bool = True,
    print_row_means: bool = True,
    print_col_means: bool = True,
    print_clipped: bool = True,
    max_clipped_listed: int = 20,
    return_stats: bool = False,
):
    """
    Compute radial reduced chi-squared values between matching (x, nH)
    matrices in two NPZ archives and optionally plot the resulting heatmap.

    If the model NPZ contains a ``Vs`` array, the corresponding propagated
    model-variance image is passed to
    ``radial_reduced_chi2_area2_bin_filtered`` as ``V``.

    ``use_N`` controls whether a multiplicative model normalization ``N`` is
    fitted per (x, nH) cell before computing chi-squared. When ``True``
    (default), ``N`` is fit by minimizing chi2(N) and one degree of freedom
    is subtracted for it. When ``False``, ``N`` is fixed to 1 (no parameter
    fitted, no dof penalty) -- appropriate when the data and model already
    share the same absolute photon normalization.

    Notes
    -----
    Cells whose value exceeds the resolved upper colour limit are clipped by
    the colour map. The colour bar is drawn with the matching ``extend``
    arrow, and the summary block lists every clipped cell so that the figure
    caption can state the convention explicitly.

    ``annotate_above`` prints the numerical value inside every cell above the
    given threshold, at ``annotate_fontsize`` points. Setting the threshold to
    the resolved upper colour limit labels exactly the clipped cells.

    ``x_display_scale`` converts stored x coordinates into the values shown
    on the axis and in the printed summary (stored 648 -> displayed 0.0648
    with the default of 1e-4). Printed and plotted coordinates therefore
    always agree.

    Returns
    -------
    E : ndarray
        Error grid with shape (len(nH_vals), len(x_vals)).
    x_vals : ndarray
        Sorted shared x coordinates (stored units, not display units).
    nH_vals : ndarray
        Sorted shared nH coordinates.
    stats : dict, optional
        Returned only when ``return_stats=True``.
    """
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")

    if abs_tol < 0.0 or not np.isfinite(abs_tol):
        raise ValueError(
            "abs_tol must be finite and non-negative."
        )

    if tick_stride < 1:
        raise ValueError(
            "tick_stride must be a positive integer."
        )

    cbar_label = r"$\chi^2_{\mathrm{red}}$"

    # --------------------------------------------------------------
    # Load model archive.
    # --------------------------------------------------------------
    with np.load(npz_model_path) as model_npz:
        required_keys = {"xs", "nHs", "mats"}
        missing_keys = required_keys.difference(
            model_npz.files
        )

        if missing_keys:
            raise KeyError(
                f"Model NPZ is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        xs_m_raw = np.asarray(
            model_npz["xs"],
            dtype=np.float64,
        ).reshape(-1)

        nHs_m_raw = np.asarray(
            model_npz["nHs"],
            dtype=np.float64,
        ).reshape(-1)

        mats_m = np.asarray(model_npz["mats"])

        Vs_m = (
            np.asarray(model_npz["Vs"])
            if "Vs" in model_npz.files
            else None
        )

    # --------------------------------------------------------------
    # Load data archive.
    # --------------------------------------------------------------
    with np.load(npz_data_path) as data_npz:
        required_keys = {"xs", "nHs", "mats"}
        missing_keys = required_keys.difference(
            data_npz.files
        )

        if missing_keys:
            raise KeyError(
                f"Data NPZ is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        xs_d_raw = np.asarray(
            data_npz["xs"],
            dtype=np.float64,
        ).reshape(-1)

        nHs_d_raw = np.asarray(
            data_npz["nHs"],
            dtype=np.float64,
        ).reshape(-1)

        mats_d = np.asarray(data_npz["mats"])

    # --------------------------------------------------------------
    # Validate archive lengths.
    # --------------------------------------------------------------
    if not (
        len(xs_m_raw)
        == len(nHs_m_raw)
        == len(mats_m)
    ):
        raise ValueError(
            "Model arrays xs, nHs, and mats must have "
            "equal lengths."
        )

    if not (
        len(xs_d_raw)
        == len(nHs_d_raw)
        == len(mats_d)
    ):
        raise ValueError(
            "Data arrays xs, nHs, and mats must have "
            "equal lengths."
        )

    if Vs_m is not None and len(Vs_m) != len(mats_m):
        raise ValueError(
            "Model Vs and mats must have equal lengths."
        )

    for name, values in (
        ("model xs", xs_m_raw),
        ("model nHs", nHs_m_raw),
        ("data xs", xs_d_raw),
        ("data nHs", nHs_d_raw),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} contain NaN or infinite values."
            )

    xs_m = np.round(xs_m_raw, decimals)
    nHs_m = np.round(nHs_m_raw, decimals)
    xs_d = np.round(xs_d_raw, decimals)
    nHs_d = np.round(nHs_d_raw, decimals)

    # --------------------------------------------------------------
    # Build data lookup.
    #
    # Keep the raw coordinates so that rounded matches can be checked
    # using abs_tol.
    # --------------------------------------------------------------
    data_map = {}

    for x_raw, nH_raw, x_key, nH_key, mat in zip(
        xs_d_raw,
        nHs_d_raw,
        xs_d,
        nHs_d,
        mats_d,
    ):
        key = (
            float(x_key),
            float(nH_key),
        )

        if key in data_map:
            raise ValueError(
                "Duplicate data coordinate after rounding: "
                f"x={key[0]}, nH={key[1]}."
            )

        mat_data = np.squeeze(
            np.asarray(mat, dtype=np.float64)
        )

        if mat_data.ndim != 2:
            raise ValueError(
                f"Data matrix at {key} must be 2-D after "
                f"squeezing; got shape {mat_data.shape}."
            )

        data_map[key] = (
            float(x_raw),
            float(nH_raw),
            mat_data,
        )

    # Reject duplicate model coordinates after rounding.
    model_keys = [
        (float(x), float(nH))
        for x, nH in zip(xs_m, nHs_m)
    ]

    if len(model_keys) != len(set(model_keys)):
        raise ValueError(
            "Duplicate model coordinates found after rounding."
        )

    shared_pairs = set(model_keys) & set(data_map.keys())

    if not shared_pairs:
        raise RuntimeError(
            "No overlapping (x, nH) pairs between the "
            "two NPZ files."
        )

    x_vals = np.array(
        sorted({
            pair[0]
            for pair in shared_pairs
        }),
        dtype=np.float64,
    )

    nH_vals = np.array(
        sorted({
            pair[1]
            for pair in shared_pairs
        }),
        dtype=np.float64,
    )

    ix = {
        value: index
        for index, value in enumerate(x_vals)
    }

    inh = {
        value: index
        for index, value in enumerate(nH_vals)
    }

    E = np.full(
        (len(nH_vals), len(x_vals)),
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Compute errors.
    # --------------------------------------------------------------
    for idx_m, (
        x_m_raw,
        nH_m_raw,
        x_m,
        nH_m,
        mat_m,
    ) in enumerate(
        zip(
            xs_m_raw,
            nHs_m_raw,
            xs_m,
            nHs_m,
            mats_m,
        )
    ):
        key = (
            float(x_m),
            float(nH_m),
        )

        if key not in data_map:
            continue

        x_d_raw, nH_d_raw, mat_data = data_map[key]

        # Verify that matching after rounding did not pair genuinely
        # different coordinates.
        if not np.isclose(
            x_m_raw,
            x_d_raw,
            rtol=0.0,
            atol=abs_tol,
        ):
            raise ValueError(
                "Rounded x coordinates matched, but raw values "
                f"differ: model={x_m_raw:.12g}, "
                f"data={x_d_raw:.12g}."
            )

        # A purely absolute tolerance such as 1e-6 is inappropriate
        # for nH values near 1e21, so use abs_tol as a relative scale
        # as well.
        if not np.isclose(
            nH_m_raw,
            nH_d_raw,
            rtol=abs_tol,
            atol=abs_tol,
        ):
            raise ValueError(
                "Rounded nH coordinates matched, but raw values "
                f"differ: model={nH_m_raw:.12g}, "
                f"data={nH_d_raw:.12g}."
            )

        mat_model = np.squeeze(
            np.asarray(mat_m, dtype=np.float64)
        )

        if mat_model.ndim != 2:
            raise ValueError(
                f"Model matrix at {key} must be 2-D after "
                f"squeezing; got shape {mat_model.shape}."
            )

        if mat_model.shape != mat_data.shape:
            raise ValueError(
                f"Shape mismatch for "
                f"(x={key[0]}, nH={key[1]}): "
                f"model={mat_model.shape}, "
                f"data={mat_data.shape}."
            )

        variance_matrix = None

        if Vs_m is not None:

            variance_matrix = np.squeeze(
                np.asarray(
                    Vs_m[idx_m],
                    dtype=np.float64,
                )
            )

            if variance_matrix.shape != mat_model.shape:
                raise ValueError(
                    f"Variance shape mismatch for "
                    f"(x={key[0]}, nH={key[1]}): "
                    f"V={variance_matrix.shape}, "
                    f"model={mat_model.shape}."
                )

            if not np.all(np.isfinite(variance_matrix)):
                raise ValueError(
                    f"Variance matrix at {key} contains "
                    "NaN or infinite values."
                )

            if np.any(variance_matrix < 0.0):
                raise ValueError(
                    f"Variance matrix at {key} contains "
                    "negative values."
                )

        err = radial_reduced_chi2_area2_bin_filtered(
            data=mat_data,
            model=mat_model,
            V=variance_matrix,
            n_annuli=n_annuli,
            pixel_scale_arcsec=pixel_scale_arcsec,
            bin_filter=bin_filter,
            Fano_Data=Fano_Data,
            Fano_Model=Fano_Model,
            use_N=use_N,
        )
        err = float(err)

        E[
            inh[key[1]],
            ix[key[0]],
        ] = err

    finite = np.isfinite(E)
    n_total = int(E.size)
    n_finite = int(finite.sum())

    stats: dict = {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_total - n_finite,
        "grid_shape": (len(nH_vals), len(x_vals)),
    }

    # --------------------------------------------------------------
    # Resolve colour limits before printing, so that the summary can
    # report the cells the figure will clip.
    # --------------------------------------------------------------
    res_vmin = res_vmax = None
    extend = "neither"
    clipped_above_idx = np.empty((0, 2), dtype=int)
    clipped_below_idx = np.empty((0, 2), dtype=int)

    if n_finite:
        res_vmin, res_vmax = _resolve_colour_limits(
            E,
            finite,
            vmin=vmin,
            vmax=vmax,
            diverging=diverging,
            log_norm=log_norm,
            vcenter=vcenter,
        )

        tiny = 1e-12
        clipped_above_idx = np.argwhere(
            finite & (E > res_vmax + tiny)
        )
        clipped_below_idx = np.argwhere(
            finite & (E < res_vmin - tiny)
        )

        has_above = clipped_above_idx.shape[0] > 0
        has_below = clipped_below_idx.shape[0] > 0

        if has_above and has_below:
            extend = "both"
        elif has_above:
            extend = "max"
        elif has_below:
            extend = "min"

        stats.update(
            {
                "vmin_used": res_vmin,
                "vmax_used": res_vmax,
                "extend": extend,
                "n_clipped_above": int(clipped_above_idx.shape[0]),
                "n_clipped_below": int(clipped_below_idx.shape[0]),
            }
        )

    # --------------------------------------------------------------
    # Summary statistics.
    # --------------------------------------------------------------
    def _fmt_x(stored_value: float) -> str:
        return f"{stored_value * x_display_scale:.4f}"

    if n_finite:
        v = E[finite]

        mean_err = float(v.mean())
        std_err = (
            float(v.std(ddof=1)) if n_finite > 1 else float("nan")
        )
        sem_err = (
            std_err / np.sqrt(n_finite)
            if n_finite > 1
            else float("nan")
        )
        median_err = float(np.median(v))

        q25, q75, q90, q95, q99 = (
            float(np.percentile(v, p))
            for p in (25, 75, 90, 95, 99)
        )

        min_err = float(v.min())
        max_err = float(v.max())

        trimmed10 = _trimmed_mean(v, 0.10)

        # Geometric mean is meaningful for a ratio-like statistic and is
        # far less outlier-driven than the arithmetic mean.
        positive_vals = v[v > 0.0]
        geo_mean = (
            float(np.exp(np.mean(np.log(positive_vals))))
            if positive_vals.size
            else float("nan")
        )

        fractions = {}
        for level in consistency_levels:
            count = int(np.count_nonzero(v <= level))
            fractions[float(level)] = (count, count / n_finite)

        stats.update(
            {
                "mean": mean_err,
                "std": std_err,
                "sem": sem_err,
                "median": median_err,
                "p25": q25,
                "p75": q75,
                "p90": q90,
                "p95": q95,
                "p99": q99,
                "min": min_err,
                "max": max_err,
                "trimmed_mean_10": trimmed10,
                "geometric_mean": geo_mean,
                "fraction_below": fractions,
            }
        )

        finite_positions = np.argwhere(finite)

        argmin = int(np.argmin(v))
        argmax = int(np.argmax(v))
        argmean = int(np.argmin(np.abs(v - mean_err)))
        argmedian = int(np.argmin(np.abs(v - median_err)))

        def _pos_to_coords(pos_idx):
            row, col = finite_positions[pos_idx]
            return (
                float(x_vals[col]),
                float(nH_vals[row]),
            )

        min_x, min_nH = _pos_to_coords(argmin)
        max_x, max_nH = _pos_to_coords(argmax)
        mean_x, mean_nH = _pos_to_coords(argmean)
        med_x, med_nH = _pos_to_coords(argmedian)

        stats.update(
            {
                "argmin_coord": (min_x, min_nH),
                "argmax_coord": (max_x, max_nH),
                "closest_to_mean_coord": (mean_x, mean_nH),
                "closest_to_median_coord": (med_x, med_nH),
            }
        )

        # Row (nH) and column (x) aggregates.
        row_means, row_medians, row_counts = [], [], []
        for row_index in range(len(nH_vals)):
            row = E[row_index]
            mask = np.isfinite(row)
            n_row = int(np.count_nonzero(mask))
            row_counts.append(n_row)
            row_means.append(
                float(np.mean(row[mask])) if n_row else float("nan")
            )
            row_medians.append(
                float(np.median(row[mask])) if n_row else float("nan")
            )

        col_means, col_medians, col_counts = [], [], []
        for col_index in range(len(x_vals)):
            col = E[:, col_index]
            mask = np.isfinite(col)
            n_col = int(np.count_nonzero(mask))
            col_counts.append(n_col)
            col_means.append(
                float(np.mean(col[mask])) if n_col else float("nan")
            )
            col_medians.append(
                float(np.median(col[mask])) if n_col else float("nan")
            )

        stats.update(
            {
                "row_means": np.array(row_means),
                "row_medians": np.array(row_medians),
                "col_means": np.array(col_means),
                "col_medians": np.array(col_medians),
                "x_vals": x_vals,
                "nH_vals": nH_vals,
                "E": E,
            }
        )

        clipped_records = [
            (
                float(x_vals[col]),
                float(nH_vals[row]),
                float(E[row, col]),
            )
            for row, col in clipped_above_idx
        ]
        clipped_records.sort(key=lambda rec: rec[2], reverse=True)
        stats["clipped_above"] = clipped_records

        if print_stats:
            width = 66
            header = title if title else "error grid"

            print("=" * width)
            print(f" Summary: {header}")
            print("=" * width)
            print(
                f" Grid                : "
                f"{len(nH_vals)} x {len(x_vals)}  "
                f"({n_total} cells, {n_finite} finite, "
                f"{n_total - n_finite} missing)"
            )
            print(f" Mean                : {mean_err:.4f}")
            print(
                f" Std (ddof=1)        : {std_err:.4f}"
                f"     SEM: {sem_err:.4f}"
            )
            print(f" Median              : {median_err:.4f}")
            print(f" Trimmed mean (10%)  : {trimmed10:.4f}")
            print(f" Geometric mean      : {geo_mean:.4f}")
            print(f" IQR                 : {q25:.4f} - {q75:.4f}")
            print(
                f" Percentiles         : "
                f"p90={q90:.4f}  p95={q95:.4f}  p99={q99:.4f}"
            )
            print(f" Min / Max           : {min_err:.4f} / {max_err:.4f}")

            for level in consistency_levels:
                count, frac = fractions[float(level)]
                print(
                    f" Fraction <= {level:<5.2f}   : "
                    f"{100.0 * frac:6.2f}%  ({count}/{n_finite})"
                )

            print("-" * width)
            print(" Extremes")
            print(
                f"   Lowest   {min_err:10.4f}  at "
                f"x={_fmt_x(min_x)}, nH={min_nH:.3e}"
            )
            print(
                f"   Highest  {max_err:10.4f}  at "
                f"x={_fmt_x(max_x)}, nH={max_nH:.3e}"
            )
            print(
                f"   ~Mean    {v[argmean]:10.4f}  at "
                f"x={_fmt_x(mean_x)}, nH={mean_nH:.3e}"
            )
            print(
                f"   ~Median  {v[argmedian]:10.4f}  at "
                f"x={_fmt_x(med_x)}, nH={med_nH:.3e}"
            )

            # ----- colour-scale clipping -----
            if print_clipped:
                print("-" * width)
                print(
                    f" Colour scale        : "
                    f"vmin={res_vmin:.4g}, vmax={res_vmax:.4g}, "
                    f"extend='{extend}'"
                )

                n_above = len(clipped_records)
                if n_above:
                    print(
                        f" Clipped above vmax  : {n_above}/{n_finite} "
                        f"cells ({100.0 * n_above / n_finite:.1f}%)"
                    )
                    for x_c, nH_c, val in clipped_records[
                        :max_clipped_listed
                    ]:
                        print(
                            f"     x={_fmt_x(x_c)}  "
                            f"nH={nH_c:.3e}   value={val:.2f}"
                        )
                    if n_above > max_clipped_listed:
                        print(
                            f"     ... and "
                            f"{n_above - max_clipped_listed} more"
                        )

                    print()
                    print(" Suggested caption sentence:")
                    print(
                        f"   Cells with $\\chi^2_{{\\rm red}} > "
                        f"{res_vmax:g}$ exceed the colour-scale limit "
                        f"and are clipped to"
                    )
                    print(
                        "   the darkest shade; their numerical values "
                        "are printed in the corresponding"
                    )
                    print("   cell.")
                else:
                    print(
                        " Clipped above vmax  : none "
                        "(no clipping note needed in the caption)"
                    )

                n_below = int(clipped_below_idx.shape[0])
                if n_below:
                    print(
                        f" Clipped below vmin  : {n_below} cells"
                    )

            if print_row_means:
                print("-" * width)
                print(" Per-nH (row) aggregates")
                print(
                    f"   {'nH':<12} {'mean':>10} {'median':>10} "
                    f"{'max':>10} {'n':>5}"
                )
                for row_index, nH_val in enumerate(nH_vals):
                    row = E[row_index]
                    mask = np.isfinite(row)
                    row_max = (
                        float(np.max(row[mask]))
                        if row_counts[row_index]
                        else float("nan")
                    )
                    print(
                        f"   {nH_val:<12.3e} "
                        f"{row_means[row_index]:>10.4f} "
                        f"{row_medians[row_index]:>10.4f} "
                        f"{row_max:>10.4f} "
                        f"{row_counts[row_index]:>5d}"
                    )

            if print_col_means:
                print("-" * width)
                print(" Per-x (column) aggregates")
                print(
                    f"   {'x':<12} {'mean':>10} {'median':>10} "
                    f"{'max':>10} {'n':>5}"
                )
                for col_index, x_val in enumerate(x_vals):
                    col = E[:, col_index]
                    mask = np.isfinite(col)
                    col_max = (
                        float(np.max(col[mask]))
                        if col_counts[col_index]
                        else float("nan")
                    )
                    print(
                        f"   {_fmt_x(x_val):<12} "
                        f"{col_means[col_index]:>10.4f} "
                        f"{col_medians[col_index]:>10.4f} "
                        f"{col_max:>10.4f} "
                        f"{col_counts[col_index]:>5d}"
                    )

            print("=" * width)

    else:
        if print_stats:
            print("No finite errors; all summary statistics are NaN.")

    # --------------------------------------------------------------
    # Plot heatmap.
    # --------------------------------------------------------------
    if show:
        if not finite.any():
            raise RuntimeError(
                "No finite errors to plot."
            )

        with plt.rc_context(JOURNAL_RC):
            fig, ax_err = plt.subplots(figsize=figsize)

            if diverging:
                from matplotlib.colors import TwoSlopeNorm

                if log_norm:
                    raise ValueError(
                        "diverging=True and log_norm=True are "
                        "mutually exclusive."
                    )

                used_cmap = "RdBu_r" if cmap == "magma" else cmap
            else:
                used_cmap = cmap

            # A copy is needed so that set_over / set_under do not mutate
            # the globally registered colour map.
            cmap_obj = plt.get_cmap(used_cmap).copy()

            if over_color is not None:
                cmap_obj.set_over(over_color)
            if under_color is not None:
                cmap_obj.set_under(under_color)

            if diverging:
                norm = TwoSlopeNorm(
                    vmin=res_vmin,
                    vcenter=vcenter,
                    vmax=res_vmax,
                )

                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=norm,
                    interpolation="nearest",
                )

            elif log_norm:
                from matplotlib.colors import LogNorm

                positive = finite & (E > 0.0)

                if not positive.any():
                    raise RuntimeError(
                        "Logarithmic normalization requires at "
                        "least one positive finite error."
                    )

                E_plot = np.where(positive, E, np.nan)

                if res_vmin <= 0.0:
                    raise ValueError(
                        "vmin must be strictly positive when "
                        "log_norm=True."
                    )
                if res_vmax <= res_vmin:
                    raise ValueError(
                        "vmax must be greater than vmin."
                    )

                im_err = ax_err.imshow(
                    E_plot,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=LogNorm(vmin=res_vmin, vmax=res_vmax),
                    interpolation="nearest",
                )

            else:
                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    vmin=res_vmin,
                    vmax=res_vmax,
                    interpolation="nearest",
                )

            if annotate_above is not None:
                for i, j in np.argwhere(
                    finite & (E > annotate_above)
                ):
                    ax_err.text(
                        j,
                        i,
                        f"{E[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=annotate_fontsize,
                        fontweight="bold",
                    )

            xticks = np.arange(len(x_vals))
            yticks = np.arange(len(nH_vals))

            x_labels = [
                f"{(value * x_display_scale):.4f}"
                for value in x_vals[::tick_stride]
            ]
            nh_labels = [
                _latex_sci(value)
                for value in nH_vals[::tick_stride]
            ]

            ax_err.set_xticks(xticks[::tick_stride])
            ax_err.set_xticklabels(
                x_labels, rotation=45, ha="right", fontsize=10
            )
            ax_err.set_yticks(yticks[::tick_stride])
            ax_err.set_yticklabels(nh_labels, fontsize=10)

            ax_err.set_xlabel(r"$x$")
            ax_err.set_ylabel(r"$N_\mathrm{H}$ [cm$^{-2}$]")

            # Categorical tick positions: keep frame, drop tick marks
            ax_err.tick_params(axis="both", which="both", length=0)

            if title:
                ax_err.set_title(title, pad=8)

            # ---- single colorbar, AFTER im_err exists ----
            cbar_err = fig.colorbar(
                im_err,
                ax=ax_err,
                pad=0.02,
                extend=extend,
            )
            cbar_err.set_label(cbar_label)
            cbar_err.ax.tick_params(direction="in", length=3)
            cbar_err.outline.set_linewidth(0.8)

            if diverging:
                ticks = cbar_err.get_ticks()
                if vcenter not in ticks:
                    ticks = np.sort(np.append(ticks, vcenter))
                cbar_err.set_ticks(
                    ticks[
                        (ticks >= res_vmin)
                        & (ticks <= res_vmax)
                    ]
                )

            plt.tight_layout()
            plt.show()

    if return_stats:
        return E, x_vals, nH_vals, stats

    return E, x_vals, nH_vals


def poisson_deviance_heatmap_between_npz(
    npz_model_path: str,
    npz_data_path: str,
    *,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    decimals: int = 6,
    abs_tol: float = 1e-6,
    use_N: bool = False,
    cmap: str = "magma",
    diverging: bool = False,
    vcenter: float = 1.0,
    title: str = "",
    log_norm: bool = False,
    tick_stride: int = 1,
    figsize=(7, 5),
    show: bool = True,
    vmin=None,
    vmax=None,
    annotate_above: float | None = None,
    annotate_fontsize: float = 13.0,
    x_display_scale: float = 1e-4,
    over_color: str | None = None,
    under_color: str | None = None,
    consistency_levels=(1.0, 2.0, 3.0),
    print_stats: bool = True,
    print_row_means: bool = True,
    print_col_means: bool = True,
    print_clipped: bool = True,
    max_clipped_listed: int = 20,
    return_stats: bool = False,
):
    """
    Compute radial reduced two-sample Poisson deviance values between
    matching (x, nH) images in two NPZ archives and optionally plot the
    resulting heatmap.

    For each matching pair, the RefleX image is passed as ``data`` and
    the generated image is passed as ``model`` to

        radial_reduced_poisson_deviance_abs

    Both images are reduced to radial annular count profiles.

    For annulus i,

        R_i = RefleX counts,
        G_i = generated counts,

    and the common expected count under the null hypothesis is

        lambda_hat_i = (R_i + G_i) / 2.

    The absolute two-sample Poisson deviance is

        D = 2 * sum_i [
            R_i * log(R_i / lambda_hat_i)
            +
            G_i * log(G_i / lambda_hat_i)
        ].

    Since no multiplicative normalization parameter is fitted,

        dof = K,

    where K is the number of radial annuli with positive RefleX/data
    counts, and

        D_red = D / K.

    This is therefore an absolute comparison: both total photon yield
    and radial morphology contribute to the discrepancy.

    Unlike the previous chi-squared heatmap:

    - no minimum-count bin filter is applied;
    - annuli with zero RefleX/data counts are excluded;
    - no Fano factors are used;
    - no variance image V is used;
    - no multiplicative normalization N is fitted.

    Notes
    -----
    If the model NPZ contains a ``Vs`` array, it is ignored. Poisson
    deviance is calculated directly from the RefleX and generated
    annular counts.

    Cells whose value exceeds the resolved upper colour limit are clipped
    by the colour map. The colour bar is drawn with the corresponding
    ``extend`` arrow, and the summary block lists clipped cells.

    ``annotate_above`` prints the numerical value inside every cell above
    the given threshold.

    ``x_display_scale`` converts stored x coordinates into the values
    displayed on the axis and in the printed summary.

    Parameters
    ----------
    npz_model_path : str
        Path to the NPZ archive containing generated images.

    npz_data_path : str
        Path to the NPZ archive containing RefleX images.

    n_annuli : int
        Number of radial annuli used for the Poisson-deviance comparison.

    pixel_scale_arcsec : float
        Pixel width in arcseconds. This is passed to the radial deviance
        function. The deviance itself is evaluated in count space.

    decimals : int
        Number of decimal places used when matching stored coordinates.

    abs_tol : float
        Tolerance used to verify that coordinates matched after rounding
        are genuinely consistent.

    use_N : bool
        Forwarded to ``radial_reduced_poisson_deviance_abs``. If True, a
        multiplicative normalization N is fitted per cell (RefleX scaled
        relative to the generated image) before computing the deviance,
        and one degree of freedom is subtracted for it. If False
        (default), N is fixed at 1 -- the original absolute, unnormalized
        comparison.

    consistency_levels : sequence of float
        Thresholds for reporting the fraction of cells satisfying

            D_red <= level.

    Returns
    -------
    E : ndarray
        Reduced Poisson-deviance grid with shape

            (len(nH_vals), len(x_vals)).

    x_vals : ndarray
        Sorted shared x coordinates in stored units.

    nH_vals : ndarray
        Sorted shared nH coordinates.

    stats : dict, optional
        Returned only when ``return_stats=True``.
    """

    # --------------------------------------------------------------
    # 1. Validate scalar options.
    # --------------------------------------------------------------
    if decimals < 0:
        raise ValueError(
            "decimals must be non-negative."
        )

    if abs_tol < 0.0 or not np.isfinite(abs_tol):
        raise ValueError(
            "abs_tol must be finite and non-negative."
        )

    if tick_stride < 1:
        raise ValueError(
            "tick_stride must be a positive integer."
        )

    if n_annuli <= 0:
        raise ValueError(
            "n_annuli must be a positive integer."
        )

    if pixel_scale_arcsec <= 0.0:
        raise ValueError(
            "pixel_scale_arcsec must be positive."
        )

    cbar_label = (
        r"$D_{\mathrm{P,shape}}$" if use_N else r"$D_{\mathrm{P}}$"
    )

    # --------------------------------------------------------------
    # 2. Load model archive.
    # --------------------------------------------------------------
    with np.load(npz_model_path) as model_npz:

        required_keys = {
            "xs",
            "nHs",
            "mats",
        }

        missing_keys = required_keys.difference(
            model_npz.files
        )

        if missing_keys:
            raise KeyError(
                f"Model NPZ is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        xs_m_raw = np.asarray(
            model_npz["xs"],
            dtype=np.float64,
        ).reshape(-1)

        nHs_m_raw = np.asarray(
            model_npz["nHs"],
            dtype=np.float64,
        ).reshape(-1)

        mats_m = np.asarray(
            model_npz["mats"]
        )

    # --------------------------------------------------------------
    # 3. Load RefleX data archive.
    # --------------------------------------------------------------
    with np.load(npz_data_path) as data_npz:

        required_keys = {
            "xs",
            "nHs",
            "mats",
        }

        missing_keys = required_keys.difference(
            data_npz.files
        )

        if missing_keys:
            raise KeyError(
                f"Data NPZ is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        xs_d_raw = np.asarray(
            data_npz["xs"],
            dtype=np.float64,
        ).reshape(-1)

        nHs_d_raw = np.asarray(
            data_npz["nHs"],
            dtype=np.float64,
        ).reshape(-1)

        mats_d = np.asarray(
            data_npz["mats"]
        )

    # --------------------------------------------------------------
    # 4. Validate archive lengths.
    # --------------------------------------------------------------
    if not (
        len(xs_m_raw)
        == len(nHs_m_raw)
        == len(mats_m)
    ):
        raise ValueError(
            "Model arrays xs, nHs, and mats must have "
            "equal lengths."
        )

    if not (
        len(xs_d_raw)
        == len(nHs_d_raw)
        == len(mats_d)
    ):
        raise ValueError(
            "Data arrays xs, nHs, and mats must have "
            "equal lengths."
        )

    for name, values in (
        ("model xs", xs_m_raw),
        ("model nHs", nHs_m_raw),
        ("data xs", xs_d_raw),
        ("data nHs", nHs_d_raw),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} contain NaN or infinite values."
            )

    # --------------------------------------------------------------
    # 5. Round coordinates for matching.
    # --------------------------------------------------------------
    xs_m = np.round(
        xs_m_raw,
        decimals,
    )

    nHs_m = np.round(
        nHs_m_raw,
        decimals,
    )

    xs_d = np.round(
        xs_d_raw,
        decimals,
    )

    nHs_d = np.round(
        nHs_d_raw,
        decimals,
    )

    # --------------------------------------------------------------
    # 6. Build RefleX lookup table.
    #
    #    Raw coordinates are retained so that rounded matches can be
    #    checked against abs_tol.
    # --------------------------------------------------------------
    data_map = {}

    for (
        x_raw,
        nH_raw,
        x_key,
        nH_key,
        mat,
    ) in zip(
        xs_d_raw,
        nHs_d_raw,
        xs_d,
        nHs_d,
        mats_d,
    ):

        key = (
            float(x_key),
            float(nH_key),
        )

        if key in data_map:
            raise ValueError(
                "Duplicate data coordinate after rounding: "
                f"x={key[0]}, nH={key[1]}."
            )

        mat_data = np.squeeze(
            np.asarray(
                mat,
                dtype=np.float64,
            )
        )

        if mat_data.ndim != 2:
            raise ValueError(
                f"Data matrix at {key} must be 2-D after "
                f"squeezing; got shape {mat_data.shape}."
            )

        if not np.all(np.isfinite(mat_data)):
            raise ValueError(
                f"Data matrix at {key} contains NaN or "
                "infinite values."
            )

        if np.any(mat_data < 0.0):
            raise ValueError(
                f"Data matrix at {key} contains negative counts."
            )

        data_map[key] = (
            float(x_raw),
            float(nH_raw),
            mat_data,
        )

    # --------------------------------------------------------------
    # 7. Reject duplicate model coordinates.
    # --------------------------------------------------------------
    model_keys = [
        (
            float(x),
            float(nH),
        )
        for x, nH in zip(
            xs_m,
            nHs_m,
        )
    ]

    if len(model_keys) != len(set(model_keys)):
        raise ValueError(
            "Duplicate model coordinates found after rounding."
        )

    # --------------------------------------------------------------
    # 8. Find shared parameter-space points.
    # --------------------------------------------------------------
    shared_pairs = (
        set(model_keys)
        & set(data_map.keys())
    )

    if not shared_pairs:
        raise RuntimeError(
            "No overlapping (x, nH) pairs between the "
            "two NPZ files."
        )

    x_vals = np.array(
        sorted(
            {
                pair[0]
                for pair in shared_pairs
            }
        ),
        dtype=np.float64,
    )

    nH_vals = np.array(
        sorted(
            {
                pair[1]
                for pair in shared_pairs
            }
        ),
        dtype=np.float64,
    )

    ix = {
        value: index
        for index, value in enumerate(
            x_vals
        )
    }

    inh = {
        value: index
        for index, value in enumerate(
            nH_vals
        )
    }

    # --------------------------------------------------------------
    # 9. Initialize reduced-deviance grid.
    # --------------------------------------------------------------
    E = np.full(
        (
            len(nH_vals),
            len(x_vals),
        ),
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # 10. Compute reduced Poisson deviance for every matching pair.
    # --------------------------------------------------------------
    for (
        x_m_raw,
        nH_m_raw,
        x_m,
        nH_m,
        mat_m,
    ) in zip(
        xs_m_raw,
        nHs_m_raw,
        xs_m,
        nHs_m,
        mats_m,
    ):

        key = (
            float(x_m),
            float(nH_m),
        )

        if key not in data_map:
            continue

        (
            x_d_raw,
            nH_d_raw,
            mat_data,
        ) = data_map[key]

        # ----------------------------------------------------------
        # Verify x coordinate.
        # ----------------------------------------------------------
        if not np.isclose(
            x_m_raw,
            x_d_raw,
            rtol=0.0,
            atol=abs_tol,
        ):
            raise ValueError(
                "Rounded x coordinates matched, but raw values "
                f"differ: model={x_m_raw:.12g}, "
                f"data={x_d_raw:.12g}."
            )

        # ----------------------------------------------------------
        # Verify nH coordinate.
        #
        # nH values are very large, so abs_tol is also used as a
        # relative tolerance.
        # ----------------------------------------------------------
        if not np.isclose(
            nH_m_raw,
            nH_d_raw,
            rtol=abs_tol,
            atol=abs_tol,
        ):
            raise ValueError(
                "Rounded nH coordinates matched, but raw values "
                f"differ: model={nH_m_raw:.12g}, "
                f"data={nH_d_raw:.12g}."
            )

        # ----------------------------------------------------------
        # Validate generated image.
        # ----------------------------------------------------------
        mat_model = np.squeeze(
            np.asarray(
                mat_m,
                dtype=np.float64,
            )
        )

        if mat_model.ndim != 2:
            raise ValueError(
                f"Model matrix at {key} must be 2-D after "
                f"squeezing; got shape {mat_model.shape}."
            )

        if mat_model.shape != mat_data.shape:
            raise ValueError(
                f"Shape mismatch for "
                f"(x={key[0]}, nH={key[1]}): "
                f"model={mat_model.shape}, "
                f"data={mat_data.shape}."
            )

        if not np.all(np.isfinite(mat_model)):
            raise ValueError(
                f"Model matrix at {key} contains NaN or "
                "infinite values."
            )

        if np.any(mat_model < 0.0):
            raise ValueError(
                f"Model matrix at {key} contains negative counts."
            )

        # ----------------------------------------------------------
        # Two-sample Poisson deviance.
        #
        # RefleX = data
        # Generated = model
        #
        # No variance image, Fano factor, or count threshold is used.
        # A multiplicative normalization N is fitted only if use_N is
        # True (see radial_reduced_poisson_deviance_abs).
        # ----------------------------------------------------------
        err = radial_reduced_poisson_deviance_abs(
            data=mat_data,
            model=mat_model,
            n_annuli=n_annuli,
            pixel_scale_arcsec=pixel_scale_arcsec,
            use_N=use_N,
        )

        err = float(err)

        E[
            inh[key[1]],
            ix[key[0]],
        ] = err

    # --------------------------------------------------------------
    # 11. Basic grid statistics.
    # --------------------------------------------------------------
    finite = np.isfinite(E)

    n_total = int(
        E.size
    )

    n_finite = int(
        finite.sum()
    )

    stats: dict = {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_total - n_finite,
        "grid_shape": (
            len(nH_vals),
            len(x_vals),
        ),
        "statistic": (
            "two-sample radial Poisson deviance"
            + (
                " (N-normalized)"
                if use_N
                else " (absolute, unnormalized)"
            )
        ),
        "use_N": use_N,
        "dof_per_cell": (
            n_annuli - 1
            if use_N
            else n_annuli
        ),
    }

    # --------------------------------------------------------------
    # 12. Resolve colour limits.
    # --------------------------------------------------------------
    res_vmin = None
    res_vmax = None

    extend = "neither"

    clipped_above_idx = np.empty(
        (0, 2),
        dtype=int,
    )

    clipped_below_idx = np.empty(
        (0, 2),
        dtype=int,
    )

    if n_finite:

        res_vmin, res_vmax = _resolve_colour_limits(
            E,
            finite,
            vmin=vmin,
            vmax=vmax,
            diverging=diverging,
            log_norm=log_norm,
            vcenter=vcenter,
        )

        tiny = 1e-12

        clipped_above_idx = np.argwhere(
            finite
            & (E > res_vmax + tiny)
        )

        clipped_below_idx = np.argwhere(
            finite
            & (E < res_vmin - tiny)
        )

        has_above = (
            clipped_above_idx.shape[0] > 0
        )

        has_below = (
            clipped_below_idx.shape[0] > 0
        )

        if has_above and has_below:
            extend = "both"

        elif has_above:
            extend = "max"

        elif has_below:
            extend = "min"

        stats.update(
            {
                "vmin_used": res_vmin,
                "vmax_used": res_vmax,
                "extend": extend,
                "n_clipped_above": int(
                    clipped_above_idx.shape[0]
                ),
                "n_clipped_below": int(
                    clipped_below_idx.shape[0]
                ),
            }
        )

    # --------------------------------------------------------------
    # 13. Formatting helper.
    # --------------------------------------------------------------
    def _fmt_x(
        stored_value: float,
    ) -> str:

        return (
            f"{stored_value * x_display_scale:.4f}"
        )

    # --------------------------------------------------------------
    # 14. Summary statistics.
    # --------------------------------------------------------------
    if n_finite:

        v = E[finite]

        mean_err = float(
            v.mean()
        )

        std_err = (
            float(
                v.std(ddof=1)
            )
            if n_finite > 1
            else float("nan")
        )

        sem_err = (
            std_err / np.sqrt(n_finite)
            if n_finite > 1
            else float("nan")
        )

        median_err = float(
            np.median(v)
        )

        q25, q75, q90, q95, q99 = (
            float(
                np.percentile(v, p)
            )
            for p in (
                25,
                75,
                90,
                95,
                99,
            )
        )

        min_err = float(
            v.min()
        )

        max_err = float(
            v.max()
        )

        trimmed10 = _trimmed_mean(
            v,
            0.10,
        )

        # Retained for continuity with the previous heatmap summary.
        # D_red is non-negative, so a geometric mean is mathematically
        # well defined whenever positive values are present.
        positive_vals = v[
            v > 0.0
        ]

        geo_mean = (
            float(
                np.exp(
                    np.mean(
                        np.log(
                            positive_vals
                        )
                    )
                )
            )
            if positive_vals.size
            else float("nan")
        )

        # ----------------------------------------------------------
        # Fractions below selected reduced-deviance levels.
        # ----------------------------------------------------------
        fractions = {}

        for level in consistency_levels:

            count = int(
                np.count_nonzero(
                    v <= level
                )
            )

            fractions[
                float(level)
            ] = (
                count,
                count / n_finite,
            )

        stats.update(
            {
                "mean": mean_err,
                "std": std_err,
                "sem": sem_err,
                "median": median_err,
                "p25": q25,
                "p75": q75,
                "p90": q90,
                "p95": q95,
                "p99": q99,
                "min": min_err,
                "max": max_err,
                "trimmed_mean_10": trimmed10,
                "geometric_mean": geo_mean,
                "fraction_below": fractions,
            }
        )

        # ----------------------------------------------------------
        # Find locations corresponding to extrema and representative
        # values.
        # ----------------------------------------------------------
        finite_positions = np.argwhere(
            finite
        )

        argmin = int(
            np.argmin(v)
        )

        argmax = int(
            np.argmax(v)
        )

        argmean = int(
            np.argmin(
                np.abs(
                    v - mean_err
                )
            )
        )

        argmedian = int(
            np.argmin(
                np.abs(
                    v - median_err
                )
            )
        )

        def _pos_to_coords(
            pos_idx,
        ):

            row, col = (
                finite_positions[
                    pos_idx
                ]
            )

            return (
                float(
                    x_vals[col]
                ),
                float(
                    nH_vals[row]
                ),
            )

        min_x, min_nH = (
            _pos_to_coords(argmin)
        )

        max_x, max_nH = (
            _pos_to_coords(argmax)
        )

        mean_x, mean_nH = (
            _pos_to_coords(argmean)
        )

        med_x, med_nH = (
            _pos_to_coords(argmedian)
        )

        stats.update(
            {
                "argmin_coord": (
                    min_x,
                    min_nH,
                ),
                "argmax_coord": (
                    max_x,
                    max_nH,
                ),
                "closest_to_mean_coord": (
                    mean_x,
                    mean_nH,
                ),
                "closest_to_median_coord": (
                    med_x,
                    med_nH,
                ),
            }
        )

        # ----------------------------------------------------------
        # Row aggregates.
        # ----------------------------------------------------------
        row_means = []
        row_medians = []
        row_counts = []

        for row_index in range(
            len(nH_vals)
        ):

            row = E[
                row_index
            ]

            mask = np.isfinite(
                row
            )

            n_row = int(
                np.count_nonzero(
                    mask
                )
            )

            row_counts.append(
                n_row
            )

            row_means.append(
                float(
                    np.mean(
                        row[mask]
                    )
                )
                if n_row
                else float("nan")
            )

            row_medians.append(
                float(
                    np.median(
                        row[mask]
                    )
                )
                if n_row
                else float("nan")
            )

        # ----------------------------------------------------------
        # Column aggregates.
        # ----------------------------------------------------------
        col_means = []
        col_medians = []
        col_counts = []

        for col_index in range(
            len(x_vals)
        ):

            col = E[
                :,
                col_index,
            ]

            mask = np.isfinite(
                col
            )

            n_col = int(
                np.count_nonzero(
                    mask
                )
            )

            col_counts.append(
                n_col
            )

            col_means.append(
                float(
                    np.mean(
                        col[mask]
                    )
                )
                if n_col
                else float("nan")
            )

            col_medians.append(
                float(
                    np.median(
                        col[mask]
                    )
                )
                if n_col
                else float("nan")
            )

        stats.update(
            {
                "row_means": np.array(
                    row_means
                ),
                "row_medians": np.array(
                    row_medians
                ),
                "col_means": np.array(
                    col_means
                ),
                "col_medians": np.array(
                    col_medians
                ),
                "x_vals": x_vals,
                "nH_vals": nH_vals,
                "E": E,
            }
        )

        # ----------------------------------------------------------
        # Clipped-cell records.
        # ----------------------------------------------------------
        clipped_records = [
            (
                float(
                    x_vals[col]
                ),
                float(
                    nH_vals[row]
                ),
                float(
                    E[row, col]
                ),
            )
            for row, col
            in clipped_above_idx
        ]

        clipped_records.sort(
            key=lambda rec: rec[2],
            reverse=True,
        )

        stats[
            "clipped_above"
        ] = clipped_records

        # ----------------------------------------------------------
        # 15. Print summary.
        # ----------------------------------------------------------
        if print_stats:

            width = 66

            header = (
                title
                if title
                else "Poisson-deviance grid"
            )

            print(
                "=" * width
            )

            print(
                f" Summary: {header}"
            )

            print(
                "=" * width
            )

            print(
                f" Statistic           : "
                f"radial two-sample Poisson deviance"
                + (
                    " (N-normalized)"
                    if use_N
                    else ""
                )
            )

            print(
                f" Grid                : "
                f"{len(nH_vals)} x {len(x_vals)}  "
                f"({n_total} cells, "
                f"{n_finite} finite, "
                f"{n_total - n_finite} missing)"
            )

            print(
                f" Annuli per image    : "
                f"{n_annuli}"
            )

            print(
                f" Mean D_red          : "
                f"{mean_err:.4f}"
            )

            print(
                f" Std (ddof=1)        : "
                f"{std_err:.4f}"
                f"     SEM: {sem_err:.4f}"
            )

            print(
                f" Median D_red        : "
                f"{median_err:.4f}"
            )

            print(
                f" Trimmed mean (10%)  : "
                f"{trimmed10:.4f}"
            )

            print(
                f" Geometric mean      : "
                f"{geo_mean:.4f}"
            )

            print(
                f" IQR                 : "
                f"{q25:.4f} - {q75:.4f}"
            )

            print(
                f" Percentiles         : "
                f"p90={q90:.4f}  "
                f"p95={q95:.4f}  "
                f"p99={q99:.4f}"
            )

            print(
                f" Min / Max           : "
                f"{min_err:.4f} / "
                f"{max_err:.4f}"
            )

            for level in consistency_levels:

                count, frac = fractions[
                    float(level)
                ]

                print(
                    f" Fraction D_red <= "
                    f"{level:<5.2f}: "
                    f"{100.0 * frac:6.2f}%  "
                    f"({count}/{n_finite})"
                )

            print(
                "-" * width
            )

            print(
                " Extremes"
            )

            print(
                f"   Lowest   "
                f"{min_err:10.4f}  at "
                f"x={_fmt_x(min_x)}, "
                f"nH={min_nH:.3e}"
            )

            print(
                f"   Highest  "
                f"{max_err:10.4f}  at "
                f"x={_fmt_x(max_x)}, "
                f"nH={max_nH:.3e}"
            )

            print(
                f"   ~Mean    "
                f"{v[argmean]:10.4f}  at "
                f"x={_fmt_x(mean_x)}, "
                f"nH={mean_nH:.3e}"
            )

            print(
                f"   ~Median  "
                f"{v[argmedian]:10.4f}  at "
                f"x={_fmt_x(med_x)}, "
                f"nH={med_nH:.3e}"
            )

            # ------------------------------------------------------
            # Colour-scale clipping.
            # ------------------------------------------------------
            if print_clipped:

                print(
                    "-" * width
                )

                print(
                    f" Colour scale        : "
                    f"vmin={res_vmin:.4g}, "
                    f"vmax={res_vmax:.4g}, "
                    f"extend='{extend}'"
                )

                n_above = len(
                    clipped_records
                )

                if n_above:

                    print(
                        f" Clipped above vmax  : "
                        f"{n_above}/{n_finite} cells "
                        f"({100.0 * n_above / n_finite:.1f}%)"
                    )

                    for (
                        x_c,
                        nH_c,
                        val,
                    ) in clipped_records[
                        :max_clipped_listed
                    ]:

                        print(
                            f"     "
                            f"x={_fmt_x(x_c)}  "
                            f"nH={nH_c:.3e}   "
                            f"D_red={val:.2f}"
                        )

                    if (
                        n_above
                        > max_clipped_listed
                    ):
                        print(
                            f"     ... and "
                            f"{n_above - max_clipped_listed} more"
                        )

                    print()

                    print(
                        " Suggested caption sentence:"
                    )

                    print(
                        f"   Cells with "
                        f"$D_{{\\rm red}} > "
                        f"{res_vmax:g}$ exceed the "
                        f"colour-scale limit and are clipped to"
                    )

                    print(
                        "   the darkest shade; their "
                        "numerical values are printed "
                        "in the corresponding cell."
                    )

                else:

                    print(
                        " Clipped above vmax  : none "
                        "(no clipping note needed in "
                        "the caption)"
                    )

                n_below = int(
                    clipped_below_idx.shape[0]
                )

                if n_below:

                    print(
                        f" Clipped below vmin  : "
                        f"{n_below} cells"
                    )

            # ------------------------------------------------------
            # Per-nH statistics.
            # ------------------------------------------------------
            if print_row_means:

                print(
                    "-" * width
                )

                print(
                    " Per-nH (row) aggregates"
                )

                print(
                    f"   {'nH':<12} "
                    f"{'mean':>10} "
                    f"{'median':>10} "
                    f"{'max':>10} "
                    f"{'n':>5}"
                )

                for (
                    row_index,
                    nH_val,
                ) in enumerate(
                    nH_vals
                ):

                    row = E[
                        row_index
                    ]

                    mask = np.isfinite(
                        row
                    )

                    row_max = (
                        float(
                            np.max(
                                row[mask]
                            )
                        )
                        if row_counts[
                            row_index
                        ]
                        else float("nan")
                    )

                    print(
                        f"   {nH_val:<12.3e} "
                        f"{row_means[row_index]:>10.4f} "
                        f"{row_medians[row_index]:>10.4f} "
                        f"{row_max:>10.4f} "
                        f"{row_counts[row_index]:>5d}"
                    )

            # ------------------------------------------------------
            # Per-x statistics.
            # ------------------------------------------------------
            if print_col_means:

                print(
                    "-" * width
                )

                print(
                    " Per-x (column) aggregates"
                )

                print(
                    f"   {'x':<12} "
                    f"{'mean':>10} "
                    f"{'median':>10} "
                    f"{'max':>10} "
                    f"{'n':>5}"
                )

                for (
                    col_index,
                    x_val,
                ) in enumerate(
                    x_vals
                ):

                    col = E[
                        :,
                        col_index,
                    ]

                    mask = np.isfinite(
                        col
                    )

                    col_max = (
                        float(
                            np.max(
                                col[mask]
                            )
                        )
                        if col_counts[
                            col_index
                        ]
                        else float("nan")
                    )

                    print(
                        f"   {_fmt_x(x_val):<12} "
                        f"{col_means[col_index]:>10.4f} "
                        f"{col_medians[col_index]:>10.4f} "
                        f"{col_max:>10.4f} "
                        f"{col_counts[col_index]:>5d}"
                    )

            print(
                "=" * width
            )

    else:

        if print_stats:
            print(
                "No finite Poisson-deviance values; "
                "all summary statistics are NaN."
            )

    # --------------------------------------------------------------
    # 16. Plot heatmap.
    # --------------------------------------------------------------
    if show:

        if not finite.any():
            raise RuntimeError(
                "No finite Poisson-deviance values to plot."
            )

        with plt.rc_context(
            JOURNAL_RC
        ):

            fig, ax_err = plt.subplots(
                figsize=figsize
            )

            if diverging:

                from matplotlib.colors import (
                    TwoSlopeNorm,
                )

                if log_norm:
                    raise ValueError(
                        "diverging=True and log_norm=True "
                        "are mutually exclusive."
                    )

                used_cmap = (
                    "RdBu_r"
                    if cmap == "magma"
                    else cmap
                )

            else:

                used_cmap = cmap

            cmap_obj = (
                plt.get_cmap(
                    used_cmap
                ).copy()
            )

            if over_color is not None:
                cmap_obj.set_over(
                    over_color
                )

            if under_color is not None:
                cmap_obj.set_under(
                    under_color
                )

            # ------------------------------------------------------
            # Diverging normalization.
            # ------------------------------------------------------
            if diverging:

                norm = TwoSlopeNorm(
                    vmin=res_vmin,
                    vcenter=vcenter,
                    vmax=res_vmax,
                )

                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=norm,
                    interpolation="nearest",
                )

            # ------------------------------------------------------
            # Logarithmic normalization.
            # ------------------------------------------------------
            elif log_norm:

                from matplotlib.colors import (
                    LogNorm,
                )

                positive = (
                    finite
                    & (E > 0.0)
                )

                if not positive.any():
                    raise RuntimeError(
                        "Logarithmic normalization "
                        "requires at least one positive "
                        "finite deviance."
                    )

                E_plot = np.where(
                    positive,
                    E,
                    np.nan,
                )

                if res_vmin <= 0.0:
                    raise ValueError(
                        "vmin must be strictly positive "
                        "when log_norm=True."
                    )

                if res_vmax <= res_vmin:
                    raise ValueError(
                        "vmax must be greater than vmin."
                    )

                im_err = ax_err.imshow(
                    E_plot,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=LogNorm(
                        vmin=res_vmin,
                        vmax=res_vmax,
                    ),
                    interpolation="nearest",
                )

            # ------------------------------------------------------
            # Standard linear normalization.
            # ------------------------------------------------------
            else:

                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    vmin=res_vmin,
                    vmax=res_vmax,
                    interpolation="nearest",
                )

            # ------------------------------------------------------
            # Cell annotations.
            # ------------------------------------------------------
            if annotate_above is not None:

                for i, j in np.argwhere(
                    finite
                    & (E > annotate_above)
                ):

                    ax_err.text(
                        j,
                        i,
                        f"{E[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=annotate_fontsize,
                        fontweight="bold",
                    )

            # ------------------------------------------------------
            # Axis ticks.
            # ------------------------------------------------------
            xticks = np.arange(
                len(x_vals)
            )

            yticks = np.arange(
                len(nH_vals)
            )

            x_labels = [
                f"{value * x_display_scale:.4f}"
                for value
                in x_vals[
                    ::tick_stride
                ]
            ]

            nh_labels = [
                _latex_sci(
                    value
                )
                for value
                in nH_vals[
                    ::tick_stride
                ]
            ]

            ax_err.set_xticks(
                xticks[
                    ::tick_stride
                ]
            )

            ax_err.set_xticklabels(
                x_labels,
                rotation=45,
                ha="right",
                fontsize=10,
            )

            ax_err.set_yticks(
                yticks[
                    ::tick_stride
                ]
            )

            ax_err.set_yticklabels(
                nh_labels,
                fontsize=10,
            )

            ax_err.set_xlabel(
                r"$x$"
            )

            ax_err.set_ylabel(
                r"$N_\mathrm{H}$ [cm$^{-2}$]"
            )

            ax_err.tick_params(
                axis="both",
                which="both",
                length=0,
            )

            if title:

                ax_err.set_title(
                    title,
                    pad=8,
                )

            # ------------------------------------------------------
            # Colourbar.
            # ------------------------------------------------------
            cbar_err = fig.colorbar(
                im_err,
                ax=ax_err,
                pad=0.02,
                extend=extend,
            )

            cbar_err.set_label(
                cbar_label
            )

            cbar_err.ax.tick_params(
                direction="in",
                length=3,
            )

            cbar_err.outline.set_linewidth(
                0.8
            )

            if diverging:

                ticks = cbar_err.get_ticks()

                if vcenter not in ticks:

                    ticks = np.sort(
                        np.append(
                            ticks,
                            vcenter,
                        )
                    )

                cbar_err.set_ticks(
                    ticks[
                        (ticks >= res_vmin)
                        & (ticks <= res_vmax)
                    ]
                )

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------------
    # 17. Return values.
    # --------------------------------------------------------------
    if return_stats:

        return (
            E,
            x_vals,
            nH_vals,
            stats,
        )

    return (
        E,
        x_vals,
        nH_vals,
    )


def poisson_deviance_heatmap_pairwise_mean_npz(
    npz_paths,
    *,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    decimals: int = 6,
    abs_tol: float = 1e-6,
    cmap: str = "magma",
    diverging: bool = False,
    vcenter: float = 1.0,
    title: str = "",
    log_norm: bool = False,
    tick_stride: int = 1,
    figsize=(7, 5),
    show: bool = True,
    vmin=None,
    vmax=None,
    annotate_above: float | None = None,
    annotate_fontsize: float = 13.0,
    x_display_scale: float = 1e-4,
    over_color: str | None = None,
    under_color: str | None = None,
    consistency_levels=(1.0, 2.0, 3.0),
    print_stats: bool = True,
    print_row_means: bool = True,
    print_col_means: bool = True,
    print_clipped: bool = True,
    max_clipped_listed: int = 20,
    return_stats: bool = False,
):
    """
    Compute, for every (x, nH) cell shared by an arbitrary number of NPZ
    archives, the mean radial reduced two-sample Poisson deviance taken
    over all unique pairs of archives, and optionally plot the resulting
    heatmap.

    This is the multi-archive counterpart of
    ``poisson_deviance_heatmap_between_npz``. Instead of one fixed "model"
    archive compared against one fixed "data" archive, an arbitrary list
    of archives is supplied via ``npz_paths`` (no archive is singled out
    as the reference). For a cell present in every archive, let
    M_1, ..., M_N be the matching matrices drawn from the N archives. The
    reduced Poisson deviance

        D_red(M_a, M_b)

    (as computed by ``radial_reduced_poisson_deviance_abs``, see
    ``poisson_deviance_heatmap_between_npz`` for the definition of
    D_red) is evaluated for every unique unordered pair (a, b), a < b,
    and the cell value is their mean:

        E_cell = mean_{a < b} D_red(M_a, M_b),

    averaged over all C(N, 2) pairs. With N=2 this reduces exactly to
    ``poisson_deviance_heatmap_between_npz``.

    Only (x, nH) coordinates present in every archive (after rounding to
    ``decimals``) are included; a cell missing from any one archive is
    left as NaN. As in ``poisson_deviance_heatmap_between_npz``, no
    minimum-count bin filter is applied, annuli with zero data-side counts
    are excluded, no Fano factors or variance images are used, and no
    multiplicative normalization is fitted.

    Parameters
    ----------
    npz_paths : sequence of str
        Paths to two or more NPZ archives, each containing ``xs``,
        ``nHs``, and ``mats`` (as produced by ``save_npy_to_npz``).

    n_annuli : int
        Number of radial annuli used for each pairwise Poisson-deviance
        evaluation.

    pixel_scale_arcsec : float
        Pixel width in arcseconds, passed to the radial deviance
        function.

    decimals : int
        Number of decimal places used when matching stored coordinates
        across archives.

    abs_tol : float
        Tolerance used to verify that coordinates matched after rounding
        are genuinely consistent across every archive.

    consistency_levels : sequence of float
        Thresholds for reporting the fraction of cells satisfying
        E_cell <= level.

    Returns
    -------
    E : ndarray
        Grid of mean pairwise reduced Poisson-deviance values, shape
        (len(nH_vals), len(x_vals)).

    x_vals : ndarray
        Sorted shared x coordinates in stored units.

    nH_vals : ndarray
        Sorted shared nH coordinates.

    stats : dict, optional
        Returned only when ``return_stats=True``. In addition to the
        fields documented for ``poisson_deviance_heatmap_between_npz``,
        includes ``n_archives`` (N), ``n_pairs_per_cell`` (C(N, 2)), and
        ``archive_labels`` (each archive's filename stem, in the order
        of ``npz_paths``).
    """

    import itertools
    from pathlib import Path

    # --------------------------------------------------------------
    # 1. Validate scalar options.
    # --------------------------------------------------------------
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")

    if abs_tol < 0.0 or not np.isfinite(abs_tol):
        raise ValueError("abs_tol must be finite and non-negative.")

    if tick_stride < 1:
        raise ValueError("tick_stride must be a positive integer.")

    if n_annuli <= 0:
        raise ValueError("n_annuli must be a positive integer.")

    if pixel_scale_arcsec <= 0.0:
        raise ValueError("pixel_scale_arcsec must be positive.")

    npz_paths = list(npz_paths)
    n_archives = len(npz_paths)

    if n_archives < 2:
        raise ValueError(
            "Provide at least 2 NPZ archive paths."
        )

    cbar_label = r"$\overline{D}_{\mathrm{red}}$"

    # --------------------------------------------------------------
    # 2. Load every archive into a per-archive coordinate -> matrix map.
    # --------------------------------------------------------------
    archive_maps = []
    archive_labels = []

    required_keys = {"xs", "nHs", "mats"}

    for path in npz_paths:

        archive_labels.append(Path(path).stem)

        with np.load(path) as npz:

            missing_keys = required_keys.difference(npz.files)

            if missing_keys:
                raise KeyError(
                    f"NPZ archive {path!r} is missing required "
                    f"keys: {sorted(missing_keys)}"
                )

            xs_raw = np.asarray(
                npz["xs"], dtype=np.float64
            ).reshape(-1)

            nHs_raw = np.asarray(
                npz["nHs"], dtype=np.float64
            ).reshape(-1)

            mats = np.asarray(npz["mats"])

        if not (len(xs_raw) == len(nHs_raw) == len(mats)):
            raise ValueError(
                f"Arrays xs, nHs, and mats in {path!r} must "
                "have equal lengths."
            )

        for name, values in (("xs", xs_raw), ("nHs", nHs_raw)):
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} in {path!r} contain NaN or "
                    "infinite values."
                )

        xs = np.round(xs_raw, decimals)
        nHs = np.round(nHs_raw, decimals)

        this_map = {}

        for x_raw, nH_raw, x_key, nH_key, mat in zip(
            xs_raw, nHs_raw, xs, nHs, mats
        ):

            key = (float(x_key), float(nH_key))

            if key in this_map:
                raise ValueError(
                    f"Duplicate coordinate after rounding in "
                    f"{path!r}: x={key[0]}, nH={key[1]}."
                )

            mat2d = np.squeeze(
                np.asarray(mat, dtype=np.float64)
            )

            if mat2d.ndim != 2:
                raise ValueError(
                    f"Matrix at {key} in {path!r} must be 2-D "
                    f"after squeezing; got shape {mat2d.shape}."
                )

            if not np.all(np.isfinite(mat2d)):
                raise ValueError(
                    f"Matrix at {key} in {path!r} contains NaN "
                    "or infinite values."
                )

            if np.any(mat2d < 0.0):
                raise ValueError(
                    f"Matrix at {key} in {path!r} contains "
                    "negative counts."
                )

            this_map[key] = (float(x_raw), float(nH_raw), mat2d)

        archive_maps.append(this_map)

    # --------------------------------------------------------------
    # 3. Find parameter-space points shared by every archive.
    # --------------------------------------------------------------
    shared_pairs = set(archive_maps[0].keys())

    for this_map in archive_maps[1:]:
        shared_pairs &= set(this_map.keys())

    if not shared_pairs:
        raise RuntimeError(
            "No (x, nH) pair is present in every supplied "
            "NPZ archive."
        )

    x_vals = np.array(
        sorted({pair[0] for pair in shared_pairs}),
        dtype=np.float64,
    )

    nH_vals = np.array(
        sorted({pair[1] for pair in shared_pairs}),
        dtype=np.float64,
    )

    ix = {value: index for index, value in enumerate(x_vals)}
    inh = {value: index for index, value in enumerate(nH_vals)}

    # --------------------------------------------------------------
    # 4. Initialize mean-deviance grid.
    # --------------------------------------------------------------
    E = np.full(
        (len(nH_vals), len(x_vals)),
        np.nan,
        dtype=np.float64,
    )

    pair_indices = list(
        itertools.combinations(range(n_archives), 2)
    )

    n_pairs_per_cell = len(pair_indices)

    # --------------------------------------------------------------
    # 5. Compute the mean pairwise reduced Poisson deviance for every
    #    shared cell.
    # --------------------------------------------------------------
    for key in shared_pairs:

        x_raw_ref, nH_raw_ref, mat_ref = archive_maps[0][key]
        mats_here = [mat_ref]

        for archive_index in range(1, n_archives):

            x_raw, nH_raw, mat = archive_maps[archive_index][key]

            if not np.isclose(
                x_raw, x_raw_ref, rtol=0.0, atol=abs_tol
            ):
                raise ValueError(
                    "Rounded x coordinates matched, but raw "
                    f"values differ: "
                    f"{archive_labels[0]}={x_raw_ref:.12g}, "
                    f"{archive_labels[archive_index]}={x_raw:.12g}."
                )

            if not np.isclose(
                nH_raw, nH_raw_ref, rtol=abs_tol, atol=abs_tol
            ):
                raise ValueError(
                    "Rounded nH coordinates matched, but raw "
                    f"values differ: "
                    f"{archive_labels[0]}={nH_raw_ref:.12g}, "
                    f"{archive_labels[archive_index]}={nH_raw:.12g}."
                )

            if mat.shape != mat_ref.shape:
                raise ValueError(
                    f"Shape mismatch for (x={key[0]}, "
                    f"nH={key[1]}): "
                    f"{archive_labels[0]}={mat_ref.shape}, "
                    f"{archive_labels[archive_index]}={mat.shape}."
                )

            mats_here.append(mat)

        pair_values = [
            float(
                radial_reduced_poisson_deviance_abs(
                    data=mats_here[a],
                    model=mats_here[b],
                    n_annuli=n_annuli,
                    pixel_scale_arcsec=pixel_scale_arcsec,
                )
            )
            for a, b in pair_indices
        ]

        E[inh[key[1]], ix[key[0]]] = float(
            np.mean(pair_values)
        )

    # --------------------------------------------------------------
    # 6. Basic grid statistics.
    # --------------------------------------------------------------
    finite = np.isfinite(E)

    n_total = int(E.size)
    n_finite = int(finite.sum())

    stats: dict = {
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_total - n_finite,
        "grid_shape": (len(nH_vals), len(x_vals)),
        "statistic": (
            "mean, over all unique archive pairs, of the "
            "absolute two-sample radial Poisson deviance"
        ),
        "dof_per_cell": n_annuli,
        "n_archives": n_archives,
        "n_pairs_per_cell": n_pairs_per_cell,
        "archive_labels": archive_labels,
    }

    # --------------------------------------------------------------
    # 7. Resolve colour limits.
    # --------------------------------------------------------------
    res_vmin = None
    res_vmax = None

    extend = "neither"

    clipped_above_idx = np.empty((0, 2), dtype=int)
    clipped_below_idx = np.empty((0, 2), dtype=int)

    if n_finite:

        res_vmin, res_vmax = _resolve_colour_limits(
            E,
            finite,
            vmin=vmin,
            vmax=vmax,
            diverging=diverging,
            log_norm=log_norm,
            vcenter=vcenter,
        )

        tiny = 1e-12

        clipped_above_idx = np.argwhere(
            finite & (E > res_vmax + tiny)
        )

        clipped_below_idx = np.argwhere(
            finite & (E < res_vmin - tiny)
        )

        has_above = clipped_above_idx.shape[0] > 0
        has_below = clipped_below_idx.shape[0] > 0

        if has_above and has_below:
            extend = "both"
        elif has_above:
            extend = "max"
        elif has_below:
            extend = "min"

        stats.update(
            {
                "vmin_used": res_vmin,
                "vmax_used": res_vmax,
                "extend": extend,
                "n_clipped_above": int(clipped_above_idx.shape[0]),
                "n_clipped_below": int(clipped_below_idx.shape[0]),
            }
        )

    # --------------------------------------------------------------
    # 8. Formatting helper.
    # --------------------------------------------------------------
    def _fmt_x(stored_value: float) -> str:
        return f"{stored_value * x_display_scale:.4f}"

    # --------------------------------------------------------------
    # 9. Summary statistics.
    # --------------------------------------------------------------
    if n_finite:

        v = E[finite]

        mean_err = float(v.mean())

        std_err = (
            float(v.std(ddof=1))
            if n_finite > 1
            else float("nan")
        )

        sem_err = (
            std_err / np.sqrt(n_finite)
            if n_finite > 1
            else float("nan")
        )

        median_err = float(np.median(v))

        q25, q75, q90, q95, q99 = (
            float(np.percentile(v, p))
            for p in (25, 75, 90, 95, 99)
        )

        min_err = float(v.min())
        max_err = float(v.max())

        trimmed10 = _trimmed_mean(v, 0.10)

        positive_vals = v[v > 0.0]

        geo_mean = (
            float(np.exp(np.mean(np.log(positive_vals))))
            if positive_vals.size
            else float("nan")
        )

        fractions = {}

        for level in consistency_levels:

            count = int(np.count_nonzero(v <= level))

            fractions[float(level)] = (
                count,
                count / n_finite,
            )

        stats.update(
            {
                "mean": mean_err,
                "std": std_err,
                "sem": sem_err,
                "median": median_err,
                "p25": q25,
                "p75": q75,
                "p90": q90,
                "p95": q95,
                "p99": q99,
                "min": min_err,
                "max": max_err,
                "trimmed_mean_10": trimmed10,
                "geometric_mean": geo_mean,
                "fraction_below": fractions,
            }
        )

        finite_positions = np.argwhere(finite)

        argmin = int(np.argmin(v))
        argmax = int(np.argmax(v))

        argmean = int(np.argmin(np.abs(v - mean_err)))
        argmedian = int(np.argmin(np.abs(v - median_err)))

        def _pos_to_coords(pos_idx):
            row, col = finite_positions[pos_idx]
            return (
                float(x_vals[col]),
                float(nH_vals[row]),
            )

        min_x, min_nH = _pos_to_coords(argmin)
        max_x, max_nH = _pos_to_coords(argmax)
        mean_x, mean_nH = _pos_to_coords(argmean)
        med_x, med_nH = _pos_to_coords(argmedian)

        stats.update(
            {
                "argmin_coord": (min_x, min_nH),
                "argmax_coord": (max_x, max_nH),
                "closest_to_mean_coord": (mean_x, mean_nH),
                "closest_to_median_coord": (med_x, med_nH),
            }
        )

        row_means = []
        row_medians = []
        row_counts = []

        for row_index in range(len(nH_vals)):

            row = E[row_index]
            mask = np.isfinite(row)
            n_row = int(np.count_nonzero(mask))

            row_counts.append(n_row)

            row_means.append(
                float(np.mean(row[mask])) if n_row else float("nan")
            )

            row_medians.append(
                float(np.median(row[mask])) if n_row else float("nan")
            )

        col_means = []
        col_medians = []
        col_counts = []

        for col_index in range(len(x_vals)):

            col = E[:, col_index]
            mask = np.isfinite(col)
            n_col = int(np.count_nonzero(mask))

            col_counts.append(n_col)

            col_means.append(
                float(np.mean(col[mask])) if n_col else float("nan")
            )

            col_medians.append(
                float(np.median(col[mask])) if n_col else float("nan")
            )

        stats.update(
            {
                "row_means": np.array(row_means),
                "row_medians": np.array(row_medians),
                "col_means": np.array(col_means),
                "col_medians": np.array(col_medians),
                "x_vals": x_vals,
                "nH_vals": nH_vals,
                "E": E,
            }
        )

        clipped_records = [
            (
                float(x_vals[col]),
                float(nH_vals[row]),
                float(E[row, col]),
            )
            for row, col in clipped_above_idx
        ]

        clipped_records.sort(key=lambda rec: rec[2], reverse=True)

        stats["clipped_above"] = clipped_records

        # ------------------------------------------------------
        # 10. Print summary.
        # ------------------------------------------------------
        if print_stats:

            width = 66

            header = (
                title if title else "Mean pairwise Poisson-deviance grid"
            )

            print("=" * width)
            print(f" Summary: {header}")
            print("=" * width)

            print(
                f" Statistic           : "
                f"mean pairwise radial two-sample Poisson deviance"
            )

            print(
                f" Archives            : "
                f"{n_archives}  ({', '.join(archive_labels)})"
            )

            print(
                f" Pairs per cell      : "
                f"{n_pairs_per_cell}"
            )

            print(
                f" Grid                : "
                f"{len(nH_vals)} x {len(x_vals)}  "
                f"({n_total} cells, "
                f"{n_finite} finite, "
                f"{n_total - n_finite} missing)"
            )

            print(f" Annuli per image    : {n_annuli}")
            print(f" Mean D_red          : {mean_err:.4f}")

            print(
                f" Std (ddof=1)        : "
                f"{std_err:.4f}     SEM: {sem_err:.4f}"
            )

            print(f" Median D_red        : {median_err:.4f}")
            print(f" Trimmed mean (10%)  : {trimmed10:.4f}")
            print(f" Geometric mean      : {geo_mean:.4f}")
            print(f" IQR                 : {q25:.4f} - {q75:.4f}")

            print(
                f" Percentiles         : "
                f"p90={q90:.4f}  p95={q95:.4f}  p99={q99:.4f}"
            )

            print(
                f" Min / Max           : "
                f"{min_err:.4f} / {max_err:.4f}"
            )

            for level in consistency_levels:

                count, frac = fractions[float(level)]

                print(
                    f" Fraction D_red <= {level:<5.2f}: "
                    f"{100.0 * frac:6.2f}%  ({count}/{n_finite})"
                )

            print("-" * width)
            print(" Extremes")

            print(
                f"   Lowest   {min_err:10.4f}  at "
                f"x={_fmt_x(min_x)}, nH={min_nH:.3e}"
            )

            print(
                f"   Highest  {max_err:10.4f}  at "
                f"x={_fmt_x(max_x)}, nH={max_nH:.3e}"
            )

            print(
                f"   ~Mean    {v[argmean]:10.4f}  at "
                f"x={_fmt_x(mean_x)}, nH={mean_nH:.3e}"
            )

            print(
                f"   ~Median  {v[argmedian]:10.4f}  at "
                f"x={_fmt_x(med_x)}, nH={med_nH:.3e}"
            )

            if print_clipped:

                print("-" * width)

                print(
                    f" Colour scale        : "
                    f"vmin={res_vmin:.4g}, vmax={res_vmax:.4g}, "
                    f"extend='{extend}'"
                )

                n_above = len(clipped_records)

                if n_above:

                    print(
                        f" Clipped above vmax  : "
                        f"{n_above}/{n_finite} cells "
                        f"({100.0 * n_above / n_finite:.1f}%)"
                    )

                    for x_c, nH_c, val in clipped_records[
                        :max_clipped_listed
                    ]:

                        print(
                            f"     x={_fmt_x(x_c)}  nH={nH_c:.3e}   "
                            f"D_red={val:.2f}"
                        )

                    if n_above > max_clipped_listed:
                        print(
                            f"     ... and "
                            f"{n_above - max_clipped_listed} more"
                        )

                    print()
                    print(" Suggested caption sentence:")

                    print(
                        f"   Cells with mean pairwise "
                        f"$D_{{\\rm red}} > {res_vmax:g}$ exceed the "
                        f"colour-scale limit and are clipped to"
                    )

                    print(
                        "   the darkest shade; their numerical "
                        "values are printed in the corresponding "
                        "cell."
                    )

                else:
                    print(
                        " Clipped above vmax  : none "
                        "(no clipping note needed in the caption)"
                    )

                n_below = int(clipped_below_idx.shape[0])

                if n_below:
                    print(
                        f" Clipped below vmin  : {n_below} cells"
                    )

            if print_row_means:

                print("-" * width)
                print(" Per-nH (row) aggregates")

                print(
                    f"   {'nH':<12} {'mean':>10} {'median':>10} "
                    f"{'max':>10} {'n':>5}"
                )

                for row_index, nH_val in enumerate(nH_vals):

                    row = E[row_index]
                    mask = np.isfinite(row)

                    row_max = (
                        float(np.max(row[mask]))
                        if row_counts[row_index]
                        else float("nan")
                    )

                    print(
                        f"   {nH_val:<12.3e} "
                        f"{row_means[row_index]:>10.4f} "
                        f"{row_medians[row_index]:>10.4f} "
                        f"{row_max:>10.4f} "
                        f"{row_counts[row_index]:>5d}"
                    )

            if print_col_means:

                print("-" * width)
                print(" Per-x (column) aggregates")

                print(
                    f"   {'x':<12} {'mean':>10} {'median':>10} "
                    f"{'max':>10} {'n':>5}"
                )

                for col_index, x_val in enumerate(x_vals):

                    col = E[:, col_index]
                    mask = np.isfinite(col)

                    col_max = (
                        float(np.max(col[mask]))
                        if col_counts[col_index]
                        else float("nan")
                    )

                    print(
                        f"   {_fmt_x(x_val):<12} "
                        f"{col_means[col_index]:>10.4f} "
                        f"{col_medians[col_index]:>10.4f} "
                        f"{col_max:>10.4f} "
                        f"{col_counts[col_index]:>5d}"
                    )

            print("=" * width)

    else:

        if print_stats:
            print(
                "No finite Poisson-deviance values; all summary "
                "statistics are NaN."
            )

    # --------------------------------------------------------------
    # 11. Plot heatmap.
    # --------------------------------------------------------------
    if show:

        if not finite.any():
            raise RuntimeError(
                "No finite Poisson-deviance values to plot."
            )

        with plt.rc_context(JOURNAL_RC):

            fig, ax_err = plt.subplots(figsize=figsize)

            if diverging:

                from matplotlib.colors import TwoSlopeNorm

                if log_norm:
                    raise ValueError(
                        "diverging=True and log_norm=True are "
                        "mutually exclusive."
                    )

                used_cmap = "RdBu_r" if cmap == "magma" else cmap

            else:
                used_cmap = cmap

            cmap_obj = plt.get_cmap(used_cmap).copy()

            if over_color is not None:
                cmap_obj.set_over(over_color)

            if under_color is not None:
                cmap_obj.set_under(under_color)

            if diverging:

                norm = TwoSlopeNorm(
                    vmin=res_vmin,
                    vcenter=vcenter,
                    vmax=res_vmax,
                )

                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=norm,
                    interpolation="nearest",
                )

            elif log_norm:

                from matplotlib.colors import LogNorm

                positive = finite & (E > 0.0)

                if not positive.any():
                    raise RuntimeError(
                        "Logarithmic normalization requires at "
                        "least one positive finite deviance."
                    )

                E_plot = np.where(positive, E, np.nan)

                if res_vmin <= 0.0:
                    raise ValueError(
                        "vmin must be strictly positive when "
                        "log_norm=True."
                    )

                if res_vmax <= res_vmin:
                    raise ValueError(
                        "vmax must be greater than vmin."
                    )

                im_err = ax_err.imshow(
                    E_plot,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    norm=LogNorm(vmin=res_vmin, vmax=res_vmax),
                    interpolation="nearest",
                )

            else:

                im_err = ax_err.imshow(
                    E,
                    origin="upper",
                    aspect="auto",
                    cmap=cmap_obj,
                    vmin=res_vmin,
                    vmax=res_vmax,
                    interpolation="nearest",
                )

            if annotate_above is not None:

                for i, j in np.argwhere(finite & (E > annotate_above)):

                    ax_err.text(
                        j,
                        i,
                        f"{E[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=annotate_fontsize,
                        fontweight="bold",
                    )

            xticks = np.arange(len(x_vals))
            yticks = np.arange(len(nH_vals))

            x_labels = [
                f"{value * x_display_scale:.4f}"
                for value in x_vals[::tick_stride]
            ]

            nh_labels = [
                _latex_sci(value) for value in nH_vals[::tick_stride]
            ]

            ax_err.set_xticks(xticks[::tick_stride])

            ax_err.set_xticklabels(
                x_labels, rotation=45, ha="right", fontsize=10
            )

            ax_err.set_yticks(yticks[::tick_stride])
            ax_err.set_yticklabels(nh_labels, fontsize=10)

            ax_err.set_xlabel(r"$x$")
            ax_err.set_ylabel(r"$N_\mathrm{H}$ [cm$^{-2}$]")

            ax_err.tick_params(axis="both", which="both", length=0)

            if title:
                ax_err.set_title(title, pad=8)

            cbar_err = fig.colorbar(
                im_err, ax=ax_err, pad=0.02, extend=extend
            )

            cbar_err.set_label(cbar_label)

            cbar_err.ax.tick_params(direction="in", length=3)
            cbar_err.outline.set_linewidth(0.8)

            if diverging:

                ticks = cbar_err.get_ticks()

                if vcenter not in ticks:
                    ticks = np.sort(np.append(ticks, vcenter))

                cbar_err.set_ticks(
                    ticks[(ticks >= res_vmin) & (ticks <= res_vmax)]
                )

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------------
    # 12. Return values.
    # --------------------------------------------------------------
    if return_stats:
        return E, x_vals, nH_vals, stats

    return E, x_vals, nH_vals


def poisson_deviance_pairwise_mean_samples_npz(
    npz_path,
    *,
    n_annuli: int = 24,
    max_samples: int | None = None,
    cmap: str = "magma",
    vmin=None,
    vmax=None,
    bins: int = 60,
    title: str = "",
    figsize=(10, 4.2),
    show: bool = True,
    consistency_levels=(1.0, 2.0, 3.0),
    print_stats: bool = True,
    return_stats: bool = False,
):
    """
    Mean radial reduced two-sample Poisson deviance over all unique pairs
    of samples generated at ONE fixed (x, nH) parameter point.

    This is the single-condition counterpart of
    ``poisson_deviance_heatmap_pairwise_mean_npz``. Instead of N archives
    each holding one matrix per grid cell, a single archive (as written
    to ``test_poisson_npz/condition_*.npz``) holds N independent samples
    at one condition:

        samples : (N, H, W) or (N, 1, H, W) float counts
        x, nH   : scalars

    For every unique unordered pair (a, b), a < b, the reduced deviance

        D_red(S_a, S_b)

    is evaluated exactly as ``radial_reduced_poisson_deviance_abs`` with
    ``data=S_a``, ``model=S_b`` (default options: no bin filter, no
    normalization, annuli with zero data-side counts excluded, dof = K).
    The statistic is the mean over all C(N, 2) pairs.

    Each sample is reduced to its annular count profile once, and the
    pairwise deviances are then evaluated in vectorized form, so all
    C(256, 2) = 32640 pairs take a fraction of a second.

    Parameters
    ----------
    npz_path : str or Path
        Path to one condition archive containing ``samples``, ``x`` and
        ``nH``.

    n_annuli : int
        Number of radial annuli.

    max_samples : int, optional
        Use only the first ``max_samples`` samples (e.g. to study the
        dependence on ensemble size). ``None`` uses all of them.

    vmin, vmax : float, optional
        Colour limits of the pairwise-matrix panel (default: data range).

    bins : int
        Number of histogram bins of the pairwise-value panel.

    consistency_levels : sequence of float
        Thresholds for reporting the fraction of pairs with
        D_red <= level.

    Returns
    -------
    mean_D : float
        Mean of D_red over all unique sample pairs.

    D_pair : ndarray, shape (N, N)
        Symmetric matrix of pairwise D_red values; entry [a, b] with a < b
        is D_red(data=S_a, model=S_b), mirrored into [b, a]. The diagonal
        is NaN.

    x, nH : float
        The condition of the archive.

    stats : dict, optional
        Returned only when ``return_stats=True``.
    """

    from pathlib import Path

    # --------------------------------------------------------------
    # 1. Validate options and load the archive.
    # --------------------------------------------------------------
    if n_annuli <= 0:
        raise ValueError("n_annuli must be a positive integer.")

    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2.")

    required_keys = {"samples", "x", "nH"}

    with np.load(npz_path, allow_pickle=False) as npz:

        missing_keys = required_keys.difference(npz.files)

        if missing_keys:
            raise KeyError(
                f"NPZ archive {str(npz_path)!r} is missing required "
                f"keys: {sorted(missing_keys)}"
            )

        samples = np.asarray(npz["samples"], dtype=np.float64)
        x_val = float(np.asarray(npz["x"]).item())
        nH_val = float(np.asarray(npz["nH"]).item())

    if samples.ndim == 4 and samples.shape[1] == 1:
        samples = samples[:, 0]

    if samples.ndim != 3:
        raise ValueError(
            "samples must have shape (N, H, W) or (N, 1, H, W); "
            f"got {samples.shape}."
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    n_samples, height, width = samples.shape

    if n_samples < 2:
        raise ValueError("At least 2 samples are required.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("samples contain NaN or infinite values.")

    if np.any(samples < 0.0):
        raise ValueError("samples contain negative counts.")

    label = Path(npz_path).stem

    # --------------------------------------------------------------
    # 2. Annular count profiles, using the same radial binning as
    #    radial_reduced_poisson_deviance_abs.
    # --------------------------------------------------------------
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    yy, xx = np.indices((height, width))
    radius_pixels = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    edges = np.linspace(0.0, radius_pixels.max(), n_annuli + 1)

    annulus = np.clip(
        np.digitize(radius_pixels.ravel(), edges, right=False) - 1,
        0,
        n_annuli - 1,
    )

    # One-hot (H*W, n_annuli) projection -> profiles (N, n_annuli).
    projector = np.zeros((height * width, n_annuli), dtype=np.float64)
    projector[np.arange(height * width), annulus] = 1.0

    profiles = samples.reshape(n_samples, -1) @ projector

    has_pixels = projector.sum(axis=0) > 0.0

    # --------------------------------------------------------------
    # 3. Vectorized pairwise deviance.
    #
    #    For pair (a, b): R = profile a (data side), G = profile b.
    #    Valid annuli: N_pixels > 0 and R > 0. With 0 * log(0) = 0,
    #
    #        D = 2 sum_i [R log(2R / (R+G)) + G log(2G / (R+G))],
    #        D_red = D / K.
    # --------------------------------------------------------------
    ia, ib = np.triu_indices(n_samples, k=1)

    R = profiles[ia]
    G = profiles[ib]

    valid = has_pixels[None, :] & (R > 0.0)

    RG = R + G

    with np.errstate(divide="ignore", invalid="ignore"):
        term_R = np.where(R > 0.0, R * np.log(2.0 * R / RG), 0.0)
        term_G = np.where(G > 0.0, G * np.log(2.0 * G / RG), 0.0)

    D = 2.0 * np.sum(np.where(valid, term_R + term_G, 0.0), axis=1)
    K = valid.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        pair_values = np.where(K > 0, D / K, np.nan)

    D_pair = np.full((n_samples, n_samples), np.nan, dtype=np.float64)
    D_pair[ia, ib] = pair_values
    D_pair[ib, ia] = pair_values

    # --------------------------------------------------------------
    # 4. Summary statistics.
    # --------------------------------------------------------------
    v = pair_values[np.isfinite(pair_values)]
    n_pairs = int(pair_values.size)
    n_finite = int(v.size)

    if n_finite == 0:
        raise RuntimeError("No finite pairwise deviance values.")

    mean_D = float(v.mean())
    std_D = float(v.std(ddof=1)) if n_finite > 1 else float("nan")
    median_D = float(np.median(v))

    q05, q25, q75, q95 = (
        float(np.percentile(v, p)) for p in (5, 25, 75, 95)
    )

    per_sample_mean = np.nanmean(D_pair, axis=1)

    fractions = {}

    for level in consistency_levels:
        count = int(np.count_nonzero(v <= level))
        fractions[float(level)] = (count, count / n_finite)

    stats = {
        "label": label,
        "x": x_val,
        "nH": nH_val,
        "n_samples": n_samples,
        "n_pairs": n_pairs,
        "n_finite": n_finite,
        "dof_per_pair": K,
        "mean": mean_D,
        "std": std_D,
        "median": median_D,
        "p05": q05,
        "p25": q25,
        "p75": q75,
        "p95": q95,
        "min": float(v.min()),
        "max": float(v.max()),
        "trimmed_mean_10": _trimmed_mean(v, 0.10),
        "fraction_below": fractions,
        "per_sample_mean": per_sample_mean,
        "worst_sample": int(np.nanargmax(per_sample_mean)),
        "statistic": (
            "mean, over all unique sample pairs at one condition, of "
            "the absolute two-sample radial Poisson deviance"
        ),
    }

    if print_stats:

        width_line = 66
        header = title if title else label

        print("=" * width_line)
        print(f" Summary: {header}")
        print("=" * width_line)
        print(f" Condition           : x={x_val:.4g}, nH={nH_val:.3e}")
        print(
            f" Samples / pairs     : {n_samples} / {n_pairs} "
            f"({n_finite} finite)"
        )
        print(f" Annuli per image    : {n_annuli}")
        print(f" Mean D_red          : {mean_D:.4f}")
        print(f" Std (ddof=1)        : {std_D:.4f}")
        print(f" Median D_red        : {median_D:.4f}")
        print(f" IQR                 : {q25:.4f} - {q75:.4f}")
        print(f" 5-95%               : {q05:.4f} - {q95:.4f}")
        print(
            f" Min / Max           : "
            f"{stats['min']:.4f} / {stats['max']:.4f}"
        )

        for level in consistency_levels:
            count, frac = fractions[float(level)]
            print(
                f" Fraction D_red <= {level:<5.2f}: "
                f"{100.0 * frac:6.2f}%  ({count}/{n_finite})"
            )

        print(
            f" Worst sample        : #{stats['worst_sample']} "
            f"(mean D_red vs others = "
            f"{per_sample_mean[stats['worst_sample']]:.4f})"
        )
        print("=" * width_line)

    # --------------------------------------------------------------
    # 5. Plot: pairwise matrix + distribution of pair values.
    # --------------------------------------------------------------
    if show:

        with plt.rc_context(JOURNAL_RC):

            fig, (ax_mat, ax_hist) = plt.subplots(
                1, 2, figsize=figsize,
                gridspec_kw={"width_ratios": [1.1, 1.0]},
            )

            im = ax_mat.imshow(
                D_pair,
                origin="upper",
                cmap=cmap,
                vmin=vmin if vmin is not None else float(v.min()),
                vmax=vmax if vmax is not None else float(v.max()),
                interpolation="nearest",
            )

            ax_mat.set_xlabel("sample $b$")
            ax_mat.set_ylabel("sample $a$")

            cbar = fig.colorbar(im, ax=ax_mat, pad=0.02)
            cbar.set_label(r"$D_{\mathrm{red}}(a, b)$")
            cbar.ax.tick_params(direction="in", length=3)

            ax_hist.hist(v, bins=bins, color="0.6", edgecolor="0.3",
                         linewidth=0.5)
            ax_hist.axvline(mean_D, color="#D55E00", linewidth=1.2,
                            label=rf"mean = {mean_D:.3f}")
            ax_hist.axvline(median_D, color="#0072B2", linewidth=1.2,
                            linestyle="--",
                            label=rf"median = {median_D:.3f}")
            ax_hist.set_xlabel(r"$D_{\mathrm{red}}$")
            ax_hist.set_ylabel("pairs")
            ax_hist.legend(fontsize=9)

            fig.suptitle(
                title if title else
                rf"$x={x_val:.4g}$, "
                rf"$N_\mathrm{{H}}=${_latex_sci(nH_val)} cm$^{{-2}}$ "
                rf"({n_samples} samples, {n_pairs} pairs)"
            )

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------------
    # 6. Return values.
    # --------------------------------------------------------------
    if return_stats:
        return mean_D, D_pair, x_val, nH_val, stats

    return mean_D, D_pair, x_val, nH_val


def matrix_sum_heatmap(
    npz_path: str,
    *,
    decimals: int = 6,
    cmap: str = "magma",
    tick_stride: int = 1,
    figsize=(7, 5),
    show: bool = True,
):
    """
    Plot a heatmap where each cell stores the total photon count (sum of matrix entries)
    for the corresponding (x, nH) matrix inside a single NPZ file.
    """
    npz = np.load(npz_path)
    xs = np.round(npz["xs"].astype(float), decimals)
    nHs = np.round(npz["nHs"].astype(float), decimals)
    mats = npz["mats"]

    x_vals = np.unique(xs)
    nH_vals = np.unique(nHs)

    ix = {val: i for i, val in enumerate(x_vals)}
    inh = {val: j for j, val in enumerate(nH_vals)}

    totals = np.full((len(nH_vals), len(x_vals)), np.nan, dtype=np.float64)

    for x, nH, mat in zip(xs, nHs, mats):
        mat2d = np.squeeze(mat)
        totals[inh[nH], ix[x]] = float(np.nansum(mat2d))

    if show:
        plt.figure(figsize=figsize)
        plt.imshow(
            totals,
            origin="upper",
            aspect="auto",
            cmap=cmap,
        )

        xticks = np.arange(len(x_vals))
        yticks = np.arange(len(nH_vals))
        plt.xticks(
            xticks[::tick_stride],
            [str(v) for v in x_vals[::tick_stride]],
            rotation=45,
        )
        plt.yticks(
            yticks[::tick_stride],
            [str(v) for v in nH_vals[::tick_stride]],
        )

        plt.xlabel("x (Energy)")
        plt.ylabel("nH (Column Density)")
        plt.title("Total Photon Count Heatmap")

        cbar = plt.colorbar()
        cbar.set_label("Total Count")

        plt.tight_layout()
        plt.show()

    return totals, x_vals, nH_vals


"""
Compare two error grids (cDDPM vs interpolation) inside and outside the
removed region of a holed training grid.

Usage
-----
E_model,  xs, nHs = error_heatmap_between_npz(..., show=False, print_stats=False)
E_interp, xs, nHs = error_heatmap_between_npz(..., show=False, print_stats=False)

compare_hole(E_model, E_interp, xs, nHs)

E has shape (len(nH_vals), len(x_vals)); row 0 is the LOWEST nH, matching
origin="upper" in the heatmap (lowest nH drawn at the top).
"""

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy optional
    wilcoxon = None


# Test conditions inside the central 13x13 hole (training indices 9-21).
# These are the physical values, not indices -- robust to axis ordering.
HOLE_X = (0.33, 0.64)          # 0.3319 ... 0.6287
HOLE_NH = (4.0e21, 1.85e22)    # 4.10e21 ... 1.81e22


def _summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, trim=np.nan,
                    gmean=np.nan, frac2=np.nan, mx=np.nan)
    lo, hi = np.percentile(v, [10, 90])
    keep = v[(v >= lo) & (v <= hi)]
    pos = v[v > 0]
    return dict(
        n=v.size,
        mean=v.mean(),
        median=np.median(v),
        trim=keep.mean() if keep.size else np.nan,
        gmean=np.exp(np.mean(np.log(pos))) if pos.size else np.nan,
        frac2=np.mean(v <= 2.0),
        mx=v.max(),
    )


def build_hole_mask(x_vals, nH_vals, x_range=HOLE_X, nH_range=HOLE_NH,
                    x_display_scale=1e-4):
    """Boolean mask of shape (len(nH_vals), len(x_vals)); True = inside hole."""
    xd = np.asarray(x_vals, float) * x_display_scale
    nh = np.asarray(nH_vals, float)
    mx = (xd >= x_range[0]) & (xd <= x_range[1])
    mn = (nh >= nH_range[0]) & (nh <= nH_range[1])
    return np.outer(mn, mx)


def compare_hole(E_model, E_interp, x_vals, nH_vals,
                 x_range=HOLE_X, nH_range=HOLE_NH, x_display_scale=1e-4,
                 label_model="cDDPM", label_interp="Interpolation",
                 full_grid_reference=1.43, n_boot=10000, seed=0,
                 print_cells=True):
    E_model = np.asarray(E_model, float)
    E_interp = np.asarray(E_interp, float)

    if E_model.shape != E_interp.shape:
        raise ValueError(f"shape mismatch: {E_model.shape} vs {E_interp.shape}")

    mask = build_hole_mask(x_vals, nH_vals, x_range, nH_range, x_display_scale)
    if mask.sum() == 0:
        raise ValueError("Hole mask selected zero cells; check x_range/nH_range.")

    xd = np.asarray(x_vals, float) * x_display_scale
    nh = np.asarray(nH_vals, float)

    W = 74
    print("=" * W)
    print(f" Hole: x in [{xd[mask.any(0)].min():.4f}, {xd[mask.any(0)].max():.4f}]"
          f"   nH in [{nh[mask.any(1)].min():.2e}, {nh[mask.any(1)].max():.2e}]")
    print(f" {int(mask.sum())} in-hole cells, {int((~mask).sum())} control cells")
    print("=" * W)

    # ---------------- summary table ----------------
    hdr = f"{'':<14}{'n':>4}{'mean':>9}{'median':>9}{'trim10':>9}{'gmean':>9}{'<=2':>8}{'max':>9}"
    print(hdr)
    print("-" * W)
    rows = {}
    for region, m in (("IN-HOLE", mask), ("CONTROL", ~mask)):
        for lbl, E in ((label_interp, E_interp), (label_model, E_model)):
            s = _summ(E[m])
            rows[(region, lbl)] = s
            print(f"{region[:3]+' '+lbl:<14}{s['n']:>4}{s['mean']:>9.3f}"
                  f"{s['median']:>9.3f}{s['trim']:>9.3f}{s['gmean']:>9.3f}"
                  f"{100*s['frac2']:>7.1f}%{s['mx']:>9.2f}")
        print("-" * W)

    # ---------------- health check ----------------
    ctrl = rows[("CONTROL", label_model)]
    print(f" HEALTH CHECK: {label_model} control mean = {ctrl['mean']:.3f} "
          f"(full-grid reference {full_grid_reference})")
    if ctrl["mean"] > 2.5 * full_grid_reference:
        print("   ** control is badly inflated -- the model itself is weaker.")
        print("   ** any in-hole 'win' may be two failure modes, not gap-bridging.")
    elif ctrl["mean"] > 1.5 * full_grid_reference:
        print("   ~ control somewhat inflated; interpret the in-hole result with care.")
    else:
        print("   OK -- model looks healthy outside the hole.")
    print("=" * W)

    # ---------------- paired test ----------------
    out = {"summary": rows, "mask": mask}
    for region, m in (("IN-HOLE", mask), ("CONTROL", ~mask)):
        a, b = E_model[m], E_interp[m]
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]                      # negative => model better
        if d.size < 5:
            continue
        rng = np.random.default_rng(seed)
        boot = np.median(rng.choice(d, (n_boot, d.size), replace=True), axis=1)
        ci = np.percentile(boot, [2.5, 97.5])
        print(f" PAIRED, {region}  (d = {label_model} - {label_interp}; d<0 = model better)")
        print(f"   n={d.size}   median d = {np.median(d):+.3f}"
              f"   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"   mean d = {d.mean():+.3f}   win rate = {100*np.mean(d < 0):.1f}%")
        if wilcoxon is not None:
            try:
                st, p = wilcoxon(d)
                print(f"   Wilcoxon signed-rank: W={st:.1f}, p={p:.4g}")
            except ValueError as e:
                print(f"   Wilcoxon skipped: {e}")
        print("-" * W)
        out[region] = dict(d=d, median=np.median(d), ci=ci,
                           win=np.mean(d < 0))

    # ---------------- per-row breakdown inside hole ----------------
    ri = np.flatnonzero(mask.any(1))
    ci_ = np.flatnonzero(mask.any(0))
    print(f" IN-HOLE per-nH row means ({len(ci_)} in-hole columns)")
    print(f"   {'nH':<12}{label_interp:>16}{label_model:>16}{'winner':>10}")
    for r in ri:
        vi = E_interp[r, ci_]
        vm = E_model[r, ci_]
        w = label_model if np.nanmean(vm) < np.nanmean(vi) else label_interp
        print(f"   {nh[r]:<12.2e}{np.nanmean(vi):>16.3f}{np.nanmean(vm):>16.3f}{w:>10}")
    print("=" * W)

    # ---------------- cell-by-cell ----------------
    if print_cells:
        print(f" IN-HOLE cells  ({label_interp} / {label_model})")
        head = "   " + f"{'nH \\ x':<11}" + "".join(f"{xd[c]:>15.4f}" for c in ci_)
        print(head)
        for r in ri:
            cells = "".join(
                f"{E_interp[r,c]:>7.2f}/{E_model[r,c]:<7.2f}" for c in ci_)
            print(f"   {nh[r]:<11.2e}{cells}")
        print("=" * W)

    return out
