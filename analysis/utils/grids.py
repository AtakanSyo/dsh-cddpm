from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from .parsing import parse_parameters


__all__ = [
    "axis_index_maps",
    "plot_matrix_grid",
    "create_interpolated_npz",
    "time_interpolation_per_image",
    "create_rbf_interpolated_npz",
    "create_interpolated_npz_cubic"
]


def axis_index_maps(
    x_vals: np.ndarray,
    nH_vals: np.ndarray,
) -> Tuple[Dict[float, int], Dict[float, int]]:
    """
    Build dictionaries that map an axis value to its index for quick placement in grids.
    """
    return (
        {float(val): idx for idx, val in enumerate(x_vals)},
        {float(val): idx for idx, val in enumerate(nH_vals)},
    )

def plot_matrix_grid(
    npz_path: str,
    *,
    decimals: int = 6,
    cmap: str = "magma",
    figsize: Optional[Tuple[float, float]] = (8, 8),
    title: str = "",
    font_size: int = 11,
    x_display_scale: float = 1e4,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize every (x, nH) matrix stored in an NPZ archive as a stitched grid,
    styled for journal (MNRAS-like) figures. Each matrix becomes a tile in a
    larger canvas so axes can show the original x/nH coordinates.
    """
    with np.load(npz_path) as data:
        xs = np.round(data["xs"].astype(float), decimals)
        nHs = np.round(data["nHs"].astype(float), decimals)
        mats = data["mats"]

    x_vals = np.unique(xs)
    nH_vals = np.unique(nHs)
    ix_map, inh_map = axis_index_maps(x_vals, nH_vals)

    if len(mats) == 0:
        raise ValueError("No matrices found in NPZ archive.")

    sample_mat = np.squeeze(np.asarray(mats[0]))
    if sample_mat.ndim != 2:
        raise ValueError("Matrices must be 2-D for grid plotting.")
    H, W = sample_mat.shape

    canvas = np.zeros((len(nH_vals) * H, len(x_vals) * W), dtype=float)
    tile_filled = np.zeros((len(nH_vals), len(x_vals)), dtype=bool)

    for x_val, nH_val, mat in zip(xs, nHs, mats):
        mat2d = np.squeeze(np.asarray(mat))
        if mat2d.shape != (H, W):
            raise ValueError("All matrices must share the same (H, W) dimensions.")
        row = inh_map[float(nH_val)]
        col = ix_map[float(x_val)]
        r0, r1 = row * H, (row + 1) * H
        c0, c1 = col * W, (col + 1) * W

        finite_mask = np.isfinite(mat2d)
        if finite_mask.any():
            local_min = float(mat2d[finite_mask].min())
            local_max = float(mat2d[finite_mask].max())
            if local_max > local_min:
                norm_tile = (mat2d - local_min) / (local_max - local_min)
            else:
                norm_tile = np.zeros_like(mat2d, dtype=float)
        else:
            norm_tile = np.zeros_like(mat2d, dtype=float)

        canvas[r0:r1, c0:c1] = norm_tile
        tile_filled[row, col] = True

    missing_fill_value = 0.15
    for row in range(len(nH_vals)):
        for col in range(len(x_vals)):
            if tile_filled[row, col]:
                continue
            r0, r1 = row * H, (row + 1) * H
            c0, c1 = col * W, (col + 1) * W
            canvas[r0:r1, c0:c1] = missing_fill_value

    imshow_kwargs = {"cmap": cmap, "aspect": "auto", "vmin": 0.0, "vmax": 1.0}

    if figsize is None:
        figsize = (8, 8)

    def _latex_sci(value: float) -> str:
        """Format 1.08e21 -> $1.08\\times10^{21}$."""
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"

    journal_rc = {
        # Serif text + STIX math to match MNRAS/Times body font
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        # Ticks: inward, on all four sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Thin frame
        "axes.linewidth": 0.8,
        # Vector-friendly output defaults
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }

    with plt.rc_context(journal_rc):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(canvas, **imshow_kwargs)
        if title:
            ax.set_title(title, fontsize=font_size, pad=8)

        for row in range(len(nH_vals)):
            for col in range(len(x_vals)):
                if tile_filled[row, col]:
                    continue
                r0, c0 = row * H, col * W
                ax.text(
                    c0 + W / 2,
                    r0 + H / 2,
                    "?",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=max(8, int(font_size * 0.85)),
                    fontweight="bold",
                )

        xticks = [(c + 0.5) * W for c in range(len(x_vals))]
        x_labels = [f"{(x / x_display_scale):.4f}" for x in x_vals]
        ax.set_xticks(xticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=font_size - 1)
        ax.set_xlabel(r"$x$", fontsize=font_size)

        yticks = [(r + 0.5) * H for r in range(len(nH_vals))]
        y_labels = [_latex_sci(nh) for nh in nH_vals]
        ax.set_yticks(yticks)
        ax.set_yticklabels(y_labels, fontsize=font_size - 1)
        ax.set_ylabel(r"$N_\mathrm{H}$ [cm$^{-2}$]", fontsize=font_size)

        # Keep the frame (journal figures are boxed), but suppress tick marks
        # at tile centres since they label categories, not a continuous axis
        ax.tick_params(axis="both", which="both", length=0)

        plt.subplots_adjust(left=0.16, right=0.96, top=0.94, bottom=0.14)

    return fig, ax


# LEGACE INTERPOLATION 1.0
def create_interpolated_npz_LEGACY(
    training_npz_path: Union[str, Path],
    testing_npz_path: Union[str, Path],
    output_npz_path: Union[str, Path],
    *,
    decimals: int = 10,
    allow_extrapolation: bool = True,
    compressed: bool = True,
) -> None:
    training_npz_path = Path(training_npz_path)
    testing_npz_path = Path(testing_npz_path)
    output_npz_path = Path(output_npz_path)

    def load_input(path: Path, *, require_mats: bool) -> dict[str, np.ndarray]:

        if path.is_dir():
            npy_paths = sorted(path.glob("*.npy"))

            if not npy_paths:
                raise FileNotFoundError(
                    f"No .npy files found in directory: {path}"
                )

            xs: list[float] = []
            nHs: list[float] = []
            mats: list[np.ndarray] = []

            for npy_path in npy_paths:
                x, nH = parse_parameters(npy_path.name)

                if x is None or nH is None:
                    raise ValueError(
                        f"Could not parse x and nH from filename: "
                        f"{npy_path.name}"
                    )

                xs.append(float(x))
                nHs.append(float(nH))

                if require_mats:
                    mats.append(np.load(npy_path))

            result: dict[str, np.ndarray] = {
                "xs": np.asarray(xs, dtype=np.float64),
                "nHs": np.asarray(nHs, dtype=np.float64),
            }

            if require_mats:
                result["mats"] = np.asarray(mats)

            return result

        if not path.is_file():
            raise FileNotFoundError(f"Input path not found: {path}")

        with np.load(path) as data:
            required_keys = (
                {"xs", "nHs", "mats"}
                if require_mats
                else {"xs", "nHs"}
            )

            missing = required_keys.difference(data.files)

            if missing:
                raise KeyError(
                    f"{path} is missing required keys: {sorted(missing)}"
                )

            return {
                key: np.asarray(data[key])
                for key in required_keys
            }

    # ------------------------------------------------------------------
    # Load training data.
    # ------------------------------------------------------------------
    training_data = load_input(
        training_npz_path,
        require_mats=True,
    )

    train_xs = np.asarray(
        training_data["xs"],
        dtype=np.float64,
    ).reshape(-1)

    train_nHs = np.asarray(
        training_data["nHs"],
        dtype=np.float64,
    ).reshape(-1)

    train_mats = np.asarray(training_data["mats"])

    # ------------------------------------------------------------------
    # Load target coordinates from the testing dataset.
    # ------------------------------------------------------------------
    testing_data = load_input(
        testing_npz_path,
        require_mats=False,
    )

    test_xs = np.asarray(
        testing_data["xs"],
        dtype=np.float64,
    ).reshape(-1)

    test_nHs = np.asarray(
        testing_data["nHs"],
        dtype=np.float64,
    ).reshape(-1)

    # ------------------------------------------------------------------
    # Validate coordinate and matrix arrays.
    # ------------------------------------------------------------------
    if (
        len(train_xs) != len(train_nHs)
        or len(train_xs) != len(train_mats)
    ):
        raise ValueError(
            "Training arrays must have equal lengths: "
            f"len(xs)={len(train_xs)}, "
            f"len(nHs)={len(train_nHs)}, "
            f"len(mats)={len(train_mats)}."
        )

    if len(test_xs) != len(test_nHs):
        raise ValueError(
            "Testing coordinate arrays must have equal lengths: "
            f"len(xs)={len(test_xs)}, "
            f"len(nHs)={len(test_nHs)}."
        )

    if len(train_mats) == 0:
        raise ValueError("The training dataset contains no matrices.")

    if len(test_xs) == 0:
        raise ValueError(
            "The testing dataset contains no requested coordinates."
        )

    if not np.all(np.isfinite(train_xs)):
        raise ValueError(
            "Training x coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(train_nHs)):
        raise ValueError(
            "Training nH coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(test_xs)):
        raise ValueError(
            "Testing x coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(test_nHs)):
        raise ValueError(
            "Testing nH coordinates contain non-finite values."
        )

    if np.any(train_nHs <= 0) or np.any(test_nHs <= 0):
        raise ValueError(
            "All nH values must be positive for log10 interpolation."
        )

    # ------------------------------------------------------------------
    # Convert all training matrices to a common 2-D shape.
    # ------------------------------------------------------------------
    squeezed_training_mats: list[np.ndarray] = []

    for index, matrix in enumerate(train_mats):
        matrix_2d = np.squeeze(np.asarray(matrix))

        if matrix_2d.ndim != 2:
            raise ValueError(
                f"Training matrix {index} has original shape "
                f"{np.asarray(matrix).shape}. After removing singleton "
                f"dimensions, its shape is {matrix_2d.shape}; expected 2-D."
            )

        matrix_2d = matrix_2d.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(matrix_2d)):
            raise ValueError(
                f"Training matrix {index} contains NaN or infinite values."
            )

        squeezed_training_mats.append(matrix_2d)

    first_shape = squeezed_training_mats[0].shape

    for index, matrix in enumerate(squeezed_training_mats):
        if matrix.shape != first_shape:
            raise ValueError(
                "All training matrices must have the same shape. "
                f"Matrix 0 has shape {first_shape}, but matrix {index} "
                f"has shape {matrix.shape}."
            )

    train_mats = np.stack(
        squeezed_training_mats,
        axis=0,
    )

    height, width = first_shape
    matrix_shape = first_shape

    # ------------------------------------------------------------------
    # Build the complete rectangular training grid.
    # ------------------------------------------------------------------
    rounded_train_xs = np.round(
        train_xs,
        decimals,
    )

    rounded_train_nHs = np.round(
        train_nHs,
        decimals,
    )

    unique_xs_rounded = np.unique(rounded_train_xs)
    unique_nHs_rounded = np.unique(rounded_train_nHs)

    if len(unique_xs_rounded) < 2:
        raise ValueError(
            "At least two distinct training x values are required."
        )

    if len(unique_nHs_rounded) < 2:
        raise ValueError(
            "At least two distinct training nH values are required."
        )

    # Representative unrounded coordinate for each rounded value.
    x_grid = np.array(
        [
            np.mean(
                train_xs[rounded_train_xs == rounded_x]
            )
            for rounded_x in unique_xs_rounded
        ],
        dtype=np.float64,
    )

    nH_grid = np.array(
        [
            np.mean(
                train_nHs[rounded_train_nHs == rounded_nH]
            )
            for rounded_nH in unique_nHs_rounded
        ],
        dtype=np.float64,
    )

    # Sort both coordinate axes.
    x_sort_indices = np.argsort(x_grid)
    nH_sort_indices = np.argsort(nH_grid)

    x_grid = x_grid[x_sort_indices]
    nH_grid = nH_grid[nH_sort_indices]

    unique_xs_rounded = unique_xs_rounded[x_sort_indices]
    unique_nHs_rounded = unique_nHs_rounded[nH_sort_indices]

    x_index = {
        float(value): index
        for index, value in enumerate(unique_xs_rounded)
    }

    nH_index = {
        float(value): index
        for index, value in enumerate(unique_nHs_rounded)
    }

    image_grid = np.empty(
        (
            len(x_grid),
            len(nH_grid),
            height,
            width,
        ),
        dtype=np.float64,
    )

    grid_filled = np.zeros(
        (len(x_grid), len(nH_grid)),
        dtype=bool,
    )

    for x_value, nH_value, matrix in zip(
        rounded_train_xs,
        rounded_train_nHs,
        train_mats,
    ):
        ix = x_index[float(x_value)]
        inh = nH_index[float(nH_value)]

        if grid_filled[ix, inh]:
            raise ValueError(
                "Duplicate training coordinate encountered at "
                f"x={x_grid[ix]:.12g}, "
                f"nH={nH_grid[inh]:.12g}. "
                "Each grid coordinate must have exactly one matrix."
            )

        image_grid[ix, inh] = matrix
        grid_filled[ix, inh] = True

    if not np.all(grid_filled):
        missing_coordinates = np.argwhere(~grid_filled)

        preview = [
            (
                float(x_grid[ix]),
                float(nH_grid[inh]),
            )
            for ix, inh in missing_coordinates[:10]
        ]

        raise ValueError(
            "The training data do not form a complete rectangular grid. "
            f"Missing {len(missing_coordinates)} coordinate combinations. "
            f"First missing coordinates: {preview}"
        )

    log_nH_grid = np.log10(nH_grid)

    # ------------------------------------------------------------------
    # Find bounding points for interpolation or extrapolation.
    # ------------------------------------------------------------------
    def bounding_indices(
        grid: np.ndarray,
        value: float,
        *,
        coordinate_name: str,
    ) -> tuple[int, int]:
        """
        Return the two grid indices used for interpolation or extrapolation.

        Exact match:
            return the same index twice.

        Inside the range:
            return the two surrounding points.

        Below the range:
            return the first two points.

        Above the range:
            return the last two points.
        """
        exact_matches = np.flatnonzero(
            np.isclose(
                grid,
                value,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        if len(exact_matches) > 0:
            exact_index = int(exact_matches[0])
            return exact_index, exact_index

        insertion_index = int(
            np.searchsorted(grid, value)
        )

        if insertion_index == 0:
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {coordinate_name}={value:.12g} is below "
                    f"the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )

            return 0, 1

        if insertion_index == len(grid):
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {coordinate_name}={value:.12g} is above "
                    f"the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )

            return len(grid) - 2, len(grid) - 1

        return insertion_index - 1, insertion_index

    # ------------------------------------------------------------------
    # Generate each requested matrix and its propagated variance.
    # ------------------------------------------------------------------
    interpolated_mats = np.empty(
        (
            len(test_xs),
            height,
            width,
        ),
        dtype=np.float64,
    )

    variance_mats = np.empty_like(interpolated_mats)   # NEW: V matrices

    extrapolated_flags = np.zeros(
        len(test_xs),
        dtype=bool,
    )

    for test_index, (x_star, nH_star) in enumerate(
        zip(test_xs, test_nHs)
    ):
        log_nH_star = np.log10(nH_star)

        ix0, ix1 = bounding_indices(
            x_grid,
            x_star,
            coordinate_name="x",
        )

        inh0, inh1 = bounding_indices(
            log_nH_grid,
            log_nH_star,
            coordinate_name="log10(nH)",
        )

        # Fractional position along x.
        if ix0 == ix1:
            u = 0.0
        else:
            u = (
                (x_star - x_grid[ix0])
                / (x_grid[ix1] - x_grid[ix0])
            )

        # Fractional position along log10(nH).
        if inh0 == inh1:
            v = 0.0
        else:
            v = (
                (log_nH_star - log_nH_grid[inh0])
                / (
                    log_nH_grid[inh1]
                    - log_nH_grid[inh0]
                )
            )

        x_is_extrapolated = (
            x_star < x_grid[0]
            or x_star > x_grid[-1]
        )

        nH_is_extrapolated = (
            log_nH_star < log_nH_grid[0]
            or log_nH_star > log_nH_grid[-1]
        )

        is_extrapolated = (
            x_is_extrapolated
            or nH_is_extrapolated
        )

        extrapolated_flags[test_index] = is_extrapolated

        if is_extrapolated:
            print(
                f"Extrapolating point {test_index}: "
                f"x={x_star:.6g}, "
                f"nH={nH_star:.6e}, "
                f"u={u:.4f}, "
                f"v={v:.4f}"
            )

        image_00 = image_grid[ix0, inh0]

        # Exact match along both coordinates.
        # The "interpolant" is a raw Poisson image: V = I (full variance).
        if ix0 == ix1 and inh0 == inh1:
            generated = image_00.copy()
            variance = image_00.copy()

        # Exact x match; interpolate/extrapolate only along nH.
        elif ix0 == ix1:
            image_01 = image_grid[ix0, inh1]

            generated = (
                (1.0 - v) * image_00
                + v * image_01
            )

            variance = (
                (1.0 - v) ** 2 * image_00
                + v ** 2 * image_01
            )

        # Exact nH match; interpolate/extrapolate only along x.
        elif inh0 == inh1:
            image_10 = image_grid[ix1, inh0]

            generated = (
                (1.0 - u) * image_00
                + u * image_10
            )

            variance = (
                (1.0 - u) ** 2 * image_00
                + u ** 2 * image_10
            )

        # Full bilinear interpolation/extrapolation.
        else:
            image_10 = image_grid[ix1, inh0]
            image_01 = image_grid[ix0, inh1]
            image_11 = image_grid[ix1, inh1]

            weight_00 = (1.0 - u) * (1.0 - v)
            weight_10 = u * (1.0 - v)
            weight_01 = (1.0 - u) * v
            weight_11 = u * v

            generated = (
                weight_00 * image_00
                + weight_10 * image_10
                + weight_01 * image_01
                + weight_11 * image_11
            )

            variance = (
                weight_00 ** 2 * image_00
                + weight_10 ** 2 * image_10
                + weight_01 ** 2 * image_01
                + weight_11 ** 2 * image_11
            )

        # Linear extrapolation may produce negative pixel values.
        # (The variance stays non-negative by construction: squared
        # weights times non-negative training counts.)
        negative_count = int(
            np.count_nonzero(generated < 0.0)
        )

        if negative_count > 0:
            print(
                f"Warning: point {test_index} produced "
                f"{negative_count} negative pixels. "
                "Clipping them to zero."
            )

            generated = np.clip(
                generated,
                0.0,
                None,
            )

        interpolated_mats[test_index] = generated
        variance_mats[test_index] = variance

    # ------------------------------------------------------------------
    # Save output.
    # ------------------------------------------------------------------
    output_npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_function = (
        np.savez_compressed
        if compressed
        else np.savez
    )

    save_function(
        output_npz_path,
        xs=test_xs,
        nHs=test_nHs,
        mats=interpolated_mats,
        Vs=variance_mats,
        extrapolated=extrapolated_flags,
    )

    print()
    print(
        f"Saved {len(interpolated_mats)} matrices to "
        f"'{output_npz_path}'."
    )
    print("Output type: deterministic interpolated/extrapolated matrices")
    print(f"Matrix shape: {matrix_shape}")
    print(
        f"Extrapolated points: "
        f"{int(np.count_nonzero(extrapolated_flags))}/"
        f"{len(extrapolated_flags)}"
    )
    print(
        "Parameter treatment: linear in x, linear in log10(nH), "
        "linear in pixel intensity."
    )
    print(
        "Propagated variance matrices saved under key 'Vs' "
        "(pass as V to the radial chi^2 function)."
    )


from pathlib import Path
from typing import Union

import numpy as np


def create_interpolated_npz(
    training_npz_path: Union[str, Path],
    testing_npz_path: Union[str, Path],
    output_npz_path: Union[str, Path],
    *,
    decimals: int = 10,
    allow_extrapolation: bool = True,
    compressed: bool = True,
    max_expand: int = 16,
) -> None:
    """
    Generate pixel-wise bilinearly interpolated or extrapolated matrices at
    the parameter coordinates contained in a testing dataset.

    Inputs may be NPZ archives or directories containing NPY matrices whose
    filenames can be interpreted by ``parse_parameters``.

    Expected training NPZ keys
    --------------------------
    xs   : array of x coordinates, shape (N_train,)
    nHs  : array of nH coordinates, shape (N_train,)
    mats : matrices, shape (N_train, H, W), or equivalent with singleton axes

    Expected testing NPZ keys
    -------------------------
    xs   : requested x coordinates, shape (N_test,)
    nHs  : requested nH coordinates, shape (N_test,)

    Any ``mats`` stored in the testing archive are ignored. Only the requested
    coordinates are used.

    Interpolation / extrapolation
    -----------------------------
    - Linear in x
    - Linear in log10(nH)
    - Linear in pixel intensity

    Inside the training domain, the four surrounding images are used.

    Outside the training domain:
    - values below an axis range use the first two grid points;
    - values above an axis range use the last two grid points.

    The same bilinear formula is then applied:

        I_out =
            (1-u)(1-v) I00
            + u(1-v)   I10
            + (1-u)v   I01
            + uv       I11

    For interpolation, u and v lie in [0, 1].

    For extrapolation, u or v may lie outside [0, 1].

    Incomplete ("holed") training grids
    -----------------------------------
    The training coordinates need not form a complete rectangular grid.
    Removing a block of training images leaves both coordinate axes intact --
    only specific (x, nH) combinations disappear -- so the immediate
    bracketing cell of a test point may have missing corners.

    In that case the interpolation cell is widened outward, one grid step at
    a time, until all four corners correspond to training images that exist.
    The tightest such cell is used. The bilinear formula and the variance
    propagation below are then applied unchanged on the widened cell; only
    the cell spacing differs, so the interpolant remains a convex combination
    of real training images and u, v stay in [0, 1].

    On a complete grid no widening is ever triggered and the result is
    bit-identical to the rectangular-grid behaviour.

    Propagated variance (V)
    -----------------------
    Because the training images are Poisson realizations
    (Var[I_ij] = I_ij per pixel), the interpolant inherits the propagated
    per-pixel variance

        V_out =
            (1-u)^2 (1-v)^2 I00
            + u^2 (1-v)^2   I10
            + (1-u)^2 v^2   I01
            + u^2 v^2       I11,

    i.e. the same linear combination with squared weights. This is computed
    for every generated matrix and saved under the NPZ key ``Vs``, to be
    passed as the ``V`` argument of the two-sided radial chi^2 function.
    Squared weights are also correct for extrapolation (weights outside
    [0, 1] enter squared, so V_out remains non-negative). For an exact
    coordinate match the "interpolant" is a raw Poisson image and
    V_out = I_out (full Poisson variance), which the formula reproduces.

    Output NPZ keys
    ---------------
    xs           : requested testing x coordinates
    nHs          : requested testing nH coordinates
    mats         : generated matrices, shape (N_test, H, W)
    Vs           : propagated per-pixel variance matrices, shape (N_test, H, W)
    extrapolated : boolean flags, shape (N_test,)
    widened      : boolean flags, shape (N_test,); True where the cell had to
                   grow because of missing training images
    cell_span    : integer cell width in grid steps, shape (N_test, 2), as
                   (x steps, log10(nH) steps). 1 means immediate neighbours.

    Parameters
    ----------
    training_npz_path
        NPZ containing training coordinates and matrices, or a directory
        containing training NPY files.
    testing_npz_path
        NPZ containing target testing coordinates, or a directory containing
        testing NPY files whose filenames contain the coordinates.
    output_npz_path
        Destination NPZ path.
    decimals
        Number of decimal places used when grouping nominally identical
        training coordinates.
    allow_extrapolation
        If True, points outside the training domain are linearly extrapolated.
        If False, an error is raised for points outside the training domain.
    compressed
        If True, save with np.savez_compressed. Otherwise use np.savez.
    max_expand
        Maximum number of grid steps the interpolation cell may be widened
        along any single edge when bridging a gap.

    Notes
    -----
    This function expects an existing ``parse_parameters(filename)`` function
    when an input path is a directory of NPY files.
    """
    training_npz_path = Path(training_npz_path)
    testing_npz_path = Path(testing_npz_path)
    output_npz_path = Path(output_npz_path)

    def load_input(path: Path, *, require_mats: bool) -> dict[str, np.ndarray]:
        """
        Load coordinates and optionally matrices from an NPZ archive or a
        directory containing NPY files.
        """
        if path.is_dir():
            npy_paths = sorted(path.glob("*.npy"))

            if not npy_paths:
                raise FileNotFoundError(
                    f"No .npy files found in directory: {path}"
                )

            xs: list[float] = []
            nHs: list[float] = []
            mats: list[np.ndarray] = []

            for npy_path in npy_paths:
                x, nH = parse_parameters(npy_path.name)  # noqa: F821

                if x is None or nH is None:
                    raise ValueError(
                        f"Could not parse x and nH from filename: "
                        f"{npy_path.name}"
                    )

                xs.append(float(x))
                nHs.append(float(nH))

                if require_mats:
                    mats.append(np.load(npy_path))

            result: dict[str, np.ndarray] = {
                "xs": np.asarray(xs, dtype=np.float64),
                "nHs": np.asarray(nHs, dtype=np.float64),
            }

            if require_mats:
                result["mats"] = np.asarray(mats)

            return result

        if not path.is_file():
            raise FileNotFoundError(f"Input path not found: {path}")

        with np.load(path) as data:
            required_keys = (
                {"xs", "nHs", "mats"}
                if require_mats
                else {"xs", "nHs"}
            )

            missing = required_keys.difference(data.files)

            if missing:
                raise KeyError(
                    f"{path} is missing required keys: {sorted(missing)}"
                )

            return {
                key: np.asarray(data[key])
                for key in required_keys
            }

    # ------------------------------------------------------------------
    # Load training data.
    # ------------------------------------------------------------------
    training_data = load_input(
        training_npz_path,
        require_mats=True,
    )

    train_xs = np.asarray(
        training_data["xs"],
        dtype=np.float64,
    ).reshape(-1)

    train_nHs = np.asarray(
        training_data["nHs"],
        dtype=np.float64,
    ).reshape(-1)

    train_mats = np.asarray(training_data["mats"])

    # ------------------------------------------------------------------
    # Load target coordinates from the testing dataset.
    # ------------------------------------------------------------------
    testing_data = load_input(
        testing_npz_path,
        require_mats=False,
    )

    test_xs = np.asarray(
        testing_data["xs"],
        dtype=np.float64,
    ).reshape(-1)

    test_nHs = np.asarray(
        testing_data["nHs"],
        dtype=np.float64,
    ).reshape(-1)

    # ------------------------------------------------------------------
    # Validate coordinate and matrix arrays.
    # ------------------------------------------------------------------
    if (
        len(train_xs) != len(train_nHs)
        or len(train_xs) != len(train_mats)
    ):
        raise ValueError(
            "Training arrays must have equal lengths: "
            f"len(xs)={len(train_xs)}, "
            f"len(nHs)={len(train_nHs)}, "
            f"len(mats)={len(train_mats)}."
        )

    if len(test_xs) != len(test_nHs):
        raise ValueError(
            "Testing coordinate arrays must have equal lengths: "
            f"len(xs)={len(test_xs)}, "
            f"len(nHs)={len(test_nHs)}."
        )

    if len(train_mats) == 0:
        raise ValueError("The training dataset contains no matrices.")

    if len(test_xs) == 0:
        raise ValueError(
            "The testing dataset contains no requested coordinates."
        )

    if not np.all(np.isfinite(train_xs)):
        raise ValueError(
            "Training x coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(train_nHs)):
        raise ValueError(
            "Training nH coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(test_xs)):
        raise ValueError(
            "Testing x coordinates contain non-finite values."
        )

    if not np.all(np.isfinite(test_nHs)):
        raise ValueError(
            "Testing nH coordinates contain non-finite values."
        )

    if np.any(train_nHs <= 0) or np.any(test_nHs <= 0):
        raise ValueError(
            "All nH values must be positive for log10 interpolation."
        )

    # ------------------------------------------------------------------
    # Convert all training matrices to a common 2-D shape.
    # ------------------------------------------------------------------
    squeezed_training_mats: list[np.ndarray] = []

    for index, matrix in enumerate(train_mats):
        matrix_2d = np.squeeze(np.asarray(matrix))

        if matrix_2d.ndim != 2:
            raise ValueError(
                f"Training matrix {index} has original shape "
                f"{np.asarray(matrix).shape}. After removing singleton "
                f"dimensions, its shape is {matrix_2d.shape}; expected 2-D."
            )

        matrix_2d = matrix_2d.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(matrix_2d)):
            raise ValueError(
                f"Training matrix {index} contains NaN or infinite values."
            )

        squeezed_training_mats.append(matrix_2d)

    first_shape = squeezed_training_mats[0].shape

    for index, matrix in enumerate(squeezed_training_mats):
        if matrix.shape != first_shape:
            raise ValueError(
                "All training matrices must have the same shape. "
                f"Matrix 0 has shape {first_shape}, but matrix {index} "
                f"has shape {matrix.shape}."
            )

    train_mats = np.stack(
        squeezed_training_mats,
        axis=0,
    )

    height, width = first_shape
    matrix_shape = first_shape

    # ------------------------------------------------------------------
    # Build the training grid. The grid may be incomplete: removing a block
    # of training images leaves both axes intact, so grid_filled tracks
    # which (x, nH) combinations actually exist.
    # ------------------------------------------------------------------
    rounded_train_xs = np.round(
        train_xs,
        decimals,
    )

    rounded_train_nHs = np.round(
        train_nHs,
        decimals,
    )

    unique_xs_rounded = np.unique(rounded_train_xs)
    unique_nHs_rounded = np.unique(rounded_train_nHs)

    if len(unique_xs_rounded) < 2:
        raise ValueError(
            "At least two distinct training x values are required."
        )

    if len(unique_nHs_rounded) < 2:
        raise ValueError(
            "At least two distinct training nH values are required."
        )

    # Representative unrounded coordinate for each rounded value.
    x_grid = np.array(
        [
            np.mean(
                train_xs[rounded_train_xs == rounded_x]
            )
            for rounded_x in unique_xs_rounded
        ],
        dtype=np.float64,
    )

    nH_grid = np.array(
        [
            np.mean(
                train_nHs[rounded_train_nHs == rounded_nH]
            )
            for rounded_nH in unique_nHs_rounded
        ],
        dtype=np.float64,
    )

    # Sort both coordinate axes.
    x_sort_indices = np.argsort(x_grid)
    nH_sort_indices = np.argsort(nH_grid)

    x_grid = x_grid[x_sort_indices]
    nH_grid = nH_grid[nH_sort_indices]

    unique_xs_rounded = unique_xs_rounded[x_sort_indices]
    unique_nHs_rounded = unique_nHs_rounded[nH_sort_indices]

    x_index = {
        float(value): index
        for index, value in enumerate(unique_xs_rounded)
    }

    nH_index = {
        float(value): index
        for index, value in enumerate(unique_nHs_rounded)
    }

    # NaN-filled so that any accidental read of an absent node propagates
    # visibly rather than silently returning uninitialised memory.
    image_grid = np.full(
        (
            len(x_grid),
            len(nH_grid),
            height,
            width,
        ),
        np.nan,
        dtype=np.float64,
    )

    grid_filled = np.zeros(
        (len(x_grid), len(nH_grid)),
        dtype=bool,
    )

    for x_value, nH_value, matrix in zip(
        rounded_train_xs,
        rounded_train_nHs,
        train_mats,
    ):
        ix = x_index[float(x_value)]
        inh = nH_index[float(nH_value)]

        if grid_filled[ix, inh]:
            raise ValueError(
                "Duplicate training coordinate encountered at "
                f"x={x_grid[ix]:.12g}, "
                f"nH={nH_grid[inh]:.12g}. "
                "Each grid coordinate must have exactly one matrix."
            )

        image_grid[ix, inh] = matrix
        grid_filled[ix, inh] = True

    n_absent = int(np.count_nonzero(~grid_filled))

    if n_absent > 0:
        print(
            f"Incomplete training grid: {n_absent} of {grid_filled.size} "
            f"({100.0 * n_absent / grid_filled.size:.1f}%) "
            f"(x, nH) nodes are absent. Interpolation cells will be widened "
            "across the gaps."
        )

    log_nH_grid = np.log10(nH_grid)

    # ------------------------------------------------------------------
    # Find bounding points for interpolation or extrapolation.
    # ------------------------------------------------------------------
    def bounding_indices(
        grid: np.ndarray,
        value: float,
        *,
        coordinate_name: str,
    ) -> tuple[int, int]:
        """
        Return the two grid indices used for interpolation or extrapolation.

        Exact match:
            return the same index twice.

        Inside the range:
            return the two surrounding points.

        Below the range:
            return the first two points.

        Above the range:
            return the last two points.
        """
        exact_matches = np.flatnonzero(
            np.isclose(
                grid,
                value,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        if len(exact_matches) > 0:
            exact_index = int(exact_matches[0])
            return exact_index, exact_index

        insertion_index = int(
            np.searchsorted(grid, value)
        )

        if insertion_index == 0:
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {coordinate_name}={value:.12g} is below "
                    f"the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )

            return 0, 1

        if insertion_index == len(grid):
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {coordinate_name}={value:.12g} is above "
                    f"the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )

            return len(grid) - 2, len(grid) - 1

        return insertion_index - 1, insertion_index

    def corners_present(ix0: int, ix1: int, inh0: int, inh1: int) -> bool:
        """All distinct corners of the cell must be existing training images."""
        return bool(
            grid_filled[ix0, inh0]
            and grid_filled[ix1, inh0]
            and grid_filled[ix0, inh1]
            and grid_filled[ix1, inh1]
        )

    def widen_cell(
        ix0: int,
        ix1: int,
        inh0: int,
        inh1: int,
        x_star: float,
        log_nH_star: float,
    ) -> tuple[int, int, int, int]:
        """
        Grow the cell outward until every corner exists, returning the
        tightest valid cell. Never called on a complete grid.
        """
        n_x = len(x_grid)
        n_nh = len(log_nH_grid)

        lo_x = list(range(ix0, max(-1, ix0 - max_expand - 1), -1))
        hi_x = list(range(ix1, min(n_x, ix1 + max_expand + 1)))
        lo_nh = list(range(inh0, max(-1, inh0 - max_expand - 1), -1))
        hi_nh = list(range(inh1, min(n_nh, inh1 + max_expand + 1)))

        x_range = x_grid[-1] - x_grid[0]
        nh_range = log_nH_grid[-1] - log_nH_grid[0]

        best_span = np.inf
        best_cell = None

        for a0 in lo_x:
            for a1 in hi_x:
                for b0 in lo_nh:
                    for b1 in hi_nh:
                        if not corners_present(a0, a1, b0, b1):
                            continue

                        span = (
                            (x_grid[a1] - x_grid[a0]) / x_range
                            + (log_nH_grid[b1] - log_nH_grid[b0]) / nh_range
                        )

                        if span < best_span:
                            best_span = span
                            best_cell = (a0, a1, b0, b1)

        if best_cell is None:
            raise ValueError(
                "No bracketing cell with four existing training images was "
                f"found within {max_expand} grid steps of "
                f"x={x_star:.12g}, log10(nH)={log_nH_star:.12g}. "
                "The gap may reach a grid edge; increase max_expand or "
                "reduce the size of the removed region."
            )

        return best_cell

    # ------------------------------------------------------------------
    # Generate each requested matrix and its propagated variance.
    # ------------------------------------------------------------------
    interpolated_mats = np.empty(
        (
            len(test_xs),
            height,
            width,
        ),
        dtype=np.float64,
    )

    variance_mats = np.empty_like(interpolated_mats)

    extrapolated_flags = np.zeros(
        len(test_xs),
        dtype=bool,
    )

    widened_flags = np.zeros(
        len(test_xs),
        dtype=bool,
    )

    cell_span = np.ones(
        (len(test_xs), 2),
        dtype=np.int64,
    )

    for test_index, (x_star, nH_star) in enumerate(
        zip(test_xs, test_nHs)
    ):
        log_nH_star = np.log10(nH_star)

        ix0, ix1 = bounding_indices(
            x_grid,
            x_star,
            coordinate_name="x",
        )

        inh0, inh1 = bounding_indices(
            log_nH_grid,
            log_nH_star,
            coordinate_name="log10(nH)",
        )

        x_is_extrapolated = (
            x_star < x_grid[0]
            or x_star > x_grid[-1]
        )

        nH_is_extrapolated = (
            log_nH_star < log_nH_grid[0]
            or log_nH_star > log_nH_grid[-1]
        )

        is_extrapolated = (
            x_is_extrapolated
            or nH_is_extrapolated
        )

        extrapolated_flags[test_index] = is_extrapolated

        # Widen across gaps only if the immediate cell has missing corners.
        # On a complete grid this branch never runs.
        was_widened = False

        if not corners_present(ix0, ix1, inh0, inh1):
            ix0, ix1, inh0, inh1 = widen_cell(
                ix0,
                ix1,
                inh0,
                inh1,
                x_star,
                log_nH_star,
            )
            was_widened = True

        widened_flags[test_index] = was_widened
        cell_span[test_index] = (ix1 - ix0, inh1 - inh0)

        # Fractional position along x.
        if ix0 == ix1:
            u = 0.0
        else:
            u = (
                (x_star - x_grid[ix0])
                / (x_grid[ix1] - x_grid[ix0])
            )

        # Fractional position along log10(nH).
        if inh0 == inh1:
            v = 0.0
        else:
            v = (
                (log_nH_star - log_nH_grid[inh0])
                / (
                    log_nH_grid[inh1]
                    - log_nH_grid[inh0]
                )
            )

        if is_extrapolated:
            print(
                f"Extrapolating point {test_index}: "
                f"x={x_star:.6g}, "
                f"nH={nH_star:.6e}, "
                f"u={u:.4f}, "
                f"v={v:.4f}"
            )

        image_00 = image_grid[ix0, inh0]

        # Exact match along both coordinates.
        # The "interpolant" is a raw Poisson image: V = I (full variance).
        if ix0 == ix1 and inh0 == inh1:
            generated = image_00.copy()
            variance = image_00.copy()

        # Exact x match; interpolate/extrapolate only along nH.
        elif ix0 == ix1:
            image_01 = image_grid[ix0, inh1]

            generated = (
                (1.0 - v) * image_00
                + v * image_01
            )

            variance = (
                (1.0 - v) ** 2 * image_00
                + v ** 2 * image_01
            )

        # Exact nH match; interpolate/extrapolate only along x.
        elif inh0 == inh1:
            image_10 = image_grid[ix1, inh0]

            generated = (
                (1.0 - u) * image_00
                + u * image_10
            )

            variance = (
                (1.0 - u) ** 2 * image_00
                + u ** 2 * image_10
            )

        # Full bilinear interpolation/extrapolation.
        else:
            image_10 = image_grid[ix1, inh0]
            image_01 = image_grid[ix0, inh1]
            image_11 = image_grid[ix1, inh1]

            weight_00 = (1.0 - u) * (1.0 - v)
            weight_10 = u * (1.0 - v)
            weight_01 = (1.0 - u) * v
            weight_11 = u * v

            generated = (
                weight_00 * image_00
                + weight_10 * image_10
                + weight_01 * image_01
                + weight_11 * image_11
            )

            variance = (
                weight_00 ** 2 * image_00
                + weight_10 ** 2 * image_10
                + weight_01 ** 2 * image_01
                + weight_11 ** 2 * image_11
            )

        # Guard: a NaN here means an absent training node was read.
        if not np.all(np.isfinite(generated)):
            raise RuntimeError(
                f"Point {test_index} (x={x_star:.6g}, nH={nH_star:.6e}) "
                "produced non-finite pixels, which indicates that an absent "
                "training node entered the interpolation. This is a bug."
            )

        # Linear extrapolation may produce negative pixel values.
        # (The variance stays non-negative by construction: squared
        # weights times non-negative training counts.)
        negative_count = int(
            np.count_nonzero(generated < 0.0)
        )

        if negative_count > 0:
            print(
                f"Warning: point {test_index} produced "
                f"{negative_count} negative pixels. "
                "Clipping them to zero."
            )

            generated = np.clip(
                generated,
                0.0,
                None,
            )

        interpolated_mats[test_index] = generated
        variance_mats[test_index] = variance

    # ------------------------------------------------------------------
    # Save output.
    # ------------------------------------------------------------------
    output_npz_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_function = (
        np.savez_compressed
        if compressed
        else np.savez
    )

    save_function(
        output_npz_path,
        xs=test_xs,
        nHs=test_nHs,
        mats=interpolated_mats,
        Vs=variance_mats,
        extrapolated=extrapolated_flags,
        widened=widened_flags,
        cell_span=cell_span,
    )

    print()
    print(
        f"Saved {len(interpolated_mats)} matrices to "
        f"'{output_npz_path}'."
    )
    print("Output type: deterministic interpolated/extrapolated matrices")
    print(f"Matrix shape: {matrix_shape}")
    print(
        f"Extrapolated points: "
        f"{int(np.count_nonzero(extrapolated_flags))}/"
        f"{len(extrapolated_flags)}"
    )
    print(
        f"Widened cells (gap bridged): "
        f"{int(np.count_nonzero(widened_flags))}/"
        f"{len(widened_flags)}"
    )
    print(
        "Parameter treatment: linear in x, linear in log10(nH), "
        "linear in pixel intensity."
    )
    print(
        "Propagated variance matrices saved under key 'Vs' "
        "(pass as V to the radial chi^2 function)."
    )

from pathlib import Path
from typing import Union

import numpy as np


def _lagrange_weights(nodes: np.ndarray, t: float) -> np.ndarray:
    """
    Lagrange basis weights for arbitrary (non-uniform) 1-D nodes.

    Returns w with sum(w) == 1 such that  f(t) ~= sum_k w_k f(nodes_k).
    With 2 nodes this is exactly linear interpolation; with 4 nodes it is
    the cubic Lagrange interpolant. Handles the non-uniform spacing that
    arises when a stencil has to straddle a gap in the training grid.
    """
    n = len(nodes)
    w = np.ones(n, dtype=np.float64)

    for k in range(n):
        for m in range(n):
            if m == k:
                continue
            w[k] *= (t - nodes[m]) / (nodes[k] - nodes[m])

    return w


def create_interpolated_npz_cubic(
    training_npz_path: Union[str, Path],
    testing_npz_path: Union[str, Path],
    output_npz_path: Union[str, Path],
    *,
    method: str = "cubic",
    decimals: int = 10,
    allow_extrapolation: bool = True,
    compressed: bool = True,
    max_expand: int = 16,
) -> None:
    """
    Generate pixel-wise interpolated matrices at the parameter coordinates
    contained in a testing dataset, using a separable polynomial interpolant
    in x and log10(nH).

    This is a generalisation of the bilinear routine. Setting
    ``method="linear"`` reproduces the bilinear result exactly; the default
    ``method="cubic"`` uses a 4-point Lagrange stencil along each axis
    (16 training images per test condition instead of 4).

    Weights and variance
    --------------------
    Any separable polynomial interpolant is a weighted sum of training
    images,

        S_out = sum_pq  W_pq S_pq,     W_pq = wx_p * wn_q,

    with sum(W) = 1. Because the training images are independent Poisson
    realizations with Var[S] = S, the propagated per-pixel variance is the
    same combination with squared weights,

        V_out = sum_pq  W_pq^2 S_pq,

    exactly as in the bilinear case. Two consequences differ from bilinear,
    and both are reported by this routine:

      * Cubic weights can be NEGATIVE, so sum |W| > 1 and sum W^2 may
        EXCEED 1. The cubic interpolant can therefore be super-Poisson,
        V_out > S_out, whereas the bilinear interpolant is always
        sub-Poisson. Reducing interpolation bias costs variance.
      * Negative weights can produce negative pixel values, which are
        clipped to zero (as for linear extrapolation).

    Incomplete ("holed") training grids
    -----------------------------------
    A cubic stencil needs 4 x-indices and 4 log10(nH)-indices such that all
    16 combinations correspond to training images that exist. The stencil is
    built by first finding the tightest bracketing cell whose four corners
    are present (as in the bilinear routine), then extending outward by one
    node on each side, expanding further if any of the 16 combinations is
    missing. Near a grid boundary the stencil is shifted inward rather than
    extrapolated.

    Output NPZ keys
    ---------------
    xs, nHs      : requested testing coordinates
    mats         : generated matrices, shape (N_test, H, W)
    Vs           : propagated per-pixel variance, shape (N_test, H, W)
    extrapolated : bool flags, shape (N_test,)
    widened      : bool flags; True where the stencil had to grow because of
                   missing training images
    stencil_span : int, shape (N_test, 2); full stencil width in grid steps
                   along (x, log10 nH). 3 is the compact cubic stencil.
    weight_l1    : float, shape (N_test,); sum |W|. 1.0 means all weights
                   non-negative (no noise amplification).
    weight_l2    : float, shape (N_test,); sum W^2. Ratio V_out/S_out for a
                   locally flat image; > 1 indicates super-Poisson output.
    """
    method = method.lower()
    if method not in ("linear", "cubic"):
        raise ValueError("method must be 'linear' or 'cubic'.")

    n_stencil = 2 if method == "linear" else 4

    training_npz_path = Path(training_npz_path)
    testing_npz_path = Path(testing_npz_path)
    output_npz_path = Path(output_npz_path)

    def load_input(path: Path, *, require_mats: bool) -> dict[str, np.ndarray]:
        if path.is_dir():
            npy_paths = sorted(path.glob("*.npy"))
            if not npy_paths:
                raise FileNotFoundError(f"No .npy files found in directory: {path}")

            xs: list[float] = []
            nHs: list[float] = []
            mats: list[np.ndarray] = []

            for npy_path in npy_paths:
                x, nH = parse_parameters(npy_path.name)  # noqa: F821
                if x is None or nH is None:
                    raise ValueError(
                        f"Could not parse x and nH from filename: {npy_path.name}"
                    )
                xs.append(float(x))
                nHs.append(float(nH))
                if require_mats:
                    mats.append(np.load(npy_path))

            result = {
                "xs": np.asarray(xs, dtype=np.float64),
                "nHs": np.asarray(nHs, dtype=np.float64),
            }
            if require_mats:
                result["mats"] = np.asarray(mats)
            return result

        if not path.is_file():
            raise FileNotFoundError(f"Input path not found: {path}")

        with np.load(path) as data:
            required_keys = (
                {"xs", "nHs", "mats"} if require_mats else {"xs", "nHs"}
            )
            missing = required_keys.difference(data.files)
            if missing:
                raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
            return {key: np.asarray(data[key]) for key in required_keys}

    # ------------------------------------------------------------------
    # Load.
    # ------------------------------------------------------------------
    training_data = load_input(training_npz_path, require_mats=True)
    train_xs = np.asarray(training_data["xs"], dtype=np.float64).reshape(-1)
    train_nHs = np.asarray(training_data["nHs"], dtype=np.float64).reshape(-1)
    train_mats = np.asarray(training_data["mats"])

    testing_data = load_input(testing_npz_path, require_mats=False)
    test_xs = np.asarray(testing_data["xs"], dtype=np.float64).reshape(-1)
    test_nHs = np.asarray(testing_data["nHs"], dtype=np.float64).reshape(-1)

    # ------------------------------------------------------------------
    # Validate.
    # ------------------------------------------------------------------
    if len(train_xs) != len(train_nHs) or len(train_xs) != len(train_mats):
        raise ValueError(
            "Training arrays must have equal lengths: "
            f"len(xs)={len(train_xs)}, len(nHs)={len(train_nHs)}, "
            f"len(mats)={len(train_mats)}."
        )

    if len(test_xs) != len(test_nHs):
        raise ValueError(
            "Testing coordinate arrays must have equal lengths: "
            f"len(xs)={len(test_xs)}, len(nHs)={len(test_nHs)}."
        )

    if len(train_mats) == 0:
        raise ValueError("The training dataset contains no matrices.")
    if len(test_xs) == 0:
        raise ValueError("The testing dataset contains no requested coordinates.")

    for name, values in (
        ("Training x", train_xs), ("Training nH", train_nHs),
        ("Testing x", test_xs), ("Testing nH", test_nHs),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} coordinates contain non-finite values.")

    if np.any(train_nHs <= 0) or np.any(test_nHs <= 0):
        raise ValueError("All nH values must be positive for log10 interpolation.")

    # ------------------------------------------------------------------
    # Common 2-D shape.
    # ------------------------------------------------------------------
    squeezed: list[np.ndarray] = []
    for index, matrix in enumerate(train_mats):
        m2 = np.squeeze(np.asarray(matrix))
        if m2.ndim != 2:
            raise ValueError(
                f"Training matrix {index} has shape {np.asarray(matrix).shape}; "
                f"after squeezing it is {m2.shape}; expected 2-D."
            )
        m2 = m2.astype(np.float64, copy=False)
        if not np.all(np.isfinite(m2)):
            raise ValueError(f"Training matrix {index} contains NaN or infinite values.")
        squeezed.append(m2)

    first_shape = squeezed[0].shape
    for index, m2 in enumerate(squeezed):
        if m2.shape != first_shape:
            raise ValueError(
                f"All training matrices must have the same shape. Matrix 0 has "
                f"{first_shape}, matrix {index} has {m2.shape}."
            )

    train_mats = np.stack(squeezed, axis=0)
    height, width = first_shape
    matrix_shape = first_shape

    # ------------------------------------------------------------------
    # Build the (possibly incomplete) training grid.
    # ------------------------------------------------------------------
    rx = np.round(train_xs, decimals)
    rn = np.round(train_nHs, decimals)
    ux = np.unique(rx)
    un = np.unique(rn)

    if len(ux) < n_stencil or len(un) < n_stencil:
        raise ValueError(
            f"method='{method}' needs at least {n_stencil} distinct values on "
            f"each axis; got {len(ux)} in x and {len(un)} in nH."
        )

    x_grid = np.array([np.mean(train_xs[rx == v]) for v in ux], dtype=np.float64)
    nH_grid = np.array([np.mean(train_nHs[rn == v]) for v in un], dtype=np.float64)

    xs_i = np.argsort(x_grid)
    ns_i = np.argsort(nH_grid)
    x_grid, nH_grid = x_grid[xs_i], nH_grid[ns_i]
    ux, un = ux[xs_i], un[ns_i]

    x_index = {float(v): i for i, v in enumerate(ux)}
    nH_index = {float(v): j for j, v in enumerate(un)}

    image_grid = np.full((len(x_grid), len(nH_grid), height, width), np.nan)
    grid_filled = np.zeros((len(x_grid), len(nH_grid)), dtype=bool)

    for xv, nv, matrix in zip(rx, rn, train_mats):
        i, j = x_index[float(xv)], nH_index[float(nv)]
        if grid_filled[i, j]:
            raise ValueError(
                f"Duplicate training coordinate at x={x_grid[i]:.12g}, "
                f"nH={nH_grid[j]:.12g}."
            )
        image_grid[i, j] = matrix
        grid_filled[i, j] = True

    n_absent = int(np.count_nonzero(~grid_filled))
    if n_absent:
        print(
            f"Incomplete training grid: {n_absent} of {grid_filled.size} "
            f"({100.0 * n_absent / grid_filled.size:.1f}%) (x, nH) nodes absent. "
            f"Stencils will be widened across the gaps."
        )

    log_nH_grid = np.log10(nH_grid)
    n_x, n_nh = len(x_grid), len(log_nH_grid)

    # ------------------------------------------------------------------
    # Stencil selection.
    # ------------------------------------------------------------------
    def bracket(grid: np.ndarray, value: float, name: str):
        """Immediate bracketing indices; (i, i) on an exact node match."""
        exact = np.flatnonzero(np.isclose(grid, value, rtol=1e-12, atol=1e-12))
        if len(exact):
            i = int(exact[0])
            return i, i

        k = int(np.searchsorted(grid, value))
        if k == 0:
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {name}={value:.12g} is below the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )
            return 0, 1
        if k == len(grid):
            if not allow_extrapolation:
                raise ValueError(
                    f"Requested {name}={value:.12g} is above the training range "
                    f"[{grid[0]:.12g}, {grid[-1]:.12g}]."
                )
            return len(grid) - 2, len(grid) - 1
        return k - 1, k

    def all_present(ia: np.ndarray, ja: np.ndarray) -> bool:
        """Every combination in the outer product must be an existing image."""
        return bool(grid_filled[np.ix_(ia, ja)].all())

    def axis_stencils(i0: int, i1: int, n_axis: int, exact: bool):
        """
        Candidate index arrays for one axis, ordered from tightest outward.

        For method='linear' this is just the bracketing pair, progressively
        widened. For 'cubic' it is that pair plus one node outside on each
        side, clamped inward at the grid boundaries.
        """
        if exact:
            return [np.array([i0])]

        out = []
        for lo in range(i0, max(-1, i0 - max_expand - 1), -1):
            for hi in range(i1, min(n_axis, i1 + max_expand + 1)):
                if method == "linear":
                    out.append((hi - lo, np.array([lo, hi])))
                    continue
                a = lo - 1
                b = hi + 1
                # Clamp inward at the boundaries rather than extrapolating.
                if a < 0:
                    a, b = lo, min(n_axis - 1, b + 1)
                    if b <= hi:
                        continue
                    idx = np.array([lo, hi, b])
                    idx = np.unique(np.concatenate([idx, [min(n_axis - 1, b + 1)]]))
                elif b > n_axis - 1:
                    b = hi
                    a2 = max(0, a - 1)
                    idx = np.unique(np.array([a2, a, lo, hi]))
                else:
                    idx = np.array([a, lo, hi, b])
                if len(idx) < 2:
                    continue
                out.append((idx[-1] - idx[0], idx))

        out.sort(key=lambda t: t[0])
        return [idx for _, idx in out]

    # ------------------------------------------------------------------
    # Generate.
    # ------------------------------------------------------------------
    mats_out = np.empty((len(test_xs), height, width), dtype=np.float64)
    vars_out = np.empty_like(mats_out)
    extrapolated_flags = np.zeros(len(test_xs), dtype=bool)
    widened_flags = np.zeros(len(test_xs), dtype=bool)
    stencil_span = np.ones((len(test_xs), 2), dtype=np.int64)
    weight_l1 = np.ones(len(test_xs), dtype=np.float64)
    weight_l2 = np.ones(len(test_xs), dtype=np.float64)

    n_negative_pixels = 0
    n_super_poisson = 0

    for t, (x_star, nH_star) in enumerate(zip(test_xs, test_nHs)):
        eta_star = np.log10(nH_star)

        ix0, ix1 = bracket(x_grid, x_star, "x")
        in0, in1 = bracket(log_nH_grid, eta_star, "log10(nH)")

        extrapolated_flags[t] = (
            x_star < x_grid[0] or x_star > x_grid[-1]
            or eta_star < log_nH_grid[0] or eta_star > log_nH_grid[-1]
        )

        cand_x = axis_stencils(ix0, ix1, n_x, ix0 == ix1)
        cand_n = axis_stencils(in0, in1, n_nh, in0 == in1)

        chosen = None
        for ia in cand_x:
            for ja in cand_n:
                if all_present(ia, ja):
                    chosen = (ia, ja)
                    break
            if chosen is not None:
                break

        if chosen is None:
            raise ValueError(
                f"No {method} stencil with all training images present was found "
                f"within {max_expand} grid steps of x={x_star:.12g}, "
                f"log10(nH)={eta_star:.12g}. Increase max_expand or reduce the "
                "size of the removed region."
            )

        ia, ja = chosen
        compact = n_stencil if not (ix0 == ix1 or in0 == in1) else 1
        widened_flags[t] = (len(ia) > compact) or (len(ja) > compact) or (
            (ia[-1] - ia[0]) > n_stencil - 1 or (ja[-1] - ja[0]) > n_stencil - 1
        )
        stencil_span[t] = (ia[-1] - ia[0], ja[-1] - ja[0])

        wx = (
            np.array([1.0]) if len(ia) == 1
            else _lagrange_weights(x_grid[ia], x_star)
        )
        wn = (
            np.array([1.0]) if len(ja) == 1
            else _lagrange_weights(log_nH_grid[ja], eta_star)
        )

        W = np.outer(wx, wn)                     # (len(ia), len(ja))
        weight_l1[t] = float(np.abs(W).sum())
        weight_l2[t] = float((W ** 2).sum())
        if weight_l2[t] > 1.0 + 1e-9:
            n_super_poisson += 1

        block = image_grid[np.ix_(ia, ja)]       # (P, Q, H, W)
        generated = np.tensordot(W, block, axes=([0, 1], [0, 1]))
        variance = np.tensordot(W ** 2, block, axes=([0, 1], [0, 1]))

        if not np.all(np.isfinite(generated)):
            raise RuntimeError(
                f"Point {t} (x={x_star:.6g}, nH={nH_star:.6e}) produced "
                "non-finite pixels: an absent training node entered the "
                "interpolation. This is a bug."
            )

        n_neg = int(np.count_nonzero(generated < 0.0))
        if n_neg:
            n_negative_pixels += n_neg
            generated = np.clip(generated, 0.0, None)

        # Squared weights times non-negative counts: variance stays >= 0.
        mats_out[t] = generated
        vars_out[t] = np.clip(variance, 0.0, None)

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compressed else np.savez
    save(
        output_npz_path,
        xs=test_xs,
        nHs=test_nHs,
        mats=mats_out,
        Vs=vars_out,
        extrapolated=extrapolated_flags,
        widened=widened_flags,
        stencil_span=stencil_span,
        weight_l1=weight_l1,
        weight_l2=weight_l2,
    )

    print()
    print(f"Saved {len(mats_out)} matrices to '{output_npz_path}'.")
    print(f"Method: separable {method} (Lagrange), {n_stencil} nodes per axis")
    print(f"Matrix shape: {matrix_shape}")
    print(
        f"Extrapolated points     : "
        f"{int(np.count_nonzero(extrapolated_flags))}/{len(extrapolated_flags)}"
    )
    print(
        f"Widened stencils        : "
        f"{int(np.count_nonzero(widened_flags))}/{len(widened_flags)}"
    )
    print(
        f"Super-Poisson (sum W^2>1): {n_super_poisson}/{len(weight_l2)}   "
        f"max sum|W| = {weight_l1.max():.3f}, max sum W^2 = {weight_l2.max():.3f}"
    )
    if n_negative_pixels:
        print(f"Negative pixels clipped : {n_negative_pixels}")
    print(
        "Parameter treatment: linear in x, linear in log10(nH), "
        f"{method} in pixel intensity."
    )
    print("Propagated variance saved under key 'Vs' (pass as V to the chi^2 function).")

from pathlib import Path
from typing import Union

import numpy as np
from scipy.interpolate import RBFInterpolator


def create_rbf_interpolated_npz(
    training_npz_path: Union[str, Path],
    testing_npz_path: Union[str, Path],
    output_npz_path: Union[str, Path],
    *,
    decimals: int = 10,
    kernel: str = "thin_plate_spline",
    degree: int = 1,
    neighbors: int | None = 50,
    smoothing: float = 0.0,
    allow_extrapolation: bool = False,
    clip_negative: bool = False,
    compressed: bool = True,
    weight_batch_size: int = 32,
) -> None:
    """
    Generate pixel-wise RBF-interpolated images at requested parameter
    coordinates.

    The interpolator operates in the two-dimensional parameter space

        q = (x, log10(nH)),

    after scaling each coordinate axis to [0, 1]. The default model is a
    local thin-plate-spline RBF with a linear polynomial term.

    Inputs may be NPZ archives or directories containing NPY matrices whose
    filenames can be interpreted by ``parse_parameters``.

    Expected training NPZ keys
    --------------------------
    xs   : x coordinates, shape (N_train,)
    nHs  : nH coordinates, shape (N_train,)
    mats : photon-count images, shape (N_train, H, W), or equivalent with
           singleton axes

    Expected testing NPZ keys
    -------------------------
    xs   : requested x coordinates, shape (N_test,)
    nHs  : requested nH coordinates, shape (N_test,)

    Any ``mats`` stored in the testing archive are ignored.

    Interpolation model
    -------------------
    Each pixel is treated as one output of the same RBF interpolator. For a
    query q*, the prediction is a linear combination of training images,

        S_out = sum_i w_i(q*) S_i.

    With ``neighbors`` set, SciPy constructs a local interpolant from the
    nearest training coordinates for each query. Setting ``neighbors=None``
    uses all training coordinates.

    Propagated Poisson variance
    ---------------------------
    Assuming that the training images are independent raw Poisson count
    realizations,

        Var(S_i) = S_i,

    the propagated per-pixel variance is

        V_out = sum_i w_i(q*)^2 S_i.

    The effective RBF weights are recovered by applying the same interpolator
    to the identity matrix. This includes the chosen RBF kernel, polynomial
    term, smoothing value, and local-neighbour selection.

    Important: this variance model is valid only when ``mats`` contains
    independent raw count images, or arrays with an equivalent known
    Var(S_i)=S_i interpretation.

    Output NPZ keys
    ---------------
    xs             : requested x coordinates
    nHs            : requested nH coordinates
    mats           : RBF predictions, shape (N_test, H, W)
    Vs             : propagated variance, shape (N_test, H, W)
    weights        : effective RBF weights, shape (N_test, N_train)
    extrapolated   : True outside the rectangular training-coordinate range
    nearest_count  : number of non-negligible effective weights per query
    train_xs       : training x coordinates in the order used by ``weights``
    train_nHs      : training nH coordinates in the order used by ``weights``
    coord_min      : lower scaling bounds for (x, log10(nH))
    coord_scale    : scaling widths for (x, log10(nH))
    kernel         : selected RBF kernel
    degree         : polynomial degree
    neighbors      : requested local-neighbour count (-1 means all points)
    smoothing      : smoothing parameter

    Parameters
    ----------
    training_npz_path
        Training NPZ or directory of training NPY files.
    testing_npz_path
        Testing-coordinate NPZ or directory of testing NPY files.
    output_npz_path
        Destination NPZ path.
    decimals
        Decimal places used to detect duplicate coordinates in the scaled
        (x, log10(nH)) space.
    kernel
        Kernel passed to scipy.interpolate.RBFInterpolator.
    degree
        Degree of the added polynomial. For thin-plate splines, degree=1 is
        the standard minimum well-posed choice.
    neighbors
        Number of nearest training coordinates used for each query. None uses
        all training coordinates.
    smoothing
        Non-negative RBF smoothing parameter. Zero gives exact interpolation
        at retained training coordinates.
    allow_extrapolation
        If False, reject queries outside the rectangular coordinate range of
        the retained training data.
    clip_negative
        If True, clip negative predicted pixels to zero. This changes the
        estimator nonlinearly, so the saved propagated variance then describes
        the unclipped RBF prediction. False is recommended for quantitative
        evaluation.
    compressed
        Save with np.savez_compressed when True.
    weight_batch_size
        Number of test queries processed together when recovering effective
        interpolation weights.

    Notes
    -----
    This function expects an existing ``parse_parameters(filename)`` function
    when an input path is a directory of NPY files.
    """
    training_npz_path = Path(training_npz_path)
    testing_npz_path = Path(testing_npz_path)
    output_npz_path = Path(output_npz_path)

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    if neighbors is not None and neighbors < 3:
        raise ValueError(
            "neighbors must be at least 3 in a two-dimensional parameter "
            "space, or None to use all training points."
        )

    if weight_batch_size < 1:
        raise ValueError("weight_batch_size must be at least 1.")

    def load_input(path: Path, *, require_mats: bool) -> dict[str, np.ndarray]:
        """Load coordinates and optionally matrices from NPZ or NPY files."""
        if path.is_dir():
            npy_paths = sorted(path.glob("*.npy"))
            if not npy_paths:
                raise FileNotFoundError(
                    f"No .npy files found in directory: {path}"
                )

            xs: list[float] = []
            nHs: list[float] = []
            mats: list[np.ndarray] = []

            for npy_path in npy_paths:
                x, nH = parse_parameters(npy_path.name)  # noqa: F821

                if x is None or nH is None:
                    raise ValueError(
                        f"Could not parse x and nH from filename: "
                        f"{npy_path.name}"
                    )

                xs.append(float(x))
                nHs.append(float(nH))

                if require_mats:
                    mats.append(np.load(npy_path))

            result: dict[str, np.ndarray] = {
                "xs": np.asarray(xs, dtype=np.float64),
                "nHs": np.asarray(nHs, dtype=np.float64),
            }
            if require_mats:
                result["mats"] = np.asarray(mats)
            return result

        if not path.is_file():
            raise FileNotFoundError(f"Input path not found: {path}")

        with np.load(path) as data:
            required_keys = (
                {"xs", "nHs", "mats"}
                if require_mats
                else {"xs", "nHs"}
            )
            missing = required_keys.difference(data.files)
            if missing:
                raise KeyError(
                    f"{path} is missing required keys: {sorted(missing)}"
                )

            return {
                key: np.asarray(data[key])
                for key in required_keys
            }

    # ------------------------------------------------------------------
    # Load coordinates and images.
    # ------------------------------------------------------------------
    training_data = load_input(training_npz_path, require_mats=True)
    testing_data = load_input(testing_npz_path, require_mats=False)

    train_xs = np.asarray(
        training_data["xs"], dtype=np.float64
    ).reshape(-1)
    train_nHs = np.asarray(
        training_data["nHs"], dtype=np.float64
    ).reshape(-1)
    train_mats_raw = np.asarray(training_data["mats"])

    test_xs = np.asarray(
        testing_data["xs"], dtype=np.float64
    ).reshape(-1)
    test_nHs = np.asarray(
        testing_data["nHs"], dtype=np.float64
    ).reshape(-1)

    # ------------------------------------------------------------------
    # Validate coordinate arrays.
    # ------------------------------------------------------------------
    if len(train_xs) != len(train_nHs) or len(train_xs) != len(train_mats_raw):
        raise ValueError(
            "Training arrays must have equal lengths: "
            f"len(xs)={len(train_xs)}, "
            f"len(nHs)={len(train_nHs)}, "
            f"len(mats)={len(train_mats_raw)}."
        )

    if len(test_xs) != len(test_nHs):
        raise ValueError(
            "Testing coordinate arrays must have equal lengths: "
            f"len(xs)={len(test_xs)}, len(nHs)={len(test_nHs)}."
        )

    if len(train_xs) == 0:
        raise ValueError("The training dataset is empty.")

    if len(test_xs) == 0:
        raise ValueError("The testing dataset contains no coordinates.")

    for name, array in (
        ("training xs", train_xs),
        ("training nHs", train_nHs),
        ("testing xs", test_xs),
        ("testing nHs", test_nHs),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values.")

    if np.any(train_nHs <= 0) or np.any(test_nHs <= 0):
        raise ValueError("All nH values must be positive.")

    if np.any(train_mats_raw < 0):
        raise ValueError(
            "Training matrices contain negative values. The propagated "
            "Poisson variance assumes non-negative raw photon counts."
        )

    # ------------------------------------------------------------------
    # Convert training matrices to a common 2-D shape.
    # ------------------------------------------------------------------
    squeezed_mats: list[np.ndarray] = []

    for index, matrix in enumerate(train_mats_raw):
        matrix_2d = np.squeeze(np.asarray(matrix))

        if matrix_2d.ndim != 2:
            raise ValueError(
                f"Training matrix {index} has original shape "
                f"{np.asarray(matrix).shape}; after squeezing it has shape "
                f"{matrix_2d.shape}, but a 2-D matrix is required."
            )

        matrix_2d = matrix_2d.astype(np.float64, copy=False)

        if not np.all(np.isfinite(matrix_2d)):
            raise ValueError(
                f"Training matrix {index} contains non-finite values."
            )

        squeezed_mats.append(matrix_2d)

    matrix_shape = squeezed_mats[0].shape

    for index, matrix in enumerate(squeezed_mats):
        if matrix.shape != matrix_shape:
            raise ValueError(
                "All training matrices must have the same shape. "
                f"Matrix 0 has shape {matrix_shape}, but matrix {index} "
                f"has shape {matrix.shape}."
            )

    train_mats = np.stack(squeezed_mats, axis=0)
    n_train, height, width = train_mats.shape
    n_pixels = height * width

    if neighbors is not None:
        neighbors_used = min(int(neighbors), n_train)
    else:
        neighbors_used = None

    # ------------------------------------------------------------------
    # Construct and scale parameter coordinates.
    #
    # Scaling prevents one axis from dominating Euclidean neighbour
    # distances merely because of its numerical range.
    # ------------------------------------------------------------------
    train_coords = np.column_stack(
        (train_xs, np.log10(train_nHs))
    )
    test_coords = np.column_stack(
        (test_xs, np.log10(test_nHs))
    )

    coord_min = train_coords.min(axis=0)
    coord_max = train_coords.max(axis=0)
    coord_scale = coord_max - coord_min

    if np.any(coord_scale <= 0):
        raise ValueError(
            "Both x and log10(nH) must vary across the training set."
        )

    train_coords_scaled = (train_coords - coord_min) / coord_scale
    test_coords_scaled = (test_coords - coord_min) / coord_scale

    # Detect duplicates after scaling, where decimal rounding is meaningful.
    rounded_coords = np.round(train_coords_scaled, decimals=decimals)
    _, duplicate_counts = np.unique(
        rounded_coords,
        axis=0,
        return_counts=True,
    )

    if np.any(duplicate_counts > 1):
        raise ValueError(
            "Duplicate or numerically indistinguishable training coordinates "
            "were found after scaling."
        )

    extrapolated_flags = np.any(
        (test_coords_scaled < 0.0) | (test_coords_scaled > 1.0),
        axis=1,
    )

    if np.any(extrapolated_flags) and not allow_extrapolation:
        bad = np.flatnonzero(extrapolated_flags)
        preview = ", ".join(map(str, bad[:10]))
        raise ValueError(
            f"{len(bad)} requested points lie outside the rectangular "
            f"training-coordinate range. Indices: {preview}. "
            "Set allow_extrapolation=True only if RBF extrapolation is "
            "intended."
        )

    # ------------------------------------------------------------------
    # Fit the image-valued RBF interpolator.
    # ------------------------------------------------------------------
    flattened_counts = train_mats.reshape(n_train, n_pixels)

    image_interpolator = RBFInterpolator(
        train_coords_scaled,
        flattened_counts,
        kernel=kernel,
        degree=degree,
        neighbors=neighbors_used,
        smoothing=smoothing,
    )

    predicted_flat = np.asarray(
        image_interpolator(test_coords_scaled),
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Recover effective linear weights.
    #
    # RBF interpolation is linear in the supplied data values. Applying the
    # identical interpolator to the N_train x N_train identity matrix returns,
    # for each test point, its effective coefficients on all training images.
    # ------------------------------------------------------------------
    identity = np.eye(n_train, dtype=np.float64)

    weight_interpolator = RBFInterpolator(
        train_coords_scaled,
        identity,
        kernel=kernel,
        degree=degree,
        neighbors=neighbors_used,
        smoothing=smoothing,
    )

    weights = np.empty(
        (len(test_coords_scaled), n_train),
        dtype=np.float64,
    )

    for start in range(0, len(test_coords_scaled), weight_batch_size):
        stop = min(start + weight_batch_size, len(test_coords_scaled))
        weights[start:stop] = weight_interpolator(
            test_coords_scaled[start:stop]
        )

    if not np.all(np.isfinite(weights)):
        raise RuntimeError("RBF weight recovery produced non-finite values.")

    # Verify that the recovered weights reproduce the image interpolator.
    reconstructed_flat = weights @ flattened_counts
    max_reconstruction_error = float(
        np.max(np.abs(reconstructed_flat - predicted_flat))
    )

    reconstruction_scale = max(
        1.0,
        float(np.max(np.abs(predicted_flat))),
    )

    if max_reconstruction_error > 1e-9 * reconstruction_scale:
        raise RuntimeError(
            "Recovered RBF weights do not reproduce the direct prediction. "
            f"Maximum absolute discrepancy: {max_reconstruction_error:.6e}."
        )

    # ------------------------------------------------------------------
    # Propagate independent Poisson variance.
    # ------------------------------------------------------------------
    variance_flat = (weights ** 2) @ flattened_counts

    if not np.all(np.isfinite(predicted_flat)):
        raise RuntimeError("RBF interpolation produced non-finite pixels.")

    if not np.all(np.isfinite(variance_flat)):
        raise RuntimeError("Variance propagation produced non-finite pixels.")

    if np.any(variance_flat < -1e-12):
        raise RuntimeError(
            "Variance propagation produced materially negative values."
        )

    variance_flat = np.maximum(variance_flat, 0.0)

    negative_pixel_counts = np.count_nonzero(
        predicted_flat < 0.0,
        axis=1,
    )

    if np.any(negative_pixel_counts > 0):
        affected = int(np.count_nonzero(negative_pixel_counts))
        total_negative = int(np.sum(negative_pixel_counts))
        print(
            f"Warning: RBF interpolation produced {total_negative} negative "
            f"pixels across {affected}/{len(test_xs)} test images."
        )

        if clip_negative:
            print(
                "Negative pixels are being clipped to zero. The saved Vs "
                "still describe the original unclipped linear RBF estimator."
            )
            predicted_flat = np.clip(predicted_flat, 0.0, None)

    interpolated_mats = predicted_flat.reshape(
        len(test_xs), height, width
    )
    variance_mats = variance_flat.reshape(
        len(test_xs), height, width
    )

    # Count weights that are numerically active. With local interpolation,
    # this should usually be close to ``neighbors_used``.
    weight_tolerance = 1e-12
    nearest_count = np.count_nonzero(
        np.abs(weights) > weight_tolerance,
        axis=1,
    ).astype(np.int64)

    # ------------------------------------------------------------------
    # Save output and metadata.
    # ------------------------------------------------------------------
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    save_function = (
        np.savez_compressed if compressed else np.savez
    )

    save_function(
        output_npz_path,
        xs=test_xs,
        nHs=test_nHs,
        mats=interpolated_mats,
        Vs=variance_mats,
        weights=weights,
        extrapolated=extrapolated_flags,
        nearest_count=nearest_count,
        train_xs=train_xs,
        train_nHs=train_nHs,
        coord_min=coord_min,
        coord_scale=coord_scale,
        kernel=np.asarray(kernel),
        degree=np.asarray(degree, dtype=np.int64),
        neighbors=np.asarray(
            -1 if neighbors_used is None else neighbors_used,
            dtype=np.int64,
        ),
        smoothing=np.asarray(smoothing, dtype=np.float64),
    )

    print()
    print(
        f"Saved {len(interpolated_mats)} RBF-interpolated matrices to "
        f"'{output_npz_path}'."
    )
    print(f"Matrix shape: {matrix_shape}")
    print(
        "Parameter treatment: Euclidean RBF interpolation in scaled "
        "(x, log10(nH)) coordinates."
    )
    print(
        f"Kernel={kernel!r}, degree={degree}, "
        f"neighbors={neighbors_used}, smoothing={smoothing:g}"
    )
    print(
        f"Extrapolated points: "
        f"{int(np.count_nonzero(extrapolated_flags))}/{len(test_xs)}"
    )
    print(
        f"Queries with negative predicted pixels: "
        f"{int(np.count_nonzero(negative_pixel_counts))}/{len(test_xs)}"
    )
    print(
        "Effective RBF weights saved under key 'weights'; propagated Poisson "
        "variance saved under key 'Vs'."
    )
    print(
        f"Maximum weight-reconstruction discrepancy: "
        f"{max_reconstruction_error:.6e}"
    )


from pathlib import Path
from typing import Union
import statistics
import time

import numpy as np


N_WARMUP = 5
N_REPEAT = 200


# ---------------------------------------------------------------------
# Setup: load the training grid once (mirrors create_interpolated_npz).
# ---------------------------------------------------------------------
def _load_training(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.is_dir():
        npy_paths = sorted(path.glob("*.npy"))
        if not npy_paths:
            raise FileNotFoundError(f"No .npy files in {path}")

        xs, nHs, mats = [], [], []
        for npy_path in npy_paths:
            x, nH = parse_parameters(npy_path.name)
            if x is None or nH is None:
                raise ValueError(
                    f"Could not parse x and nH from {npy_path.name}"
                )
            xs.append(float(x))
            nHs.append(float(nH))
            mats.append(np.load(npy_path))

        return (
            np.asarray(xs, dtype=np.float64),
            np.asarray(nHs, dtype=np.float64),
            np.asarray(mats),
        )

    with np.load(path) as data:
        return (
            np.asarray(data["xs"], dtype=np.float64).reshape(-1),
            np.asarray(data["nHs"], dtype=np.float64).reshape(-1),
            np.asarray(data["mats"]),
        )


def _build_grid(train_xs, train_nHs, train_mats, decimals=10):
    """Return (x_grid, log_nH_grid, image_grid) with the same layout as
    create_interpolated_npz."""
    mats_2d = []
    for matrix in train_mats:
        m = np.squeeze(np.asarray(matrix)).astype(np.float64, copy=False)
        if m.ndim != 2:
            raise ValueError(f"Expected 2-D training matrix, got {m.shape}")
        mats_2d.append(m)

    height, width = mats_2d[0].shape

    rounded_xs = np.round(train_xs, decimals)
    rounded_nHs = np.round(train_nHs, decimals)

    unique_xs = np.unique(rounded_xs)
    unique_nHs = np.unique(rounded_nHs)

    x_grid = np.array(
        [np.mean(train_xs[rounded_xs == r]) for r in unique_xs],
        dtype=np.float64,
    )
    nH_grid = np.array(
        [np.mean(train_nHs[rounded_nHs == r]) for r in unique_nHs],
        dtype=np.float64,
    )

    x_order = np.argsort(x_grid)
    nH_order = np.argsort(nH_grid)

    x_grid = x_grid[x_order]
    nH_grid = nH_grid[nH_order]
    unique_xs = unique_xs[x_order]
    unique_nHs = unique_nHs[nH_order]

    x_index = {float(v): i for i, v in enumerate(unique_xs)}
    nH_index = {float(v): i for i, v in enumerate(unique_nHs)}

    image_grid = np.empty(
        (len(x_grid), len(nH_grid), height, width),
        dtype=np.float64,
    )
    filled = np.zeros((len(x_grid), len(nH_grid)), dtype=bool)

    for x_value, nH_value, matrix in zip(rounded_xs, rounded_nHs, mats_2d):
        ix = x_index[float(x_value)]
        inh = nH_index[float(nH_value)]
        image_grid[ix, inh] = matrix
        filled[ix, inh] = True

    if not np.all(filled):
        raise ValueError(
            f"Training grid incomplete: {int((~filled).sum())} "
            "missing coordinate combinations."
        )

    return x_grid, np.log10(nH_grid), image_grid


def _bounding_indices(grid, value):
    exact = np.flatnonzero(np.isclose(grid, value, rtol=1e-12, atol=1e-12))
    if exact.size:
        i = int(exact[0])
        return i, i

    j = int(np.searchsorted(grid, value))
    if j == 0:
        return 0, 1
    if j == len(grid):
        return len(grid) - 2, len(grid) - 1
    return j - 1, j


# ---------------------------------------------------------------------
# The timed unit: everything needed to produce one image.
# ---------------------------------------------------------------------
def _generate_one(x_grid, log_nH_grid, image_grid, x_star, nH_star):
    log_nH_star = np.log10(nH_star)

    ix0, ix1 = _bounding_indices(x_grid, x_star)
    inh0, inh1 = _bounding_indices(log_nH_grid, log_nH_star)

    u = (
        0.0
        if ix0 == ix1
        else (x_star - x_grid[ix0]) / (x_grid[ix1] - x_grid[ix0])
    )
    v = (
        0.0
        if inh0 == inh1
        else (log_nH_star - log_nH_grid[inh0])
        / (log_nH_grid[inh1] - log_nH_grid[inh0])
    )

    i00 = image_grid[ix0, inh0]

    if ix0 == ix1 and inh0 == inh1:
        generated = i00.copy()
        variance = i00.copy()

    elif ix0 == ix1:
        i01 = image_grid[ix0, inh1]
        generated = (1.0 - v) * i00 + v * i01
        variance = (1.0 - v) ** 2 * i00 + v**2 * i01

    elif inh0 == inh1:
        i10 = image_grid[ix1, inh0]
        generated = (1.0 - u) * i00 + u * i10
        variance = (1.0 - u) ** 2 * i00 + u**2 * i10

    else:
        i10 = image_grid[ix1, inh0]
        i01 = image_grid[ix0, inh1]
        i11 = image_grid[ix1, inh1]

        w00 = (1.0 - u) * (1.0 - v)
        w10 = u * (1.0 - v)
        w01 = (1.0 - u) * v
        w11 = u * v

        generated = w00 * i00 + w10 * i10 + w01 * i01 + w11 * i11
        variance = (
            w00**2 * i00 + w10**2 * i10 + w01**2 * i01 + w11**2 * i11
        )

    if np.any(generated < 0.0):
        generated = np.clip(generated, 0.0, None)

    return generated, variance


def _report(label, times_seconds):
    t = [x * 1e3 for x in times_seconds]  # ms
    print(f"{label}")
    print(f"  median : {statistics.median(t):.4f} ms")
    print(f"  mean   : {statistics.mean(t):.4f} ms")
    if len(t) > 1:
        print(f"  stdev  : {statistics.stdev(t):.4f} ms")
    print(f"  min/max: {min(t):.4f} / {max(t):.4f} ms")
    print()


def time_interpolation_per_image(
    training_npz_path: Union[str, Path],
    testing_npz_path: Union[str, Path],
    *,
    decimals: int = 10,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
    reflex_hours_per_image: float | None = None,
) -> float:
    """
    Time the generation of a single interpolated image.

    Setup (file I/O, grid construction) is performed once and excluded.
    Test coordinates are cycled through so the measurement averages over
    the grid rather than repeating one favourable cell.

    Returns
    -------
    float
        Median per-image time in seconds.
    """
    training_npz_path = Path(training_npz_path)
    testing_npz_path = Path(testing_npz_path)

    setup_start = time.perf_counter()

    train_xs, train_nHs, train_mats = _load_training(training_npz_path)
    x_grid, log_nH_grid, image_grid = _build_grid(
        train_xs, train_nHs, train_mats, decimals=decimals
    )

    if testing_npz_path.is_dir():
        test_xs, test_nHs, _ = _load_training(testing_npz_path)
    else:
        with np.load(testing_npz_path) as data:
            test_xs = np.asarray(data["xs"], dtype=np.float64).reshape(-1)
            test_nHs = np.asarray(data["nHs"], dtype=np.float64).reshape(-1)

    setup_seconds = time.perf_counter() - setup_start

    n_test = len(test_xs)
    print(
        f"Setup: {image_grid.shape[0]}x{image_grid.shape[1]} training grid, "
        f"{image_grid.shape[2]}x{image_grid.shape[3]} images, "
        f"{n_test} test points"
    )
    print(f"Setup time (excluded from per-image figure): {setup_seconds:.2f} s\n")

    for k in range(n_warmup):
        idx = k % n_test
        _generate_one(
            x_grid, log_nH_grid, image_grid,
            float(test_xs[idx]), float(test_nHs[idx]),
        )

    times = []
    for k in range(n_repeat):
        idx = k % n_test
        x_star = float(test_xs[idx])
        nH_star = float(test_nHs[idx])

        start = time.perf_counter()
        _generate_one(x_grid, log_nH_grid, image_grid, x_star, nH_star)
        times.append(time.perf_counter() - start)

    _report(
        f"Bilinear interpolation, single image "
        f"({n_repeat} repeats over {n_test} test points)",
        times,
    )

    median_seconds = statistics.median(times)

    # Amortised figure: setup spread over the full test grid. Reported
    # only for contrast -- it is not the per-image cost.
    amortised_ms = (setup_seconds / n_test + median_seconds) * 1e3
    print(
        f"For contrast, setup amortised over {n_test} images: "
        f"{amortised_ms:.4f} ms per image"
    )

    if reflex_hours_per_image is not None:
        ratio = (reflex_hours_per_image * 3600.0) / median_seconds
        print(f"\nSpeed-up over RefleX: {ratio:.4g}  (10^{np.log10(ratio):.1f})")

    return median_seconds