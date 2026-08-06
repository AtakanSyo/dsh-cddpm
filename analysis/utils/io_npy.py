import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from .parsing import parse_parameters

def format_x(x):
    """Convert numeric or string x -> file-safe format (9700.0 -> '9700_0')."""
    if isinstance(x, (int, float)):
        x = f"{x:.1f}"             # 9700.0
    parts = str(x).split(".")
    return f"{parts[0]}_{parts[1]}"

def format_nH(nH):
    """Convert numeric or scientific notation -> file-safe format (2.21e+11 -> '2_21e_11')."""
    if isinstance(nH, (int, float)):
        nH = f"{nH:.2e}"           # 2.21e+11
    return nH.replace(".", "_").replace("+", "_")

def find_npy(path_to_npy_folder, x, nH):
    x_str  = format_x(x)
    nH_str = format_nH(nH)
    for fn in os.listdir(path_to_npy_folder):
        if fn.endswith(".npy") and f"x_{x_str}" in fn and f"nH_{nH_str}" in fn:
            return os.path.join(path_to_npy_folder, fn)
    return None




import os, glob, math, warnings
import numpy as np
from astropy.io import fits

# assumes you already have: from PIL import Image  # (not used here)
# and a parser: parse_parameters(filename) -> (x: float | None, nH: float | None)

def save_npy_to_npz(
    npy_dir: str,
    out_path: str = "matrices_with_params.npz",
    checkpoint: str | None = None,
    pattern: str = "*.npy",
    fits_dir: str | None = None,
    exclude_nH: list[float] | tuple[float, ...] | None = None,
    exclude_x: list[float] | tuple[float, ...] | None = None,
    nH_tol: float = 0.0,
    x_tol: float = 0.0,
    dtype: str = "float32",   # stored dtype for matrices in the NPZ
    k: float = 1.0,
    resize_to_48: bool = False,
    interpolate_to_48: bool = False,
):
    """
    Save NxN matrices (from .npy files) and their parameters into an NPZ,
    along with dmin and dmax from matching FITS files.

    Args:
        npy_dir: Directory containing NPYs (e.g., .../SAMPLES_NPY_*).
        out_path: Base output .npz path.
        checkpoint: Optional checkpoint string appended as _checkpoint<str>.npz.
        pattern: Glob pattern for NPYs.
        fits_dir: Directory containing the matching FITS files.
                  If None, assumes sibling 'SAMPLES_FITS'.
        exclude_nH: Iterable of nH values to skip (exact match within `nH_tol`).
        exclude_x: Iterable of x values to skip (exact match within `x_tol`).
        nH_tol: Absolute tolerance for comparing nH (default 0.0 -> exact compare).
        x_tol: Absolute tolerance for comparing x.
        dtype: NPY arrays are cast to this dtype before saving ('float32' or 'float64').
        k: Scalar multiplier applied to every matrix element prior to saving.
        resize_to_48: If True, crops 50x50 inputs down to 48x48 (strictly for 50x50 inputs).
        interpolate_to_48: If True, bilinearly interpolates matrices to 48x48.
    """
    paths = sorted(glob.glob(os.path.join(npy_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No NPYs matching {pattern} in {npy_dir}")

    # infer fits_dir if not provided
    if fits_dir is None:
        parent, _ = os.path.split(os.path.abspath(npy_dir))
        fits_dir = os.path.join(parent, "SAMPLES_FITS")

    xs, nHs, mats = [], [], []
    dmins, dmaxs = [], []
    H = W = None

    # normalize exclusion list
    excl_nH = list(exclude_nH) if exclude_nH is not None else []
    excl_x = list(exclude_x) if exclude_x is not None else []
    n_skipped = 0

    for p in paths:
        fname = os.path.basename(p)
        x, nH = parse_parameters(fname)
        if x is None:
            continue

        # skip if nH is in exclusion list (within tolerance)
        if excl_nH and any(math.isclose(nH, v, rel_tol=0.0, abs_tol=nH_tol) for v in excl_nH):
            n_skipped += 1
            continue

        if excl_x and any(math.isclose(x, v, rel_tol=0.0, abs_tol=x_tol) for v in excl_x):
            n_skipped += 1
            continue

        # load NPY (expect 2D after squeeze)
        try:
            arr = np.load(p)
        except Exception as e:
            warnings.warn(f"Failed to load {fname}: {e}")
            continue

        arr = np.squeeze(arr)
        if arr.ndim != 2:
            warnings.warn(f"Skipping {fname}: expected 2D, got shape {arr.shape}")
            continue

        if resize_to_48:
            if arr.shape == (50, 50):
                # take a 48x48 crop centered within the 50x50 matrix
                arr = arr[1:-1, 1:-1]
            elif arr.shape != (50, 50):
                raise ValueError("resize_to_48 expects 50x50 inputs only.")

        if interpolate_to_48 and arr.shape != (48, 48):
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
            arr = (
                F.interpolate(
                    tensor, size=(48, 48), mode="bilinear", align_corners=False
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )

        # enforce consistent shape across all entries
        if H is None:
            H, W = arr.shape
        elif arr.shape != (H, W):
            raise ValueError(f"Matrix size mismatch for {fname}: {arr.shape} vs {(H, W)}")

        if k != 1.0:
            arr = arr * k

        xs.append(x)
        nHs.append(nH)
        mats.append(arr.astype(dtype))

        # match FITS (same basename)
        base, _ = os.path.splitext(fname)
        fits_path = os.path.join(fits_dir, f"{base}.fits")
        if os.path.isfile(fits_path):
            try:
                with fits.open(fits_path, memmap=False) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        dmin = float(np.nanmin(data))
                        dmax = float(np.nanmax(data))
                    else:
                        dmin, dmax = np.nan, np.nan
            except Exception as e:
                warnings.warn(f"Failed to read FITS for {fname}: {e}")
                dmin, dmax = np.nan, np.nan
        else:
            dmin, dmax = np.nan, np.nan

        dmins.append(dmin)
        dmaxs.append(dmax)

    if not xs:
        raise RuntimeError("Parsed 0 matrices with valid parameters after exclusions.")

    base, ext = os.path.splitext(out_path)
    ext = ext or ".npz"
    if checkpoint:
        out_path = f"{base}_{checkpoint}k{ext}"

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        out_path,
        xs=np.asarray(xs, dtype=np.float64),
        nHs=np.asarray(nHs, dtype=np.float64),
        mats=np.stack(mats, axis=0),               # shape: [N, H, W], dtype per arg
        dmin=np.asarray(dmins, dtype=np.float64),
        dmax=np.asarray(dmaxs, dtype=np.float64),
        H=np.int64(H), W=np.int64(W),
    )
    msg = f"Saved {len(xs)} matrices to {out_path} with shape {(H, W)} + dmin/dmax arrays."
    if excl_nH or excl_x:
        msg += " Skipped {} due to".format(n_skipped)
        if excl_nH:
            msg += f" exclude_nH={excl_nH} (tol={nH_tol})"
        if excl_x:
            if excl_nH:
                msg += " and"
            msg += f" exclude_x={excl_x} (tol={x_tol})"
        msg += "."
    print(msg)
