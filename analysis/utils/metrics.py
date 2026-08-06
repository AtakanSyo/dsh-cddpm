import numpy as np
from typing import Optional


def mean_error_between_npz(
    npz_model_path: str,
    npz_data_path: str,
    error_func,
    *,
    err_kwargs=None,
    decimals: int = 6,
    abs_tol: float = 1e-6,
):
    """
    Compute the mean and standard deviation of errors between matching (x, nH)
    matrices contained in two NPZ archives. No plotting is performed.

    Returns:
        (mean_error, std_error)
    """
    if err_kwargs is None:
        err_kwargs = {}

    model_npz = np.load(npz_model_path)
    xs_m = np.round(model_npz["xs"].astype(float), decimals)
    nHs_m = np.round(model_npz["nHs"].astype(float), decimals)
    mats_m = model_npz["mats"]

    data_npz = np.load(npz_data_path)
    xs_d = np.round(data_npz["xs"].astype(float), decimals)
    nHs_d = np.round(data_npz["nHs"].astype(float), decimals)
    mats_d = data_npz["mats"]

    data_map = {}
    for x, nH, mat in zip(xs_d, nHs_d, mats_d):
        data_map[(float(x), float(nH))] = np.squeeze(mat)

    shared_pairs = set(zip(xs_m, nHs_m)) & set(data_map.keys())
    if not shared_pairs:
        raise RuntimeError("No overlapping (x, nH) pairs between the two NPZ files.")

    errors = []
    for x_m, nH_m, mat_m in zip(xs_m, nHs_m, mats_m):
        key = (float(x_m), float(nH_m))
        if key not in data_map:
            continue

        mat_model = np.squeeze(mat_m)
        mat_data = data_map[key]

        if mat_model.shape != mat_data.shape:
            raise ValueError(
                f"Shape mismatch for (x={key[0]}, nH={key[1]}): "
                f"{mat_model.shape} vs {mat_data.shape}"
            )

        err_val = error_func(mat_data, mat_model, **err_kwargs)
        if isinstance(err_val, (list, tuple)):
            err_val = err_val[0]
        errors.append(float(err_val))

    errors_arr = np.asarray(errors, dtype=float)
    finite_mask = np.isfinite(errors_arr)
    if not finite_mask.any():
        raise RuntimeError("No finite errors to aggregate.")

    finite_errors = errors_arr[finite_mask]
    mean_err = float(np.mean(finite_errors))
    std_err = float(np.std(finite_errors))

    return mean_err, std_err






