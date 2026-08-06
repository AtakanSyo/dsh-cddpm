import os
import re
import math
import glob
import json
import torch
import logging
import numpy as np
import torchvision
import torch.nn.functional as F
from astropy.io import fits

def build_conditions_from_dataset(dataset_dir, device="cpu", stats_path=None):
    """
    Parse the actual matrix files in a dataset directory into Z-score
    normalized condition tensors.

    One condition is created for each existing NPY file. The function does
    not construct a Cartesian product of all unique x and nH values.
    """
    project_root = os.getcwd()
    data_path = os.path.abspath(
        os.path.join(
            project_root,
            os.pardir,
            "data",
            dataset_dir,
            "matrices",
        )
    )

    # --------------------------------------------------------------
    # Load condition-normalization statistics used during training.
    # --------------------------------------------------------------
    if stats_path is None:
        stats_path = os.path.join(
            project_root,
            "outputs",
            "condition_stats.json",
        )

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Could not find stats file at {stats_path}. "
            "Make sure condition_stats.json was saved during training."
        )

    with open(stats_path, "r") as file:
        stats = json.load(file)

    mean_c1 = float(stats["mean_c1"])
    std_c1 = float(stats["std_c1"])
    mean_c2 = float(stats["mean_c2"])
    std_c2 = float(stats["std_c2"])

    # --------------------------------------------------------------
    # Collect and parse actual files.
    # --------------------------------------------------------------
    img_files = sorted(
        glob.glob(os.path.join(data_path, "*.npy"))
    )

    pattern = re.compile(
        r"""^dust_x_
            (?P<x_integer>\d+)
            _(?P<x_decimal>\d+)
            _nH_
            (?P<man_integer>\d+)
            _(?P<man_decimal>\d+)
            e_(?P<exp>\d+)
            \.npy$
        """,
        re.VERBOSE,
    )

    parsed_samples = []

    for fpath in img_files:
        fname = os.path.basename(fpath)
        match = pattern.match(fname)

        if match is None:
            logging.warning(
                f"Skipping file with unexpected format: {fname}"
            )
            continue

        try:
            x_val = float(
                f"{match.group('x_integer')}."
                f"{match.group('x_decimal')}"
            )

            nH_val = float(
                f"{match.group('man_integer')}."
                f"{match.group('man_decimal')}"
                f"e{match.group('exp')}"
            )

        except ValueError as error:
            logging.warning(
                f"Skipping {fname}: parse error: {error}"
            )
            continue

        parsed_samples.append(
            {
                "filename": fname,
                "x": x_val,
                "nH": nH_val,
            }
        )

    if not parsed_samples:
        raise ValueError(
            f"No valid matrix files found in {data_path!r}"
        )

    # --------------------------------------------------------------
    # Sort actual samples into a stable order.
    #
    # This retains only parameter pairs that truly exist.
    # --------------------------------------------------------------
    parsed_samples.sort(
        key=lambda sample: (
            sample["nH"],
            sample["x"],
        )
    )

    # --------------------------------------------------------------
    # Normalize each actual parameter pair.
    # --------------------------------------------------------------
    def zscore_x(x):
        return (
            x - mean_c1
        ) / (
            std_c1 + 1e-8
        )

    def zscore_lognH(nH):
        return (
            math.log10(nH) - mean_c2
        ) / (
            std_c2 + 1e-8
        )

    cond_list = [
        [
            zscore_x(sample["x"]),
            zscore_lognH(sample["nH"]),
        ]
        for sample in parsed_samples
    ]

    conditions_all = torch.tensor(
        cond_list,
        dtype=torch.float32,
        device=device,
    )

    ordered_fnames = [
        sample["filename"]
        for sample in parsed_samples
    ]

    x_raw = [
        sample["x"]
        for sample in parsed_samples
    ]

    nH_raw = [
        sample["nH"]
        for sample in parsed_samples
    ]

    meta = {
        "grid_w": None,
        "grid_h": None,
        "x_raw": x_raw,
        "nH_raw": nH_raw,
        "ordered_filenames": ordered_fnames,
        "stats": {
            "mean_c1": mean_c1,
            "std_c1": std_c1,
            "mean_c2": mean_c2,
            "std_c2": std_c2,
            "stats_path": stats_path,
        },
        "dataset_dir": dataset_dir,
        "data_path": data_path,
        "num_samples": len(parsed_samples),
    }

    logging.info(
        f"Built {len(parsed_samples)} conditions from actual files."
    )

    return conditions_all, meta

def estimate_k_from_fits_single_pixel(fits_path):
    """
    Estimate the photon→brightness scaling factor (k) for a FITS image.
    Assumes PNGs were made by scaling max photon count to brightness 255.
    
    k = max_photon_count / 255
    """
    with fits.open(fits_path) as hdul:
        # Find first HDU with data
        # data = next((hdu.data for hdu in hdul if hdu.data is not None), None)
        # if data is None:
        #     logging.warning(f"No image data found in {fits_path}")
        #     return None

        # max_photon_count = float(data.max())
        # k = max_photon_count / 255.0
        #logging.info(f"{fits_path} → max photon count: {max_photon_count}, k: {k}")
        with fits.open(fits_path) as hdul:
            # Find first HDU with data
            data = next((hdu.data for hdu in hdul if hdu.data is not None), None)
            if data is None:
                logging.warning(f"No image data found in {fits_path}")
                return None

            data_min = float(data.min())
            data_max = float(data.max())

        return data_max, data_min


def load_originals_from_dataset(ordered_filenames, dataset_dir, size=48):
    """
    Load and preprocess original dataset matrices in the same order as conditions,
    and attach each image's photon→brightness scaling factor k from its FITS file.

    Args:
        ordered_filenames (list[str]): Filenames in correct grid order.
        size (int): Resize target (square).

    Returns:
        list[tuple]: List of (tensor[1,H,W], k_value) pairs.
    """

    # an example file name is dust_x_9700_0_nH_5_62e_07.fits.
    # extract the x = 9700.0 and nH = 5.62e07 info, and return it chat.
    project_root = os.getcwd()
    data_path = os.path.join(project_root, os.pardir, 'data', dataset_dir, 'matrices')
    data_path = os.path.abspath(data_path)

    results = []

    for fn in ordered_filenames:

        if not os.path.isabs(fn):
            fn = os.path.join(data_path, fn)
        if not os.path.exists(fn):
            logging.warning(f"Missing image file: {fn}")
            continue

        matrix = np.load(fn)
        img_tensor = torch.from_numpy(matrix).float()
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 3 and img_tensor.shape[0] != 1:
            img_tensor = img_tensor[:1]

        # Derive FITS path
        base = os.path.splitext(os.path.basename(fn))[0]
        fits_path = os.path.join(os.path.dirname(fn), "fits_images", base + ".fits")

        # Extract x and nH from filename
        match = re.search(r"x_([0-9_]+)_nH_([0-9_]+e_[0-9]+)", base)

        if match:
            x_str = match.group(1).replace("_", ".")
            nH_str = match.group(2).replace("_", ".", 1).replace("_", "") # important 
            # prevents multiplication by 100 error
            try:
                x_val = float(x_str)
                nH_val = float(nH_str)
            except ValueError:
                x_val, nH_val = None, None
        else:
            x_val, nH_val = None, None

        data_min = None
        data_max = None
        if os.path.exists(fits_path):
            data_max, data_min = estimate_k_from_fits_single_pixel(fits_path)
        else:
            logging.warning(f"Missing FITS: {fits_path}")

        results.append((img_tensor, data_max, data_min, x_val, nH_val))

    return results