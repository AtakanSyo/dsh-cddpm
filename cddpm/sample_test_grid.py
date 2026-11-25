"""Generate one DDPM and one DDIM image for every condition of the test grid.

    python sample_test_grid.py --variant C --epochs 12000 19000
    python sample_test_grid.py --epochs 12000 --checkpoint-dir ../377_Aux_Poisson_Deviance_Corrected/outputs

Counts are recovered with the transform the checkpoint was trained with
(Anscombe for B and C, linear for A).
Writes SAMPLES_NPY_<DDPM|DDIM>/<run>_<NN>k_<DDPM|DDIM>/SAMPLE_dust_x_..._nH_....npy,
the naming used by the analysis notebook. FITS copies are also written when
the dataset has reference FITS headers in matrices/fits_images.
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from config import DATA_ROOT, RUN_DIR, VARIANTS, model_dir
from dsh.checkpoint import checkpoint_path, load_model, run_tag
from dsh.data import matrix_files, normalize_condition, parse_filename, to_counts
from dsh.sampling import ddpm_scheduler, sample_ddim, sample_ddpm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_conditions(dataset):
    """(file name, x in pc, N_H) for every matrix, sorted by N_H then x."""
    rows = [(p.name, *parse_filename(p.name)) for p in matrix_files(DATA_ROOT / dataset / "matrices")]
    return sorted(rows, key=lambda row: (row[2], row[1]))


def sample_name(x_pc, nh):
    """e.g. x=9700.0, N_H=1e22 -> SAMPLE_dust_x_9700_0_nH_1_00e_22"""
    return f"SAMPLE_dust_x_{x_pc:.1f}".replace(".", "_") + "_nH_" + f"{nh:.2e}".replace(".", "_").replace("+", "_")


def save_images(counts, rows, tag, reference_fits_dir):
    """NPY files always; FITS copies only where a reference FITS header exists."""
    npy_dir = RUN_DIR / f"SAMPLES_NPY_{tag.rsplit('_', 1)[1]}" / tag
    fits_dir = RUN_DIR / f"SAMPLES_FITS_{tag.rsplit('_', 1)[1]}" / tag
    npy_dir.mkdir(parents=True, exist_ok=True)
    n_fits = 0
    for image, (filename, x_pc, nh) in zip(counts[:, 0].cpu().numpy(), rows):
        name = sample_name(x_pc, nh)
        np.save(npy_dir / f"{name}.npy", image)
        reference = reference_fits_dir / filename.replace(".npy", ".fits")
        if reference.exists():
            fits_dir.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data=image, header=fits.getheader(reference)).writeto(
                fits_dir / f"{name}.fits", overwrite=True)
            n_fits += 1
    logging.info("Saved %d samples to %s (%d FITS copies)", len(rows), npy_dir, n_fits)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, nargs="+", required=True, help="Checkpoint epochs, e.g. 12000")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="Use this folder's model_<variant>/")
    parser.add_argument("--checkpoint-dir", type=Path, help="Any other checkpoint folder, e.g. a 377 outputs/")
    parser.add_argument("--dataset", default="reflex_data_8_testing")
    parser.add_argument("--ddim-steps", type=int, default=50, help="0 skips DDIM sampling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.variant is None) == (args.checkpoint_dir is None):
        parser.error("give exactly one of --variant or --checkpoint-dir")
    checkpoint_dir = args.checkpoint_dir or model_dir(args.variant)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    rows = test_conditions(args.dataset)
    fits_dir = DATA_ROOT / args.dataset / "matrices" / "fits_images"
    run_name = run_tag(checkpoint_dir)

    for epoch in args.epochs:
        model, metadata, _ = load_model(checkpoint_path(checkpoint_dir, epoch), device)
        stats = metadata["condition_stats"]
        convert = lambda images: to_counts(images, metadata["data_max"], metadata["use_anscombe"])
        conditions = torch.tensor([normalize_condition(x, nh, stats) for _, x, nh in rows],
                                  dtype=torch.float32, device=device)
        tag = f"{run_name}_{epoch // 1000:02d}k"

        counts = convert(sample_ddpm(model, conditions, ddpm_scheduler()))
        save_images(counts, rows, f"{tag}_DDPM", fits_dir)
        if args.ddim_steps > 0:
            counts = convert(sample_ddim(model, conditions, steps=args.ddim_steps))
            save_images(counts, rows, f"{tag}_DDIM", fits_dir)


if __name__ == "__main__":
    main()
