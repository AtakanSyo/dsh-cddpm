#!/usr/bin/env python

import os, sys
ROOT = os.path.realpath(os.path.dirname(__file__))
MODEL_FOLDER_NAME = os.path.basename(ROOT)
PKG  = os.path.join(ROOT, "testing_helpers")
if ROOT not in sys.path: sys.path.insert(0, ROOT)
if PKG  not in sys.path: sys.path.insert(0, PKG)
# ----------------------------------------------------------------------------

import logging
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits

from sampling_config import SamplingConfig

from testing_helpers.seed_utils import set_seed
from testing_helpers.ddp_utils import init_distributed, get_device, is_rank0, cleanup
from testing_helpers.model import ConditionalUnet
from testing_helpers.io_utils import load_checkpoint
from testing_helpers.sampler import make_scheduler, sample_grid, sample_grid_DDIM
from testing_helpers.data_utils import (
    build_conditions_from_dataset,
    load_originals_from_dataset
)


def setup_logging(rank0: bool):
    if not rank0:
        return
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def save_scaled_tensor_as_fits(tensor_, output_path, ref_fits_path):
    """
    Scale a normalized tensor image [0,1] to physical photon counts
    and save it as a FITS file with header metadata from a reference file.
    """
    try:
        # HERE MULTIPLY THE TENSOR BY DATA_MAX SO THAT WE GET PHOTON COUNTS. 
        tensor_01 = tensor_
        if tensor_01.ndim == 3 and tensor_01.size(0) == 1:
            tensor_01 = tensor_01.squeeze(0)
        elif tensor_01.ndim != 2:
            raise ValueError(f"Expected tensor shape (1,H,W) or (H,W), got {tensor_01.shape}")

        # Load reference FITS header
        if not os.path.exists(ref_fits_path):
            raise FileNotFoundError(f"Reference FITS not found: {ref_fits_path}")

        with fits.open(ref_fits_path) as ref_hdul:
            ref_header = ref_hdul[0].header.copy()

        hdu = fits.PrimaryHDU(data=tensor_01.cpu().numpy(), header=ref_header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(output_path, overwrite=True)

    except Exception as e:
        logging.error(f"Failed to save scaled FITS for {output_path}: {e}")

def build_scheduler_safe(steps: int, kind: str):
    """
    Try to construct a scheduler with various possible signatures of make_scheduler.
    Prefers keyword 'kind' if supported; falls back to positional-only."""

    return make_scheduler(steps, kind=kind)

def gather_all_samples(is_distributed, rank, world_size, local_tensor):
    if is_distributed:
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, local_tensor.cpu())
        return torch.cat(gathered, dim=0)
    else:
        return local_tensor.cpu()


def finalize_and_save_samples(mode_tag: str,
                              local_samples_ancombe: torch.Tensor,
                              meta: dict,
                              dataset_directory: str,
                              checkpoint_name: str,
                              root_dir: Path,
                              is_distributed: bool,
                              rank: int,
                              world_size: int,
                              anscombe_global_max: float = 78.1888738888085):
    anscombe_hat = local_samples_ancombe * anscombe_global_max

    # Inverse Anscombe (approximate)
    counts_hat = (anscombe_hat / 2.0) ** 2 - 3.0 / 8.0
    local_samples = counts_hat.clamp(min=0)

    full_samples = gather_all_samples(is_distributed, rank, world_size, local_samples)

    if is_rank0(is_distributed, rank) and full_samples is not None and full_samples.numel() > 0:
        logging.info(f"--- [{mode_tag}] Generated Samples Statistics ---")
        logging.info(f"  Mean: {full_samples.mean().item():.4f}")
        logging.info(f"  Max:  {full_samples.max().item():.4f}")
        logging.info(f"  Min:  {full_samples.min().item():.4f}")
        logging.info("------------------------------------------")

    if is_rank0(is_distributed, rank):
        if dataset_directory:
            save_batch(mode_tag, full_samples, meta, dataset_directory, checkpoint_name, root_dir)
        else:
            logging.warning(f"[{mode_tag}] No dataset directory provided; cannot save FITS without reference metadata.")

def save_batch(mode_tag: str,
               full_samples: torch.Tensor,
               meta: dict,
               dataset_directory: str,
               checkpoint_name: str,
               root_dir: Path):
    """
    Save a batch of generated samples (PNG + FITS) under mode-specific folders.

    mode_tag: "DDPM" or "DDIM"
    """
    logging.info(f"[{mode_tag}] Preparing to save samples...")

    # Paths to original dataset (for reference FITS headers and metadata)
    data_root  = root_dir.parent / "data" / dataset_directory
    images_dir = data_root / "matrices"
    fits_dir   = images_dir / "fits_images"

    # Output folders per mode
    fits_output_dir = root_dir / f"SAMPLES_FITS_{mode_tag}" / f"{checkpoint_name}_{mode_tag}"
    npy_output_dir  = root_dir / f"SAMPLES_NPY_{mode_tag}" / f"{checkpoint_name}_{mode_tag}"

    fits_output_dir.mkdir(parents=True, exist_ok=True)
    npy_output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"[{mode_tag}] Saving FITS to: {fits_output_dir}")
    logging.info(f"[{mode_tag}] Saving NPY  to: {npy_output_dir}")

    ordered_filenames = meta["ordered_filenames"]
    orig_pairs = load_originals_from_dataset(ordered_filenames, dataset_directory, size=48)

    # Unpack metadata

    orig_data_x   = [x for _, _, _, x, _ in orig_pairs]
    orig_data_nH  = [nH for _, _, _, _, nH in orig_pairs]

    num_samples = len(full_samples)
    for i, generated_sample in enumerate(full_samples):

        x_val = orig_data_x[i]
        nH_val = orig_data_nH[i]

        original_png_filename = Path(ordered_filenames[i])
        fits_filename = original_png_filename.stem + '.fits'
        ref_fits_path = fits_dir / fits_filename

        # x like 9700.0 -> "9700_0"
        x_parts = f"{x_val:.1f}".split('.')
        x_str = f"{x_parts[0]}_{x_parts[1]}"
        # nH like 2.21e+11 -> "2_21e_11"
        nH_str = f"{nH_val:.2e}".replace('.', '_').replace('+', '_')

        base_output_filename = f"SAMPLE_dust_x_{x_str}_nH_{nH_str}"

        fits_output_path = fits_output_dir / (base_output_filename + ".fits")
        npy_output_path  = npy_output_dir  / (base_output_filename + ".npy")

        # saving the fits file
        save_scaled_tensor_as_fits(
            tensor_=generated_sample,
            output_path=str(fits_output_path),
            ref_fits_path=str(ref_fits_path)
        )
        # saving the npy file
        np.save(npy_output_path, generated_sample.squeeze(0).cpu().numpy())

        print(f"\r[{mode_tag}] Saved sample {i+1}/{num_samples}", end="")

    print("")  # newline
    logging.info(f"[{mode_tag}] Finished saving {num_samples} samples.")

def main():
    cfg = SamplingConfig()
    if not cfg.model_names:
        raise ValueError("SamplingConfig.model_names must contain at least one checkpoint name.")

    # --- DDP & Device Setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", str(cfg.local_rank)))
    is_distributed, rank, world_size, local_rank = init_distributed(local_rank)
    device = get_device(local_rank)

    setup_logging(is_rank0(is_distributed, rank))

    if is_rank0(is_distributed, rank):
        logging.info(f"Using device: {device}")

    set_seed(cfg.noise_seed)
    if is_rank0(is_distributed, rank):
        logging.info("Random seed set for reproducibility.")

    cond_all, meta = build_conditions_from_dataset(cfg.dataset_directory, device=device)

    # Split for DDP
    if is_distributed:
        local_conds = torch.chunk(cond_all, world_size, dim=0)[rank]
    else:
        local_conds = cond_all

    root_path = Path(__file__).resolve().parent

    for model_name in cfg.model_names:
        model = ConditionalUnet(
            sample_size=cfg.image_size,
            in_channels=1,
            out_channels=1,
            num_conditions=2,
            condition_embed_dim=cfg.condition_embed_dim,
            label_hidden_dim=cfg.label_hidden_dim,
            predict_labels=cfg.predict_labels,
        ).to(device)

        checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{model_name}.pth")
        load_checkpoint(model, checkpoint_path, device)

        checkpoint_epoch_info = f"{MODEL_FOLDER_NAME}_{model_name[-5:-3]}k"
        logging.info(f"Saving to {checkpoint_epoch_info}")

        if is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        model.eval()

        ddpm_scheduler = build_scheduler_safe(cfg.ddpm_timesteps, kind="ddpm")
        local_samples_ddpm_ancombe = sample_grid(
            model,
            ddpm_scheduler,
            local_conds,
            image_size=cfg.image_size,
            device=device,
        )
        finalize_and_save_samples(
            "DDPM",
            local_samples_ddpm_ancombe,
            meta,
            cfg.dataset_directory,
            checkpoint_epoch_info,
            root_path,
            is_distributed,
            rank,
            world_size,
            anscombe_global_max=cfg.anscombe_global_max,
        )

        local_samples_ddim = sample_grid_DDIM(
            model,
            local_conds,
            image_size=cfg.image_size,
            device=device,
            steps=cfg.ddim_steps,
        )
        finalize_and_save_samples(
            "DDIM",
            local_samples_ddim,
            meta,
            cfg.dataset_directory,
            checkpoint_epoch_info,
            root_path,
            is_distributed,
            rank,
            world_size,
            anscombe_global_max=cfg.anscombe_global_max,
        )

    if is_distributed:
        cleanup()

if __name__ == "__main__":
    main()