# dsh-cddpm

Conditional diffusion model for simulating X-ray dust-scattering halos (DSH), paired with the [RefleX] (Stephane Paltani) radiative transfer simulations used to generate its training data.

## Repository structure

```
simulations/   RefleX simulation pipeline
cddpm/         Conditional DDPM model: training, sampling, and shared utilities
analysis/      Notebook and utilities for evaluating model samples against RefleX ground truth
```

### `simulations/`
- `code/` — scripts to prepare, launch (SLURM), and gather output from RefleX simulation runs across a grid of physical parameters (dust distance `x`, column density `nH`) on TUBITAK's TRUBA HPC.
- `data/` — resulting photon-count matrices (`.npy`) and FITS images (`.fits`) per parameter grid point.

### `cddpm/`
- `configs.py` / `sampling_config.py` — training and sampling configuration.
- `diffusion_model.py` — training entry point.
- `save_samples.py` — generates samples from a trained checkpoint.
- `utils/`, `testing_helpers/` — model, dataset, trainer, and sampling utilities.
- `run_model.sh` / `save_samples.sh` — SLURM job scripts for training/sampling on Sabanci University's
Tosun HPC.

### `analysis/`
- `analysis.ipynb` — data analysis pipeline used for generating plots used for this paper.
- `npy_data/` — sample/ground-truth matrices stored as NPY files.
- `npz_data/` — sample/ground-truth matrices stored as NPZ files.

## Requirements

Python 3.10+, with:

```
torch torchvision diffusers numpy scipy astropy matplotlib pillow tqdm pyyaml
```

## Usage

1. **Simulate data** — run the RefleX pipeline in `simulations/code/` to produce photon-count matrices in `simulations/data/`.
2. **Train** — from `cddpm/`, adjust `configs.py` as needed and run `python diffusion_model.py` (or submit `run_model.sh`).
3. **Sample** — adjust `sampling_config.py` and run `python save_samples.py` (or submit `save_samples.sh`) to draw samples from a trained checkpoint.
4. **Analyze** — open `analysis/analysis.ipynb` to compare samples against RefleX data.
