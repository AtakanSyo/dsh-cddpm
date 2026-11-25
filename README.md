# dsh-cddpm

Conditional denoising diffusion probabilistic models (cDDPMs) that emulate
[RefleX](https://www.astro.unige.ch/reflex/) Monte Carlo simulations of X-ray
dust scattering halo (DSH) images, conditioned on the dust distance ratio *x*
and the hydrogen column density *N*<sub>H</sub>.

This repository accompanies the article *Conditional Diffusion Modeling of
Poisson-Distributed X-Ray Dust Scattering Halo Images* (Saraçyakupoğlu,
Kalemci, Kaya & Abbasi, submitted to *Astronomy and Computing*). It contains
the simulation pipeline, the RefleX datasets, the training and sampling code,
the trained models of the paper, and the code that reproduces the paper's data
figures and numbers.

Archived on Zenodo: [10.5281/zenodo.21823977](https://doi.org/10.5281/zenodo.21823977)
(this DOI always resolves to the latest release).

## Repository structure

```
simulations/
  code/                       RefleX pipeline: prepare, launch (SLURM) and gather runs over the (x, N_H) grid
  data/
    reflex_data_8_corrected_camera_grid/       training set: 32 x 32 grid, 1024 images (.npy + .fits), seed 42
    reflex_data_8_testing_seed287/             test set used for all model evaluation: 15 x 15 grid, 225 images, seed 287
    reflex_data_8_testing/                     second realization of the test grid, seed 42 (noise floor only)
    reflex_data_8_testing_singlescat_seed287/  single-scattering-only run of the test grid, seed 287
cddpm/
  config.py                   all settings; VARIANTS defines models A, B and C
  train.py                    train one variant
  sample_test_grid.py         one DDPM (+ optional DDIM) sample per test-grid condition
  sample_ensemble.py          many samples at fixed conditions
  smoke_test.py               quick CPU check of all three variants
  dsh/                        model, data handling, auxiliary loss, samplers, trainer, checkpoints
  slurm/                      job scripts used on the Tosun HPC cluster
  model_A/ model_B/ model_C/  trained models of the paper (see below)
analysis/
  make_paper_figures.py       reproduces the paper's data figures and numbers
  npy_data/38{0,1,2}_*_DDPM/  samples from every checkpoint (1k-20k epochs) of models A, B and C
  utils/                      radial binning, Poisson deviance, heatmaps, profile plots
  analysis_exploratory.ipynb  the working notebook, kept with its outputs for reference
```

### Random seeds of the RefleX datasets

The training set and `reflex_data_8_testing` were simulated with RefleX seed
42; `reflex_data_8_testing_seed287` and the single-scattering run use seed 287.
All model evaluation in the paper uses the **seed-287** test set, so the test
images share no Monte Carlo noise with the training images. The seed-42 test
grid is used only as the second, independent realization for the Poisson
noise floor.

## Models of the paper

| Model | Anscombe transform | Auxiliary Poisson-deviance loss | Selected checkpoint | Mean *D*<sub>P</sub> (seed-287 test set) |
|---|---|---|---|---|
| A | no  | no  | `cddpm/model_A/conditional_unet_epoch_19000.pth` | 14.31 |
| B | yes | no  | `cddpm/model_B/conditional_unet_epoch_13000.pth` | 2.10 |
| C | yes | yes | `cddpm/model_C/conditional_unet_epoch_15000.pth` | 1.48 |

Each `model_?/` folder also holds the run configuration (`run_config.json`),
the conditioning and data statistics, and the full training-loss log
(`loss_log.csv`, `losses.png`). Only the selected checkpoints are included.
All three variants were trained once, with seed 42.

Training settings (identical for all variants): 20 000 epochs, batch size 256,
Adam with learning rate 1e-4 decayed to 1e-6 by a cosine schedule,
gradient-norm clipping at 1.0, 1000-step linear noise schedule
(beta from 1e-4 to 0.02), label-loss weight 0.05, auxiliary-loss weight 1e-4
with Min-SNR-gamma weighting (gamma = 5).

## Requirements

Python 3.10+ with

```
torch diffusers numpy scipy pandas astropy matplotlib tqdm
```

## Usage

**Train** (from `cddpm/`)

```bash
python train.py --variant C          # A, B or C
```

Checkpoints and logs are written to `cddpm/model_<variant>/`. The data folder
is found automatically (`../simulations/data`); set `DSH_DATA_ROOT` to use
another location. `slurm/` contains the SLURM job scripts used for the paper.

**Sample** (from `cddpm/`)

```bash
python sample_test_grid.py --variant C --epochs 15000
```

This draws one DDPM (1000 steps) and one DDIM (50 steps) sample for each of
the 225 test conditions, converted back to photon counts with the inverse of
the transform the model was trained with (algebraic inverse Anscombe for B and
C, linear rescaling for A).

**Reproduce the paper's figures and numbers** (from `analysis/`)

```bash
python make_paper_figures.py
```

This writes the figures to `analysis/figures/` and the numbers quoted in the
paper (Table 2, the checkpoint curve, heatmap statistics, the chi-square
cross-check and the per-figure deviances) to `analysis/paper_numbers.json`.

**Simulate new data** (from `simulations/code/`)

`prepare_simulations.py` builds the parameter grid from
`template_params_par/params.par`, `run_all_sims.sh` and `run_reflex.sh` launch
RefleX on a SLURM cluster, and `gather_data.py` collects the images. RefleX
itself is not distributed here; it is available from its developers.

## Versions

* **v2.0**: corrected training code (the label-prediction head is now trained,
  and samples are converted with the algebraic inverse Anscombe transform),
  retrained models A, B and C, evaluation against the independent seed-287
  test set, and the figure-reproduction script. These are the results reported
  in the paper.
* **v1.0**: earlier code and analysis, superseded by v2.0.

## License

MIT (see `LICENSE`).
