"""Reproduce the data figures and numbers of the paper from this repository.

    cd analysis
    python make_paper_figures.py            # writes figures/ and paper_numbers.json

Inputs (all inside the repository):
  ../simulations/data/reflex_data_8_corrected_camera_grid   RefleX training set (seed 42)
  ../simulations/data/reflex_data_8_testing_seed287          RefleX test set used for all model evaluation (seed 287)
  ../simulations/data/reflex_data_8_testing                  second test realization (seed 42), for the noise floor
  ../simulations/data/reflex_data_8_testing_singlescat_seed287  single-scattering run, for the multiple-scattering study
  npy_data/38{0,1,2}_Model_{A,B,C}_Corrected_model_?_NNk_DDPM   model samples, checkpoints 1k-20k
  ../cddpm/model_{A,B,C}/loss_log.csv                          training losses

Selected checkpoints (minimum mean D_P against the seed-287 test set): A 19k, B 13k, C 15k.
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.io_npy import save_npy_to_npz
from utils.metrics import mean_error_between_npz
from utils.error_functions import (compare_npz_by_params_poisson_deviance, radial_reduced_chi2_area2_bin_filtered,
                                   radial_reduced_poisson_deviance_abs)
from utils.heatmaps import poisson_deviance_heatmap_between_npz

DATA = HERE.parent / "simulations" / "data"
CACHE = HERE / "npz_cache"
OUT = HERE / "figures"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

RUNS = {"A": "380_Model_A_Corrected_model_A", "B": "381_Model_B_Corrected_model_B", "C": "382_Model_C_Corrected_model_C"}
SELECTED = {"A": 19, "B": 13, "C": 15}
CHECKPOINTS = list(range(1, 21))


def quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def npz_from_folder(folder, name):
    """Convert a folder of .npy images to the NPZ format used by utils (cached)."""
    path = CACHE / f"{name}.npz"
    if not path.exists():
        quiet(save_npy_to_npz, str(folder), out_path=str(path), pattern="*.npy", k=1.0)
    return str(path)


TEST = npz_from_folder(DATA / "reflex_data_8_testing_seed287" / "matrices", "REFLEX_TEST_SEED287")
TEST_SEED42 = npz_from_folder(DATA / "reflex_data_8_testing" / "matrices", "REFLEX_TEST_SEED42")
SINGLE = npz_from_folder(DATA / "reflex_data_8_testing_singlescat_seed287" / "matrices", "REFLEX_SINGLESCAT_SEED287")


def model_npz(variant, checkpoint):
    return npz_from_folder(HERE / "npy_data" / f"{RUNS[variant]}_{checkpoint:02d}k_DDPM", f"MODEL_{variant}_{checkpoint:02d}k")


def save(name, dpi=300):
    fig = plt.figure(plt.get_fignums()[-1])
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print("saved", name)


RC = {"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
      "mathtext.fontset": "stix", "font.size": 11, "xtick.direction": "in", "ytick.direction": "in",
      "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.8, "legend.frameon": False}
HEATMAP = dict(n_annuli=24, diverging=True, cmap="magma", figsize=(9, 6), vmin=0.0, vmax=4.0,
               annotate_above=4, annotate_fontsize=13)
PROFILE = dict(n_annuli=24, pixel_scale_arcsec=45, a=200.0, center="geom", figsize=(10, 10), font_size=12)
numbers = {}


def heatmap(model, data, name, **extra):
    (grid, _, _), _ = quiet(poisson_deviance_heatmap_between_npz, npz_model_path=model, npz_data_path=data,
                            **HEATMAP, **extra)
    save(name)
    grid = np.asarray(grid, float)
    return {"mean": float(grid.mean()), "std": float(grid.std()), "max": float(grid.max())}


def profile(x, nh, paths, labels, name, **extra):
    _, text = quiet(compare_npz_by_params_poisson_deviance, x, nh, npz_paths=paths, labels=labels,
                    cmaps=["magma"] * len(paths), **PROFILE, **extra)
    save(name)
    return [line.strip() for line in text.splitlines() if "D_red" in line]


# ---- Section 2.3 and Appendix A: single vs multiple scattering (data only) ----
numbers["multiple_scattering_heatmap"] = heatmap(SINGLE, TEST_SEED42, "multiscat_heatmap.png", use_N=True)
for x, nh, name in [(3319.0, 1.00e22, "appA_x_3319_0_nH_1e_22.png"), (4506.0, 2.44e22, "appA_x_4506_0_nH_2_44e_22.png"),
                    (2429.0, 5.12e22, "appA_x_2429_0_nH_5_12e_22.png")]:
    # The figures plot only bins with > 5 photons; the D_P,shape quoted in the captions uses all bins.
    profile(x, nh, [SINGLE, TEST_SEED42], ["Single Scattering", "Multiple Scattering"], name, bin_filter=5, use_N=True)
    numbers[name] = profile(x, nh, [SINGLE, TEST_SEED42], ["Single Scattering", "Multiple Scattering"],
                            "_stats_only.png", bin_filter=0, use_N=True)
(OUT / "_stats_only.png").unlink(missing_ok=True)

# ---- Section 3.2: Poisson noise floor (two independent RefleX realizations) ----
numbers["noise_floor_heatmap"] = heatmap(TEST, TEST_SEED42, "poisson_noise_floor_heatmap.png")

# ---- Section 3.1: training losses ----
with plt.rc_context(RC):
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.2), sharex=True)
    for ax, variant in zip(axes, "ABC"):
        log = pd.read_csv(HERE.parent / "cddpm" / f"model_{variant}" / "loss_log.csv")
        ax.plot(log.epoch, log.total, color="#0072B2", lw=0.8, label="Total loss")
        ax.plot(log.epoch, log.noise_mse.rolling(20, min_periods=1).mean(), color="#E69F00", lw=0.9, ls="--",
                label="Noise MSE loss (20-epoch avg.)")
        ax.plot(log.epoch, log.label_mse, color="#009E73", lw=0.8, label="Label loss")
        if variant == "C":
            ax.plot(log.epoch, log.aux, color="#CC79A7", lw=0.8, label=r"Aux. $D_{\mathrm{P}}$")
        ax.set_yscale("log"); ax.set_ylabel("Loss"); ax.set_xlim(0, 20000)
        ax.text(0.97, 0.88, f"Model {variant}", transform=ax.transAxes, ha="right", fontsize=11)
    axes[-1].set_xlabel("Epoch")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "All_Loss_Plots.png", dpi=400, bbox_inches="tight"); plt.close(fig)
print("saved All_Loss_Plots.png")

# ---- Section 3.3: mean D_P per checkpoint, and Table 2 ----
kwargs = {"n_annuli": 24, "pixel_scale_arcsec": 45.0}
curve = {v: [quiet(mean_error_between_npz, npz_model_path=model_npz(v, c), npz_data_path=TEST,
                   error_func=radial_reduced_poisson_deviance_abs, err_kwargs=kwargs)[0][0] for c in CHECKPOINTS]
         for v in "ABC"}
numbers["mean_DP_per_checkpoint"] = {v: dict(zip(CHECKPOINTS, map(float, curve[v]))) for v in "ABC"}
numbers["table2"] = {v: {"selected_epoch": SELECTED[v] * 1000, "mean_DP": float(curve[v][SELECTED[v] - 1])} for v in "ABC"}
with plt.rc_context(RC):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for v, marker, color in [("A", "o", "#E69F00"), ("B", "s", "#009E73"), ("C", "^", "#D55E00")]:
        ax.plot(CHECKPOINTS, curve[v], marker=marker, markersize=4, linewidth=1.0, color=color, label=f"Model {v}")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=0.9, label="Noise floor", zorder=1)
    ax.set_xlabel(r"Checkpoint ($\times 10^{3}$ epochs)"); ax.set_ylabel(r"$\overline{D_{\mathrm{P}}}$"); ax.set_yscale("log")
    ax.set_xticks(CHECKPOINTS); ax.set_xticklabels([str(c) if c % 2 == 0 or c == 1 else "" for c in CHECKPOINTS])
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.02), ncol=4, fontsize=9, handlelength=1.6,
               columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "mean_error_over_checkpoints.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("saved mean_error_over_checkpoints.png")

# ---- Section 3.5: deviance maps for B and C ----
for v in "BC":
    numbers[f"heatmap_{v}"] = heatmap(model_npz(v, SELECTED[v]), TEST, f"error_heatmap_model_{v.lower()}_{SELECTED[v]}k.png")

# ---- Section 4.2: chi2 cross-check ----
kwargs_chi2 = {"n_annuli": 24, "pixel_scale_arcsec": 45.0, "Fano_Data": 1, "Fano_Model": 1, "use_N": False, "bin_filter": 10}
numbers["chi2_red"] = {v: float(quiet(mean_error_between_npz, npz_model_path=model_npz(v, SELECTED[v]), npz_data_path=TEST,
                                      error_func=radial_reduced_chi2_area2_bin_filtered, err_kwargs=kwargs_chi2)[0][0])
                       for v in "ABC"}

# ---- Figure 10 and Appendix C: radial profiles ----
A, C = model_npz("A", SELECTED["A"]), model_npz("C", SELECTED["C"])
numbers["fig10"] = profile(2429.0, 2.63e21, [A, C, TEST], ["Model A", "Model C", "RefleX"],
                           "x_2429_0_nH_2_63e_21_Model_A_and_C_Comparison.png", use_N=False)
numbers["appC_1"] = profile(2429.0, 2.63e21, [C, TEST], ["Model C", "RefleX"], "appC_x_2429_0_nH_2_63e_21.png", use_N=False)
numbers["appC_2"] = profile(1835.0, 7.43e21, [C, TEST], ["Model C", "RefleX"], "appC_x_1835_0_nH_7_43e_21.png", use_N=False)
numbers["appC_3"] = profile(3912.0, 1.00e22, [C, TEST], ["Model C", "RefleX"], "appC_x_3912_0_nH_1e_22.png",
                            use_N=False, bin_filter=10)

(HERE / "paper_numbers.json").write_text(json.dumps(numbers, indent=1))
print("Table 2:", {v: round(numbers["table2"][v]["mean_DP"], 2) for v in "ABC"})
print("wrote paper_numbers.json")
