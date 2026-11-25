"""All settings for this run. Change values here, not inside the code."""
import os
from dataclasses import dataclass
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
RUN_NAME = RUN_DIR.name

# The paper's ablation (Table 1): which components each model uses.
VARIANTS = {
    "A": dict(use_anscombe=False, use_aux_loss=False),   # raw photon counts
    "B": dict(use_anscombe=True, use_aux_loss=False),    # + Anscombe transform
    "C": dict(use_anscombe=True, use_aux_loss=True),     # + auxiliary Poisson deviance
}


def find_data_root() -> Path:
    """Folder that contains the dataset folders.

    Order: the DSH_DATA_ROOT environment variable, then this repository's
    layout (../simulations/data), then a run folder inside a checkout
    (../../simulations/data), then the Tosun layout (../data).
    """
    override = os.environ.get("DSH_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    candidates = (RUN_DIR.parent / "simulations" / "data",
                  RUN_DIR.parents[1] / "simulations" / "data",
                  RUN_DIR.parent / "data")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


DATA_ROOT = find_data_root()


def model_dir(variant):
    """Everything a trained variant produces: checkpoints, stats, loss log, plots, previews."""
    return RUN_DIR / f"model_{variant}"


@dataclass
class TrainConfig:
    variant: str = "C"

    # Data
    dataset: str = "reflex_data_8_corrected_camera_grid"
    image_size: int = 48
    batch_size: int = 256
    num_workers: int = 8
    seed: int = 42

    # Optimisation (Adam, cosine decay to 1% of the initial rate)
    n_epochs: int = 20000
    lr: float = 1e-4
    grad_clip: float = 1.0

    # Loss = lambda_noise * noise MSE + lambda_label * label MSE + lambda_aux * aux deviance
    lambda_noise: float = 1.0
    lambda_label: float = 0.05
    lambda_aux: float = 1e-4
    snr_gamma: float = 5.0         # Min-SNR cap for the auxiliary loss weight

    # Bookkeeping
    checkpoint_every: int = 1000
    log_every: int = 200
    preview_every: int = 30
    preview_images: int = 16
    preview_ddim_steps: int = 2

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {self.variant!r}")

    @property
    def use_anscombe(self):
        return VARIANTS[self.variant]["use_anscombe"]

    @property
    def use_aux_loss(self):
        return VARIANTS[self.variant]["use_aux_loss"]
