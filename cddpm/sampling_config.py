import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SamplingConfig:
    noise_seed: int = 42
    ddpm_timesteps: int = 1000
    ddim_steps: int = 50
    dataset_directory: str = "reflex_data_8_testing"
    local_rank: int = 0
    random_samples: int = 0

    checkpoint_dir: str = "outputs"
    global_stats_path: str = "outputs/global_data_stats.csv"
    model_names: tuple[str, ...] = (
        "conditional_unet_epoch_07000",
        "conditional_unet_epoch_06000",
        "conditional_unet_epoch_05000",
        "conditional_unet_epoch_04000",
        "conditional_unet_epoch_03000",
        "conditional_unet_epoch_02000",
    )

    image_size: int = 48
    anscombe_global_max: float = field(init=False)
    condition_embed_dim: int = 16
    label_hidden_dim: int = 128
    predict_labels: bool = False

    def __post_init__(self):
        self.anscombe_global_max = self._load_anscombe_global_max()

    def _load_anscombe_global_max(self) -> float:
        stats_path = Path(self.global_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Could not find global data stats file at {stats_path}. "
                "Run training first so outputs/global_data_stats.csv is created."
            )

        with stats_path.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if row.get("max_global")]

        if not rows:
            raise ValueError(f"No max_global values found in {stats_path}.")

        return float(rows[0]["max_global"])