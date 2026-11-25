"""Train one model of the paper's ablation. Other settings are in config.py.

    python train.py --variant A     raw counts, no auxiliary loss
    python train.py --variant B     Anscombe transform, no auxiliary loss
    python train.py --variant C     Anscombe transform + auxiliary loss
    (or: sbatch slurm/train.sh --variant C)

Each variant writes to its own folder, model_<variant>/.
"""
import argparse
import logging

from config import DATA_ROOT, VARIANTS, TrainConfig, model_dir
from dsh.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig(variant=args.variant)
    output_dir = model_dir(cfg.variant)

    data = DATA_ROOT / cfg.dataset / "matrices"
    if not data.is_dir():
        raise SystemExit(f"Training data not found: {data}\n"
                         "Set DSH_DATA_ROOT to the folder that contains the dataset folders.")
    if any(output_dir.glob("conditional_unet_epoch_*.pth")):
        raise SystemExit(f"{output_dir} already has checkpoints; move them away to start a new run.")

    trainer = Trainer(cfg, DATA_ROOT, output_dir)
    trainer.train()
    logging.info("Training finished")


if __name__ == "__main__":
    main()
