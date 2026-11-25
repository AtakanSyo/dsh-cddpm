"""Saving and loading checkpoints. Loading also accepts checkpoints from run 377."""
import logging
from pathlib import Path

import torch

from .model import ConditionalUNet

# Parameter names used by the 377 code -> names used here.
_OLD_TO_NEW = {"model.": "unet.", "_label_mlp.": "label_head."}


def checkpoint_path(output_dir, epoch):
    return Path(output_dir) / f"conditional_unet_epoch_{epoch:05d}.pth"


def run_tag(checkpoint_dir):
    """Name used for sample folders: '377_..._Corrected' for a 377 outputs/
    folder, '378_..._Clean_model_C' for a 378 model_<variant> folder."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if checkpoint_dir.name == "outputs":
        return checkpoint_dir.parent.name
    return f"{checkpoint_dir.parent.name}_{checkpoint_dir.name}"


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, metadata):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
    }, path)
    logging.info("Saved checkpoint %s", path)


def _rename(key):
    key = key.removeprefix("module.")
    for old, new in _OLD_TO_NEW.items():
        if key.startswith(old):
            return new + key[len(old):]
    return key


def load_model(path, device, predict_labels=False):
    """Build a ConditionalUNet and load weights from a 377 or 378 checkpoint.

    Returns (model in eval mode, metadata, epoch). metadata always contains
    'condition_stats', 'use_anscombe' and 'data_max', which is what
    dsh.data.to_counts needs. The label head is only needed for training, so
    it is skipped by default.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = dict(checkpoint.get("metadata", {}))
    if "data_max" not in metadata:  # 377 checkpoints: Anscombe only
        if metadata.get("inverse") != "algebraic_anscombe" or "anscombe_global_max" not in metadata:
            raise ValueError(f"{path}: checkpoint lacks the normalisation metadata written by runs 377/378")
        metadata.update(use_anscombe=True, data_max=metadata["anscombe_global_max"])

    model = ConditionalUNet(image_size=int(metadata.get("config", {}).get("image_size", 48)),
                            predict_labels=predict_labels).to(device)
    state = {_rename(k): v for k, v in checkpoint["model"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if not predict_labels:
        unexpected = [k for k in unexpected if not k.startswith("label_head.")]
    if missing or unexpected:
        raise RuntimeError(f"{path}: architecture mismatch, missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, metadata, checkpoint.get("epoch")
