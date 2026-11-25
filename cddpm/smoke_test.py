"""Quick CPU check of the whole pipeline. Writes only to a temporary folder.

    python smoke_test.py
"""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import DATA_ROOT, TrainConfig
from dsh.checkpoint import checkpoint_path, load_model, save_checkpoint
from dsh.data import (anscombe, inverse_anscombe, matrix_files, normalize_condition, parse_filename,
                      to_counts, to_model_space)
from dsh.loss import aux_poisson_deviance
from dsh.sampling import ddpm_scheduler, sample_ddpm
from dsh.trainer import Trainer


def numpy_deviance(predicted, reference):
    """Independent NumPy version of the auxiliary loss, on photon counts."""
    h, w = reference.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    annulus = np.clip(np.digitize(r, np.linspace(0, r.max(), 25)) - 1, 0, 23).ravel()
    y = np.bincount(annulus, weights=reference.ravel(), minlength=24)
    mu = np.bincount(annulus, weights=predicted.ravel(), minlength=24)
    active, has_data = (y > 0) | (mu > 0), y > 0
    y, mu = y + 0.5, mu + 0.5
    mu = mu * y[has_data].sum() / mu[has_data].sum()
    terms = np.maximum(2 * (mu - y + y * np.log(y / mu)), 0)
    return terms[active].sum() / active.sum()


def check_loss():
    rng = np.random.default_rng(7)
    scale = 80.0
    counts_ref = torch.from_numpy(rng.poisson(0.4, (2, 1, 48, 48))).double()
    counts_pred = torch.from_numpy(rng.poisson(0.6, (2, 1, 48, 48))).double()
    counts_ref[1] = 0
    counts_ref[1, 0, 23, 23] = 10          # one bright pixel, empty wings
    to_model = lambda c: anscombe(c) / scale * 2 - 1
    predicted = to_model(counts_pred).requires_grad_()
    loss = aux_poisson_deviance(predicted, to_model(counts_ref), scale)
    expected = [numpy_deviance(to_counts(p, scale)[0].detach().numpy(), to_counts(r, scale)[0].numpy())
                for p, r in zip(predicted, to_model(counts_ref))]
    np.testing.assert_allclose(loss.detach().numpy(), expected, rtol=1e-9)
    loss.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    same = aux_poisson_deviance(to_model(counts_ref), to_model(counts_ref), scale)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-9, rtol=0)
    counts = torch.tensor([0.0, 1, 2, 5, 10, 100, 1516], dtype=torch.float64)
    torch.testing.assert_close(inverse_anscombe(anscombe(counts)), counts)
    print("PASS loss: matches NumPy, zero for identical images, finite gradients, Anscombe round trip")


def check_pipeline(variant):
    cpu = torch.device("cpu")
    with tempfile.TemporaryDirectory(prefix=f"dsh378-smoke-{variant}-") as tmp:
        tmp = Path(tmp)
        cfg = replace(TrainConfig(variant=variant), batch_size=2, num_workers=0, n_epochs=1, preview_images=2)
        trainer = Trainer(cfg, DATA_ROOT, tmp / "model", device=cpu)
        images = trainer.loader.dataset.images
        assert images.min() >= -1.0 - 1e-6 and images.max() <= 1.0 + 1e-6

        head_before = [p.detach().clone() for p in trainer.model.label_head.parameters()]
        batch, conditions = next(iter(trainer.loader))
        for _ in range(2):
            losses = trainer.train_step(batch, conditions)
            assert all(np.isfinite(v) for v in losses.values()), losses
        assert (losses["aux"] != 0) == cfg.use_aux_loss, losses
        assert all(not torch.equal(a, b) for a, b in zip(head_before, trainer.model.label_head.parameters()))

        trainer.loader = [(batch, conditions)]
        trainer.train()
        trainer.save_preview(1)
        path = checkpoint_path(tmp / "model", 1)
        save_checkpoint(path, trainer.model, trainer.optimizer, trainer.lr_scheduler, 1, trainer.metadata)
        model, metadata, _ = load_model(path, cpu)
        assert metadata["use_anscombe"] == cfg.use_anscombe
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, trainer.model.state_dict()[name])

        rows = [parse_filename(p.name) for p in matrix_files(DATA_ROOT / "reflex_data_8_testing" / "matrices")]
        assert len(rows) == 225, len(rows)
        cond = torch.tensor([normalize_condition(x, nh, metadata["condition_stats"]) for x, nh in rows[:2]],
                            dtype=torch.float32)
        counts = to_counts(sample_ddpm(model, cond, ddpm_scheduler(8), progress=False),
                           metadata["data_max"], metadata["use_anscombe"])
        assert counts.shape == (2, 1, 48, 48) and torch.isfinite(counts).all() and (counts >= 0).all()
        print(f"PASS model {variant} (Anscombe {cfg.use_anscombe}, aux {cfg.use_aux_loss}): "
              f"training, preview, checkpoint round trip, 8-step sampling | losses {losses}")


def check_transforms():
    counts = torch.tensor([0.0, 1, 2, 5, 10, 100, 1516], dtype=torch.float64)
    for use_anscombe, data_max in ((True, 78.0), (False, 1516.0)):
        model_values = to_model_space(counts, data_max, use_anscombe)
        assert model_values.min() >= -1 and model_values.max() <= 1
        torch.testing.assert_close(to_counts(model_values, data_max, use_anscombe), counts)
    print("PASS transforms: counts -> [-1, 1] -> counts, with and without Anscombe")


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_loss()
    check_transforms()
    for variant in ("A", "B", "C"):
        check_pipeline(variant)
    print("ALL CHECKS PASSED (nothing written outside a temporary folder)")
