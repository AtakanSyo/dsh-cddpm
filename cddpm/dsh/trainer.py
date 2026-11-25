"""Training loop (single GPU)."""
import csv
import json
import logging
import math
import random
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import checkpoint_path, save_checkpoint
from .data import HaloDataset, condition_stats, global_stats
from .loss import aux_poisson_deviance
from .model import ConditionalUNet

LOSS_NAMES = ("total", "noise_mse", "label_mse", "aux")


class Trainer:
    def __init__(self, cfg, data_root, output_dir, device=None):
        """output_dir receives checkpoints, stats, loss log, plots and previews."""
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.preview_dir = self.output_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._load_data(Path(data_root) / cfg.dataset / "matrices")

        self.model = ConditionalUNet(image_size=cfg.image_size, predict_labels=True).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01)
        self.noise_schedule = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        # TF32 matrix maths on recent NVIDIA GPUs (still FP32 storage).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.history = {name: [] for name in LOSS_NAMES}

    # ------------------------------------------------------------------
    def _load_data(self, folder):
        self.stats = condition_stats(folder)
        mean, std, self.data_max = global_stats(folder, self.cfg.use_anscombe)
        logging.info("Model %s (Anscombe: %s, aux loss: %s) | condition stats %s | data max %.6f",
                     self.cfg.variant, self.cfg.use_anscombe, self.cfg.use_aux_loss, self.stats, self.data_max)

        (self.output_dir / "condition_stats.json").write_text(json.dumps(self.stats, indent=4))
        (self.output_dir / "global_data_stats.csv").write_text(
            f"mean_global,std_global,max_global\n{mean},{std},{self.data_max}\n")
        config = {**asdict(self.cfg), "use_anscombe": self.cfg.use_anscombe, "use_aux_loss": self.cfg.use_aux_loss}
        (self.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
        self.metadata = {
            "config": config,
            "condition_stats": self.stats,
            "use_anscombe": self.cfg.use_anscombe,
            "data_max": self.data_max,
            "auxiliary": "aux_poisson_deviance (identical to 374/377)" if self.cfg.use_aux_loss else "none",
            "versions": {name: version(name) for name in ("torch", "torchvision", "diffusers", "numpy")},
        }
        if self.cfg.use_anscombe:  # keys read by 377-era tools
            self.metadata.update(anscombe_global_max=self.data_max, inverse="algebraic_anscombe")

        dataset = HaloDataset(folder, self.data_max, self.stats, self.cfg.image_size, self.cfg.use_anscombe)
        self.loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True,
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=self.device.type == "cuda",
                                 persistent_workers=self.cfg.num_workers > 0)

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        for epoch in range(1, cfg.n_epochs + 1):
            self.model.train()
            sums = dict.fromkeys(LOSS_NAMES, 0.0)
            n_seen = 0
            for images, conditions in tqdm(self.loader, desc=f"Epoch {epoch}"):
                losses = self.train_step(images.to(self.device, non_blocking=True),
                                         conditions.to(self.device, non_blocking=True))
                for name in LOSS_NAMES:
                    sums[name] += losses[name] * images.shape[0]
                n_seen += images.shape[0]

            if epoch % cfg.preview_every == 0:
                self.save_preview(epoch)

            for name in LOSS_NAMES:
                self.history[name].append(sums[name] / max(n_seen, 1))
            logging.info("Epoch %d | total %.4f | noise %.4f | label %.4f | aux %.4f | lr %.10f",
                         epoch, *(self.history[name][-1] for name in LOSS_NAMES),
                         self.optimizer.param_groups[0]["lr"])
            self.lr_scheduler.step()

            if epoch % cfg.log_every == 0:
                self.save_loss_log()
            if epoch % cfg.checkpoint_every == 0:
                save_checkpoint(checkpoint_path(self.output_dir, epoch), self.model, self.optimizer,
                                self.lr_scheduler, epoch, self.metadata)
        self.save_loss_log()

    def train_step(self, x, conditions):
        """One optimisation step. Returns the (weighted) loss terms as floats."""
        cfg = self.cfg
        t = torch.randint(0, 1000, (x.shape[0],), device=self.device).long()
        noise = torch.randn_like(x)
        noisy = self.noise_schedule.add_noise(x, noise, t)

        out = self.model(noisy, t, conditions)
        noise_loss = cfg.lambda_noise * F.mse_loss(out["noise"], noise)
        label_loss = cfg.lambda_label * F.mse_loss(out["labels"], conditions)

        total = noise_loss + label_loss
        aux_loss = torch.zeros(())
        if cfg.use_aux_loss:
            # Clean-image estimate and its Min-SNR weight (paper eqs. 19-21).
            alpha_bar = self.noise_schedule.alphas_cumprod.to(x.device, dtype=x.dtype)[t].view(-1, 1, 1, 1)
            x0_hat = (noisy - torch.sqrt(1.0 - alpha_bar) * out["noise"]) / (torch.sqrt(alpha_bar) + 1e-8)
            snr = alpha_bar.view(-1) / (1.0 - alpha_bar.view(-1)).clamp_min(1e-8)
            weight = torch.minimum(snr, torch.as_tensor(cfg.snr_gamma, device=x.device, dtype=x.dtype))
            weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=1.0).detach()
            deviance = aux_poisson_deviance(x0_hat, x, self.data_max, cfg.use_anscombe)
            aux_loss = cfg.lambda_aux * (deviance * weight).mean()
            total = total + aux_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        return {"total": float(total.detach()), "noise_mse": float(noise_loss.detach()),
                "label_mse": float(label_loss.detach()), "aux": float(aux_loss.detach())}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def save_preview(self, epoch, seed=123):
        """Two-step DDIM denoising of a few training images at random noise levels."""
        was_training = self.model.training
        self.model.eval()
        images, conditions = next(iter(self.loader))
        x = images[:self.cfg.preview_images].to(self.device)
        conditions = conditions[:self.cfg.preview_images].to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed + epoch)
        t = torch.randint(0, 1000, (x.shape[0],), device=self.device, generator=generator).long()
        noisy = self.noise_schedule.add_noise(
            x, torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator), t)
        x0 = self._ddim_x0(noisy, t, conditions, self.cfg.preview_ddim_steps)

        rows = int(math.sqrt(max(1, x0.shape[0])))
        torchvision.utils.save_image(x0, self.preview_dir / f"preview_epoch_{epoch:05d}.png",
                                     normalize=True, value_range=(-1, 1), nrow=rows)
        low = x0.amin(dim=(1, 2, 3), keepdim=True)
        high = x0.amax(dim=(1, 2, 3), keepdim=True)
        torchvision.utils.save_image(((x0 - low) / (high - low).clamp(min=1e-8)).clamp(0, 1),
                                     self.preview_dir / f"preview_epoch_{epoch:05d}_stretched.png", nrow=rows)
        if was_training:
            self.model.train()

    def _ddim_x0(self, x, t, conditions, steps):
        """Deterministic DDIM jumps from t to 0 in `steps` model calls; returns x0 in [-1, 1]."""
        alpha_bar = self.noise_schedule.alphas_cumprod.to(x.device)
        start = t.long()
        for i in range(steps):
            eps = self.model(x, t, conditions)["noise"]
            a = alpha_bar[t].view(-1, 1, 1, 1)
            x0 = (x - torch.sqrt(1.0 - a) * eps) / (torch.sqrt(a) + 1e-8)
            if i == steps - 1:
                return x0.clamp(-1.0, 1.0)
            t = (start * (steps - 1 - i)) // steps
            a_next = alpha_bar[t].view(-1, 1, 1, 1)
            x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps

    def save_loss_log(self):
        with open(self.output_dir / "loss_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", *LOSS_NAMES])
            for epoch, row in enumerate(zip(*(self.history[n] for n in LOSS_NAMES)), start=1):
                writer.writerow([epoch, *row])

        fig, ax = plt.subplots(figsize=(8, 6))
        for name in LOSS_NAMES:
            if name != "aux" or self.cfg.use_aux_loss:
                ax.plot(self.history[name], label=name)
        ax.set(xlabel="Epoch", ylabel="Loss", yscale="log", title=f"Training losses, model {self.cfg.variant}")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "losses.png", dpi=200)
        plt.close(fig)
