# utils/plotting.py

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import logging

def save_loss_plots(
    total_losses,
    noise_losses,
    label_losses,
    sbp_losses,
    kl_losses,         # <--- now used
    val_sbp_losses,
    out_dir
):
    """
    Save training and validation loss curves (Total, Noise, Label, SBP,
    KL loss, Validation SBP) with log-scaled y-axis.
    """
    os.makedirs(out_dir, exist_ok=True)

    def save_curve(values, label, fname, ylog=True):
        plt.figure(figsize=(6, 4))
        plt.plot(values, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)" if ylog else "Loss")
        plt.title(f"{label} Curve")
        if ylog:
            plt.yscale("log")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    # --- Individual curves ---
    save_curve(total_losses, "Total Loss", "total_loss_curve.png")
    save_curve(noise_losses, "Noise (MSE) Loss", "noise_loss_curve.png")

    if label_losses is not None and len(label_losses) > 0:
        save_curve(label_losses, "Label (MSE) Loss", "label_loss_curve.png")

    save_curve(sbp_losses, "Train SBP Loss", "train_sbp_loss_curve.png")

    if kl_losses is not None and len(kl_losses) > 0:
        save_curve(kl_losses, "KL Loss", "kl_loss_curve.png")   # <---

    save_curve(val_sbp_losses, "Validation SBP Loss", "val_sbp_loss_curve.png")

    # --- Combined plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(total_losses, label="Total Loss")
    plt.plot(noise_losses, label="Noise (MSE)")
    if label_losses is not None and len(label_losses) > 0:
        plt.plot(label_losses, label="Label (MSE)")
    plt.plot(sbp_losses, label="Train SBP")
    if kl_losses is not None and len(kl_losses) > 0:
        plt.plot(kl_losses, label="KL Loss")                    # <---
    plt.plot(val_sbp_losses, label="Validation SBP")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training & Validation Losses (Log Scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_losses.png"), dpi=200)
    plt.close()

    logging.info(f"[✓] Saved loss plots to: {out_dir}")
    

def save_image_grid(model, scheduler, device, c1, c2, epoch, cfg):
    num = cfg.num_samples
    cond = torch.tensor([[c1, c2]] * num, device=device)
    x = torch.randn(num, 1, cfg.image_size, cfg.image_size, device=device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            res = model(x, t, cond)
        x = scheduler.step(res, t, x).prev_sample
    x = x.clamp(-1,1).add(1).mul(0.5).cpu()
    grid = torchvision.utils.make_grid(x, nrow=int(num**0.5))
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0).numpy(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    os.makedirs(cfg.sample_dir, exist_ok=True)
    path = os.path.join(cfg.sample_dir, f"sample_{epoch}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()