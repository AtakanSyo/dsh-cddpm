import torch
from tqdm import tqdm
import logging
from diffusers import DDPMScheduler, DDIMScheduler
import os
from torchvision.utils import save_image
import numpy as np

def make_scheduler(timesteps: int = 1000, kind = ""):

    if kind == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",    # <-- important
        )
    elif kind == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",    # <-- important
        )
    scheduler.set_timesteps(timesteps)  # explicit inference timesteps
    logging.info(f"DDPMScheduler initialized with {timesteps} steps.")
    return scheduler

def save_sampled_images(samples, conditions, out_dir="SAMPLED_GRID"):
    """
    Save each generated sample into a PNG file with its condition values in the filename.

    Args:
        samples (Tensor): Shape (N, 1, H, W), values in [0,1].
        conditions (Tensor): Shape (N, num_conditions), each row is the conditioning vector.
        out_dir (str): Directory to save the PNG files.
    """
    # Directory where main.py is located
    # Get directory one level above main.py
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

    save_dir = os.path.join(parent_dir, out_dir)
    os.makedirs(save_dir, exist_ok=True)

    samples = samples.detach().cpu()
    conditions = conditions.detach().cpu()

    for idx, (img, cond) in enumerate(zip(samples, conditions)):
        # Create a filename like: sample_idx_cond1=value1_cond2=value2.png
        cond_str = "_".join(f"{c:.4g}" for c in cond)
        filename = f"sample_{idx:04d}_{cond_str}.png"
        filepath = os.path.join(save_dir, filename)
        save_image(img, filepath)

@torch.no_grad()
def sample_grid(
    model,
    scheduler,                 # DDPMScheduler (already set_timesteps)
    conditions: torch.Tensor,  # (N, num_conditions)
    image_size: int = 48,
    device="cpu",
    rank: int = 0,
    is_distributed: bool = False,
):
    """
    Run the DDPM sampling loop for the given conditions.

    Returns:
        Tensor in [0,1], shape (N, 1, H, W).
    """
    # optional rank-0-only progress bar
    show_progress = (not is_distributed) or (rank == 0)

    # move/format inputs
    conditions = conditions.to(device=device, dtype=torch.float32)
    N = int(conditions.shape[0])
    x = torch.randn(N, 1, image_size, image_size, device=device, dtype=torch.float32)

    # ensure scheduler has timesteps set
    if not hasattr(scheduler, "timesteps") or len(scheduler.timesteps) == 0:
        if isinstance(scheduler, DDPMScheduler):
            scheduler.set_timesteps(getattr(scheduler.config, "num_train_timesteps", 1000))
        else:
            scheduler.set_timesteps(1000)

    was_training = model.training
    model.eval()

    try:
        iterator = scheduler.timesteps
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling (DDPM)", dynamic_ncols=True)

        for t in iterator:
            t_int = int(t) if not isinstance(t, int) else t
            t_batch = torch.full((N,), t_int, device=device, dtype=torch.long)

            # your model returns a dict; extract epsilon (noise) tensor
            out = model(x, t_batch, conditions)
            eps = out["noise"] if isinstance(out, dict) else out

            # DDPM step (keyword args)
            step = scheduler.step(model_output=eps, timestep=t_int, sample=x)
            x = step.prev_sample

    finally:
        if was_training:
            model.train()

    # map [-1,1] → [0,1] once, clamp for safety
    norm_x = ((x + 1.0) * 0.5).clamp(0, 1)

    # keep your existing saver call if you like
    # save_sampled_images(norm_x, conditions)

    return norm_x


@torch.no_grad()
def sample_grid_DDIM(
    model,
    conditions: torch.Tensor,
    image_size: int = 48,
    device="cuda",
    steps: int = 50,      # inference steps (subsampled from 1000)
):
    # 1) recreate the *training* schedule exactly
    ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")
    alphas_bar = ddpm.alphas_cumprod.to(device)         # (1000,)
    sqrt_ab    = torch.sqrt(alphas_bar)
    sqrt_1mab  = torch.sqrt(1.0 - alphas_bar)

    # 2) noise start
    B = conditions.shape[0]
    x = torch.randn(B, 1, image_size, image_size, device=device, dtype=torch.float32)

    # 3) choose decreasing timestep indices (e.g., 50 steps)
    ts = np.linspace(0, len(alphas_bar)-1, steps, dtype=int)[::-1].copy()
    ts_prev = np.append(ts[1:], -1)

    conditions = conditions.to(device=device, dtype=torch.float32)

    for t, t_prev in zip(ts, ts_prev):
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)

        # model predicts epsilon at x_t
        eps = model(x, t_batch, conditions)
        if isinstance(eps, dict): eps = eps["noise"]

        # x0 estimate from epsilon
        x0 = (x - sqrt_1mab[t] * eps) / (sqrt_ab[t] + 1e-8)

        # alpha_{t-1}
        if t_prev >= 0:
            ab_prev = alphas_bar[t_prev]
        else:
            ab_prev = torch.tensor(1.0, device=device)

        # deterministic DDIM update (η=0)
        x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps

    return ((x + 1.0) * 0.5).clamp(0, 1)