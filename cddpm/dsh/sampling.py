"""DDPM and DDIM samplers. Both return images in model space [-1, 1]."""
import numpy as np
import torch
from diffusers import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

TRAIN_STEPS = 1000


def ddpm_scheduler(inference_steps=TRAIN_STEPS):
    """The training noise schedule (1000 linear steps), used for sampling."""
    scheduler = DDPMScheduler(num_train_timesteps=TRAIN_STEPS, beta_schedule="linear",
                              prediction_type="epsilon")
    scheduler.set_timesteps(inference_steps)
    return scheduler


@torch.no_grad()
def sample_ddpm(model, conditions, scheduler, image_size=48, generators=None, progress=True):
    """Ancestral DDPM sampling.

    generators: optional list with one torch.Generator per sample, for
    reproducible independent realizations. Without it the global RNG is used.
    """
    n = conditions.shape[0]
    shape = (n, 1, image_size, image_size)
    device = conditions.device
    if generators is None:
        x = torch.randn(shape, device=device, dtype=torch.float32)
    else:
        x = randn_tensor(shape, generator=generators, device=device, dtype=torch.float32)

    steps = tqdm(scheduler.timesteps, desc="DDPM", dynamic_ncols=True) if progress else scheduler.timesteps
    for t in steps:
        t = int(t)
        eps = model(x, torch.full((n,), t, device=device, dtype=torch.long), conditions)["noise"]
        if generators is None:
            x = scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample
        else:
            x = scheduler.step(eps, t, x, generator=generators).prev_sample
    return x


@torch.no_grad()
def sample_ddim(model, conditions, steps=50, image_size=48):
    """Deterministic DDIM (eta = 0) on `steps` evenly spaced training timesteps."""
    alpha_bar = DDPMScheduler(num_train_timesteps=TRAIN_STEPS, beta_schedule="linear",
                              prediction_type="epsilon").alphas_cumprod.to(conditions.device)
    n = conditions.shape[0]
    x = torch.randn(n, 1, image_size, image_size, device=conditions.device, dtype=torch.float32)

    timesteps = np.linspace(0, TRAIN_STEPS - 1, steps, dtype=int)[::-1].copy()
    previous = np.append(timesteps[1:], -1)
    for t, t_prev in zip(timesteps, previous):
        eps = model(x, torch.full((n,), int(t), device=conditions.device, dtype=torch.long), conditions)["noise"]
        x0 = (x - torch.sqrt(1.0 - alpha_bar[t]) * eps) / (torch.sqrt(alpha_bar[t]) + 1e-8)
        alpha_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=conditions.device)
        x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1.0 - alpha_prev) * eps
    return x
