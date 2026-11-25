"""Radial profiles and the auxiliary Poisson-deviance loss (paper Section 2.8).

Numerically identical to utils/aux.py:radial_poisson_deviance_profile_per_sample
in runs 374 and 377.
"""
from functools import lru_cache

import torch

from .data import to_counts

N_ANNULI = 24
PSEUDOCOUNT = 0.5


@lru_cache(maxsize=8)
def _annulus_index_cpu(height, width, n_annuli):
    y = torch.arange(height, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, width)
    radius = torch.sqrt((x - (width - 1) / 2.0) ** 2 + (y - (height - 1) / 2.0) ** 2)
    edges = torch.linspace(0, radius.max(), n_annuli + 1)
    return (torch.bucketize(radius.reshape(-1), edges) - 1).clamp_(0, n_annuli - 1)


def annulus_index(height, width, n_annuli=N_ANNULI, device=None):
    """Annulus number of every pixel: equal-width rings out to the image corner."""
    return _annulus_index_cpu(height, width, n_annuli).to(device or "cpu")


def annulus_counts(images, index, n_annuli=N_ANNULI):
    """Sum of each (B, 1, H, W) image in each annulus -> (B, n_annuli)."""
    flat = images.view(images.shape[0], -1)
    sums = flat.new_zeros((images.shape[0], n_annuli))
    return sums.scatter_add_(1, index.view(1, -1).expand(images.shape[0], -1), flat)


def aux_poisson_deviance(predicted, reference, data_max, use_anscombe=True, n_annuli=N_ANNULI):
    """One-sample, shape-only Poisson deviance between radial profiles.

    predicted, reference: images in model space [-1, 1], shape (B, 1, H, W).
    The predicted annular counts mu_i are the expectation and the reference
    counts y_i are the data. Steps:
      1. add a pseudocount of 0.5 to both, so the logarithm stays finite;
      2. scale mu by N = sum(y) / sum(mu) over annuli with y_i > 0
         (no gradient through N), so only the profile shape is compared;
      3. sum 2 [N mu - y + y ln(y / N mu)] over annuli that are non-empty in
         at least one image, and divide by their number.
    Returns one value per sample, shape (B,).
    """
    _, _, height, width = predicted.shape
    index = annulus_index(height, width, n_annuli, predicted.device)
    mu = annulus_counts(to_counts(predicted, data_max, use_anscombe), index, n_annuli)
    y = annulus_counts(to_counts(reference, data_max, use_anscombe), index, n_annuli)

    pixels_per_annulus = torch.bincount(index, minlength=n_annuli).to(predicted)
    exists = (pixels_per_annulus > 0).unsqueeze(0)
    active = exists & ((y > 0) | (mu > 0))
    has_data = (exists & (y > 0)).to(predicted.dtype)

    mu = mu + PSEUDOCOUNT
    y = y + PSEUDOCOUNT
    scale = ((y * has_data).sum(1) / (mu * has_data).sum(1).clamp_min(1e-12)).detach()
    mu = scale.unsqueeze(1) * mu

    terms = 2.0 * (mu - y + y * torch.log(y.clamp_min(1e-12) / mu))
    terms = terms.clamp_min(0.0) * active.to(predicted.dtype)
    return terms.sum(1) / active.sum(1).clamp_min(1).to(predicted.dtype)
