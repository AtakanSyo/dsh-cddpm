import torch
from functools import lru_cache
import torch.nn.functional as F
import math

@lru_cache(maxsize=32)
def _cached_radial_bins(H, W, nbins, cx, cy):
    device = torch.device("cpu")  # store on CPU; move to target device on use
    y = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    x = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    rmap = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = rmap.max()
    edges = torch.linspace(0, max_r, nbins + 1, device=device)
    bin_idx = torch.bucketize(rmap.reshape(-1), edges) - 1
    bin_idx.clamp_(0, nbins - 1)
    return bin_idx  # CPU tensor

def _precompute_radial_bins(H, W, nbins=24, center=None, device=None):
    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center
    idx = _cached_radial_bins(H, W, nbins, float(cx), float(cy))
    return idx.to(device or torch.device("cpu"))

def compute_radial_profile(img, bin_idx, nbins=24):
    B, C, H, W = img.shape
    assert C == 1
    flat = img.view(B, -1)
    bin_sums = torch.zeros(B, nbins, device=img.device, dtype=flat.dtype)
    bin_sums.scatter_add_(1, bin_idx.view(1, -1).expand(B, -1), flat)
    return bin_sums

def bray_curtis(a, b, eps=1e-8):
    num = torch.sum(torch.abs(a - b), dim=1)
    den = torch.sum(torch.abs(a + b), dim=1) + eps
    return num / den


def radial_profile_loss_per_sample(
    pred_img, true_img, nbins=24, center=None, eps=1e-8,
    positive_mode="shift", normalize=False
):
    """
    Computes **MSE between radial profiles** (per-sample).
    - Enforces non-negativity (shift/relu/softplus).
    - Normalizes each profile to sum=1 if requested.
    """
    B, _, H, W = pred_img.shape
    device = pred_img.device
    bin_idx = _precompute_radial_bins(H, W, nbins, center, device)

    # --- Enforce non-negativity ---
    if positive_mode == "shift":
        pred = (pred_img + 1.0) * 0.5
        true = (true_img + 1.0) * 0.5
    elif positive_mode == "relu":
        pred = pred_img.relu()
        true = true_img.relu()
    elif positive_mode == "softplus":
        pred = F.softplus(pred_img)
        true = F.softplus(true_img)
    else:
        pred, true = pred_img, true_img

    # --- Compute radial profiles ---
    rp_pred = compute_radial_profile(pred, bin_idx, nbins)
    rp_true = compute_radial_profile(true, bin_idx, nbins)

    # --- Optional normalization (to treat profiles as distributions) ---
    if normalize:
        rp_pred = rp_pred / (rp_pred.sum(dim=1, keepdim=True) + eps)
        rp_true = rp_true / (rp_true.sum(dim=1, keepdim=True) + eps)

    # --- Mean Squared Error per sample ---
    mse_per_sample = ((rp_pred - rp_true) ** 2).mean(dim=1)
    return mse_per_sample


@lru_cache(maxsize=32)
def _cached_radial_angular_bins(H, W, nrad, nang, cx, cy):
    """
    Returns a flattened LongTensor of size (H*W,) with a unique bin index
    for each pixel: bin = r_bin * nang + a_bin, where
      - r_bin in [0, nrad-1] (radial ring)
      - a_bin in [0, nang-1] (angular sector)
    Cached on CPU to avoid recomputation across calls.
    """
    device = torch.device("cpu")
    y = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    x = torch.arange(W, device=device, dtype=torch.float32).view(1, W)

    dx = x - cx
    dy = y - cy

    rmap = torch.sqrt(dx ** 2 + dy ** 2)
    # atan2 returns (-pi, pi]; map to [0, 2*pi)
    theta = torch.atan2(dy, dx)
    two_pi = 2.0 * math.pi
    theta = (theta + two_pi) % two_pi

    # Edges
    max_r = rmap.max()
    r_edges = torch.linspace(0.0, max_r, nrad + 1, device=device)
    a_edges = torch.linspace(0.0, two_pi, nang + 1, device=device)

    # Bucketize
    rbin = torch.bucketize(rmap.reshape(-1), r_edges) - 1
    abin = torch.bucketize(theta.reshape(-1), a_edges) - 1

    rbin.clamp_(0, nrad - 1)
    abin.clamp_(0, nang - 1)

    # Combined bin index
    bin_idx = rbin * nang + abin  # (H*W,)
    return bin_idx.to(torch.long)  # CPU tensor

def _precompute_radial_angular_bins(H, W, nrad=24, nang=30, center=None, device=None):
    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center
    idx = _cached_radial_angular_bins(H, W, int(nrad), int(nang), float(cx), float(cy))
    return idx.to(device or torch.device("cpu"))

@torch.no_grad()
def _precompute_radial_angular_bins_with_offset(
    H, W, nrad=24, nang=30, center=None, device=None, theta0: float = 0.0
):
    """
    Build combined (r,a) bin index with a *continuous* angular offset theta0 (radians).
    Returns flat (H*W,) long tensor of indices: a_idx + nang * r_idx
    """
    device = device or torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    # radius and angle
    dx, dy = (xx - cx), (yy - cy)
    rr = torch.sqrt(dx * dx + dy * dy)                # (H, W)
    theta = torch.atan2(dy, dx)                       # (-pi, pi]
    theta = (theta + theta0) % (2.0 * math.pi)        # [0, 2pi)

    # radial bins
    r_max = rr.max()
    r_edges = torch.linspace(0.0, r_max, steps=int(nrad) + 1, device=device)
    r_idx = torch.bucketize(rr.reshape(-1), r_edges) - 1  # (H*W,)

    # angular bins
    a_edges = torch.linspace(0.0, 2.0 * math.pi, steps=int(nang) + 1, device=device)
    a_idx = torch.bucketize(theta.reshape(-1), a_edges) - 1

    # clamp (bucketize can return nbins at the upper edge)
    r_idx.clamp_(0, int(nrad) - 1)
    a_idx.clamp_(0, int(nang) - 1)

    bin_idx = a_idx + int(nang) * r_idx               # (H*W,)
    return bin_idx


def compute_radial_angular_profile(img, bin_idx, nrad=24, nang=30):
    """
    img: (B, 1, H, W), values typically in [0,1] after positivity handling
    bin_idx: (H*W,) combined r+a index
    Returns: (B, nrad, nang)
    """
    B, C, H, W = img.shape
    assert C == 1, "Expected single-channel images."
    flat = img.view(B, -1)  # (B, H*W)
    nbins = int(nrad) * int(nang)
    sums = torch.zeros(B, nbins, device=img.device, dtype=flat.dtype)
    sums.scatter_add_(1, bin_idx.view(1, -1).expand(B, -1), flat)
    return sums.view(B, int(nrad), int(nang))


def radial_angular_profile_loss_per_sample(
    pred_img,
    true_img,
    nrad=24,
    nang=30,
    center=None,
    eps=1e-8,
    positive_mode="shift",
    normalize=False,
):
    """
    MSE between radial+angular profiles with **continuous angular jitter** (per-batch).
    """
    B, C, H, W = pred_img.shape
    device = pred_img.device

    # --- draw continuous angular offset in [0, 2π) ---
    theta0 = float(torch.rand((), device=device) * (2.0 * math.pi))

    # recompute *angular* bins with θ -> (θ + θ0) mod 2π
    bin_idx = _precompute_radial_angular_bins_with_offset(
        H, W, nrad, nang, center, device, theta0=theta0
    )

    # positivity handling
    if positive_mode == "shift":
        pred = (pred_img + 1.0) * 0.5
        true = (true_img + 1.0) * 0.5
    elif positive_mode == "relu":
        pred = pred_img.relu()
        true = true_img.relu()
    elif positive_mode == "softplus":
        pred = F.softplus(pred_img)
        true = F.softplus(true_img)
    else:
        pred, true = pred_img, true_img

    # profiles
    rap_pred = compute_radial_angular_profile(pred, bin_idx, nrad, nang)
    rap_true = compute_radial_angular_profile(true, bin_idx, nrad, nang)

    # optional normalization (probability simplex)
    if normalize:
        rap_pred = rap_pred / (rap_pred.sum(dim=(1, 2), keepdim=True) + eps)
        rap_true = rap_true / (rap_true.sum(dim=(1, 2), keepdim=True) + eps)

    # per-sample MSE over bins
    mse_per_sample = ((rap_pred - rap_true) ** 2).mean(dim=(1, 2))
    return mse_per_sample

# --------------------------------------------------------------
# 1) Sum of photon counts per bin (SUMS, not means)
# --------------------------------------------------------------
def _counts_per_bin(img, bin_idx, n_bins):
    """
    img:     (B,1,H,W) photon counts (float)
    bin_idx: (H*W,) LongTensor with values [0 .. n_bins-1]
    returns: (B, n_bins) summed counts per bin
    """
    B, _, H, W = img.shape
    flat = img.view(B, -1)                  # (B, H*W)
    out = flat.new_zeros((B, n_bins))
    out.scatter_add_(1, bin_idx.expand(B, -1), flat)
    return out


def _kl_profile_per_sample(mu_bins, y_bins, *, eps=1e-8, tau=1e-3):
    """
    Per-sample KL(P||Q) between *distributions* over bins.
    Inputs:
      mu_bins, y_bins: (B, n_bins)  unnormalized bin sums
    Returns:
      kl_vec: (B,)  per-sample KL
    """
    # Laplace smoothing
    mu = mu_bins + tau
    y  = y_bins  + tau

    # Normalize to distributions (shape only)
    q = (mu / mu.sum(dim=1, keepdim=True).clamp_min(eps)).clamp_min(eps)  # pred
    p = (y  / y.sum(dim=1, keepdim=True).clamp_min(eps)).clamp_min(eps)   # true

    # Per-sample KL (no reduction across batch)
    kl_vec = (p * (p.log() - q.log())).sum(dim=1)  # (B,)
    return kl_vec


def auxiliary_kl_loss(
    pred_img,
    true_img,
    *,
    nrad,
    nang,
    center=None,
    eps=1e-8,
    tau=1e-3,
    positive_mode="shift",
    jitter_angle=True,
):
    """
    KL-on-SBP (radial×angular) SHAPE loss.
    Returns per-sample vector (B,) so you can weight/gate per-sample.
    """
    # ---- positivity only; no image-level normalization here ----
    if positive_mode == "shift":
        pred = torch.clamp((pred_img + 1.0) * 0.5, 0.0, 1.0)
        true = torch.clamp((true_img + 1.0) * 0.5, 0.0, 1.0)
    elif positive_mode == "relu":
        pred, true = pred_img.relu(), true_img.relu()
    elif positive_mode == "softplus":
        pred, true = F.softplus(pred_img), F.softplus(true_img)
    else:
        pred, true = pred_img, true_img

    B, _, H, W = pred.shape
    device = pred.device

    # optional angular jitter
    theta0 = float(torch.rand((), device=device) * (2.0 * math.pi)) if jitter_angle else 0.0
    bin_idx = _precompute_radial_angular_bins_with_offset(
        H, W, nrad=nrad, nang=nang, center=center, device=device, theta0=theta0
    )
    n_bins = int(nrad) * int(nang)

    # bin sums → (B, n_bins)
    mu_bins = _counts_per_bin(pred, bin_idx, n_bins)
    y_bins  = _counts_per_bin(true, bin_idx, n_bins)

    # per-sample KL
    kl_vec = _kl_profile_per_sample(mu_bins, y_bins, eps=eps, tau=tau)  # (B,)
    return kl_vec



PIXEL_SCALE_ARCSEC = 43.2  # arcsec / pixel

def _counts_per_bin(img: torch.Tensor, bin_idx: torch.Tensor, nbins: int) -> torch.Tensor:
    B, _, H, W = img.shape
    flat = img.view(B, -1)
    out = flat.new_zeros((B, nbins))
    out.scatter_add_(1, bin_idx.view(1, -1).expand(B, -1), flat)
    return out

def _inverse_anscombe_from_norm01(z01: torch.Tensor, anscombe_glob_max: float) -> torch.Tensor:
    a = z01 * float(anscombe_glob_max)
    x = (a / 2.0) ** 2 - 3.0 / 8.0
    return x.clamp_min(0.0)

def radial_chi2_profile_per_sample(
    pred_img: torch.Tensor,
    true_img: torch.Tensor,
    *,
    nbins: int = 24,
    center=None,
    inputs_are_neg1_to_1: bool = True,      # set False if images are already physical counts
    anscombe_glob_max: float | None = None, # required if inputs_are_neg1_to_1=True
    eps: float = 1e-8,
    tau_counts: float = 1e-3,               # Poisson variance floor on counts
    return_N: bool = False,
):
    """
    Reduced χ² on **surface-brightness profiles** (counts / arcsec²),
    using optimal multiplicative N and Poisson variance propagated to profiles.
    """
    B, C, H, W = pred_img.shape
    assert C == 1, "Expected single-channel images."
    device = pred_img.device

    # ---- Build annulus bins here (as requested) ----
    bin_idx = _precompute_radial_bins(H, W, nbins=nbins, center=center, device=device)  # (H*W,)

    # ---- Map to physical counts if needed ----
    if inputs_are_neg1_to_1:
        assert anscombe_glob_max is not None, "Provide anscombe_glob_max when inputs_are_neg1_to_1=True"
        pred01 = torch.clamp((pred_img + 1.0) * 0.5, 0.0, 1.0)
        true01 = torch.clamp((true_img + 1.0) * 0.5, 0.0, 1.0)
        pred_counts = _inverse_anscombe_from_norm01(pred01, anscombe_glob_max)
        true_counts = _inverse_anscombe_from_norm01(true01, anscombe_glob_max)
    else:
        pred_counts = pred_img.clamp_min(0)
        true_counts = true_img.clamp_min(0)

    # ---- Sum counts per annulus ----
    S_mu  = _counts_per_bin(pred_counts, bin_idx, nbins=nbins)   # (B, nbins)
    S_y   = _counts_per_bin(true_counts, bin_idx, nbins=nbins)   # (B, nbins)

    # ---- Annulus areas in arcsec^2 ----
    pix_counts = torch.bincount(bin_idx.to(torch.long), minlength=nbins).to(device=device, dtype=S_mu.dtype)
    A = (pix_counts * (PIXEL_SCALE_ARCSEC ** 2)).clamp_min(1e-12)           # (nbins,)

    # ---- Surface-brightness profiles: counts / arcsec^2 ----
    P_mu = S_mu / A.unsqueeze(0)
    P_y  = S_y  / A.unsqueeze(0)

    # ---- Variance of profiles: Var(P) ≈ Var(S)/A^2 with Var(S) ≈ S (Poisson) ----
    var_P = (S_y + tau_counts) / (A.unsqueeze(0) ** 2)                       # (B, nbins)
    w = 1.0 / (var_P + eps)

    # ---- Mask: only bins where true sum > 0 photons ----
    # per-sample mask: for this sample and this annulus, we have info only if S_y > 0
    active_mask = (S_y > 0.0)                                               # (B, nbins)
    w = w * active_mask.float()                                             # zero-out inactive bins

    # ---- Optimal N in profile space (weighted) ----
    num = (P_y * P_mu * w).sum(dim=1)                                       # Σ (y μ / Var) over active bins
    den = ((P_mu ** 2) * w).sum(dim=1).clamp_min(eps)                       # Σ (μ^2 / Var) over active bins
    N = num / den                                                           # (B,)

    # ---- Reduced χ² ----
    resid = N.unsqueeze(1) * P_mu - P_y
    chi2  = (resid.pow(2) * w).sum(dim=1)                                   # (B,)
    dof   = active_mask.sum(dim=1).sub(1).clamp_min(1)                      # per-sample DoF (only true>0 bins)
    chi2_red = chi2 / dof

    if return_N:
        return chi2_red, N
    return chi2_red