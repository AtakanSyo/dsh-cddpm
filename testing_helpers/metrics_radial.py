import math
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import math

# ---------- Core profiles ----------


def photon_counts_and_areas_of_annulus(img, center=None, nbins=12, dmax=None, dmin=None, logInfo = False, logStr = ""):
    """
    Surface-density radial profile in `nbins` annuli:
    (sum of pixel values) / (geometric annulus area)
    img: (B,1,H,W) → returns (B, nbins)
    """
    B, C, H, W = img.shape
    assert C == 1, "Only single-channel images are supported."

    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    y = torch.arange(H, device=img.device, dtype=torch.float32).view(H, 1)
    x = torch.arange(W, device=img.device, dtype=torch.float32).view(1, W)
    radius_map = torch.sqrt((x - cx).pow(2) + (y - cy).pow(2))

    corners = torch.tensor([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]], device=img.device, dtype=torch.float32)
    cxy = torch.tensor([cx, cy], device=img.device, dtype=torch.float32)
    max_r = torch.linalg.norm(corners - cxy, dim=1).max()

    bin_edges = torch.linspace(0.0, float(max_r), nbins + 1, device=img.device)

    r_in, r_out = bin_edges[:-1], bin_edges[1:]
    area_bins = math.pi * (r_out.pow(2) - r_in.pow(2))  # pixel-units
    # Convert bin areas to parsec^2
    # Convert pixel-units to pc^2 (or arcsec^2 depending on your scaling)
    pixel_arcsec = 43.2
    pixel_area_pc2 = pixel_arcsec**2
    # Includes annulus areas where are there no pixels.
    area_bins_pc2 = area_bins * pixel_area_pc2

    flat_radius = radius_map.view(-1)
    bin_idx = torch.bucketize(flat_radius, bin_edges) - 1
    bin_idx = bin_idx.clamp(0, nbins - 1)

    # Doesn't include annulus areas where there are no pixels.
    #pixels_per_bin = torch.bincount(bin_idx.to(torch.int64), minlength=nbins).to(dtype=torch.float32)
    #area_bins_pc2 = pixels_per_bin * pixel_area_pc2

    bin_sums = torch.zeros((B, nbins), device=img.device)
    flat_pixels = (((img)* (dmax - dmin))+dmin).view(B, -1)
    flat_pixels = torch.round(flat_pixels)
    for b in range(B):
        bin_sums[b].index_add_(0, bin_idx, flat_pixels[b])
    
    if logInfo:
        for b in range(B):
            # Logging photons and areas
            for k in range(nbins):
                photons = bin_sums[b, k].item()
                area    = area_bins_pc2[k].item()
                logging.info(f"Image {b}, Bin {k+1}: Photons = {photons:.0f}, Area = {area:.3f} (pc^2)")

    if logInfo:
        logging.info(f"{bin_sums}")
        logging.info(f"{area_bins_pc2}")
    return bin_sums, area_bins_pc2 # return photon counts per bin and area of each bin


# def reduced_chi_square(RefleX_counts_per_ring, AI_counts_per_ring, annulus_areas=None ,dof=None):

#     mask = (RefleX_counts_per_ring != 0) & (AI_counts_per_ring != 0)

#     # log masked bins
#     total_bins = mask.numel()
#     kept_bins  = mask.sum().item()
#     masked_bins_cnt = total_bins - kept_bins

#     # Apply the mask to both tensors
#     reflex_cnts = RefleX_counts_per_ring[mask]  # avoid div 0
#     ai_cnts = AI_counts_per_ring[mask]
#     # Expand annulus_areas (nbins,) → (B, nbins)
#     areas_expanded = annulus_areas.unsqueeze(0).expand_as(RefleX_counts_per_ring)
#     # Now mask works
#     #logging.info(f"Dimension of RefleX_counts_per_ring {RefleX_counts_per_ring.shape}")
#     annulus_areas_masked = areas_expanded[mask]

#     prof_AI = ai_cnts / (annulus_areas_masked)
#     prof_RefleX = reflex_cnts / (annulus_areas_masked)

#     w   = (annulus_areas_masked)**2 / torch.clamp(reflex_cnts, min=1)  # standard deviation squared.
#     #logging.info(f"Reflex counts squared: {reflex_cnts**2}")
#     N_best = torch.sum(w * prof_AI * prof_RefleX) / (torch.sum(w * prof_AI * prof_AI))
#     res   = prof_RefleX - N_best * prof_AI # reflex_cnts - N * AI_cnts
#     chi2  = torch.sum((w * (res**2) )) # w*(reflex_cnts - N * AI_cnts)^2
#     if dof is None:
#         dof = max(reflex_cnts.numel() - 1, 1) # number of rings - 1 is dof.
#     return chi2 / dof, N_best, dof

def reduced_chi_square(RefleX_counts_per_ring, AI_counts_per_ring, annulus_areas=None, dof=None):
    """
    Calculates the reduced chi-squared value without masking any bins.
    """
    # Expand annulus_areas to match the shape of the count tensors
    areas_expanded = annulus_areas.unsqueeze(0).expand_as(RefleX_counts_per_ring)

    # Calculate surface brightness profiles for all bins
    prof_AI = AI_counts_per_ring / areas_expanded
    prof_RefleX = RefleX_counts_per_ring / areas_expanded

    # Calculate weights, clamping the reference counts to avoid division by zero.
    # This is the main safeguard now that the mask is removed.
    w = (areas_expanded)**2 / torch.clamp(RefleX_counts_per_ring, min=1)

    # Calculate the optimal scaling factor N_best
    # A small epsilon is added to the denominator for numerical stability.
    epsilon = 1e-9
    numerator = torch.sum(w * prof_AI * prof_RefleX)
    denominator = torch.sum(w * prof_AI * prof_AI)
    N_best = numerator / (denominator + epsilon)

    # Calculate the residual and the final chi-squared value
    res = prof_RefleX - prof_AI
    chi2 = torch.sum(w * (res**2))

    # Degrees of freedom is the number of bins minus 1 (for the fitted N_best parameter)
    if dof is None:
        num_bins = RefleX_counts_per_ring.shape[-1]
        dof = max(num_bins - 1, 1)

    return chi2 / dof, N_best, dof

def sbp_error(RefleX_counts_per_ring, AI_counts_per_ring, annulus_areas=None, dof=None):
    """
    Calculates the reduced chi-squared value without masking any bins.
    """
    # Expand annulus_areas to match the shape of the count tensors
    areas_expanded = annulus_areas.unsqueeze(0).expand_as(RefleX_counts_per_ring)

    # Calculate surface brightness profiles for all bins
    prof_AI = AI_counts_per_ring / areas_expanded
    prof_RefleX = RefleX_counts_per_ring / areas_expanded

    # Calculate weights, clamping the reference counts to avoid division by zero.
    # This is the main safeguard now that the mask is removed
    w = (areas_expanded)**2
    #w = (areas_expanded)**2 / torch.clamp(RefleX_counts_per_ring, min=1)

    # Calculate the optimal scaling factor N_best
    # A small epsilon is added to the denominator for numerical stability.
    epsilon = 1e-9
    numerator = torch.sum(w * prof_AI * prof_RefleX)
    denominator = torch.sum(w * prof_AI * prof_AI)
    N_best = numerator / (denominator + epsilon)

    # Calculate the residual and the final chi-squared value
    res = prof_RefleX - prof_AI
    sbp = torch.sum(w * (res**2))

    # Degrees of freedom is the number of bins minus 1 (for the fitted N_best parameter)
    if dof is None:
        num_bins = RefleX_counts_per_ring.shape[-1]
        dof = max(num_bins - 1, 1)

    return sbp / dof, N_best, dof

# ---------- Optional: profile plotter used by testing.py ----------
def radial_profile_mse(true_counts, pred_counts, annulus_areas):
    """
    Calculates the Mean Squared Error (MSE) between two radial profiles.

    Args:
        true_counts (torch.Tensor): Photon counts of the true/original image per ring.
        pred_counts (torch.Tensor): Photon counts of the predicted/AI image per ring.
        annulus_areas (torch.Tensor): Area of each annulus (ring).

    Returns:
        torch.Tensor: A single scalar value for the MSE.
    """
    # Expand annulus_areas to match the shape of the count tensors
    areas_expanded = annulus_areas.unsqueeze(0).expand_as(true_counts)
    
    # Avoid division by zero if an area is zero
    epsilon = 1e-9

    # Calculate the radial profiles (surface brightness)
    rp_true = true_counts / (areas_expanded + epsilon)
    rp_pred = pred_counts / (areas_expanded + epsilon)

    # Calculate the Mean Squared Error between the two profiles
    mse = torch.mean((rp_pred - rp_true) ** 2)

    return mse

def compute_radial_profile_with_plot(
    pg: torch.Tensor,
    po: torch.Tensor,
    error_tensor: torch.Tensor,      # single set of errors for original
    areas: torch.Tensor,
    plot_dir: str,
    img_file: str,
    plot: bool = True,
    pixel_scale: float = 122.1880517,
    k = "",
    chi2 = None,
    total_photons = None,
    nbins=None,  # optional, unused
    normalization_factor = 1
):
    """
    Plot generated vs original photon counts per radial bin,
    with vertical error bars for the original profile.
    """

    #logging.info(f"Original image counts {po}")
    def _to_1d(t: torch.Tensor, name="tensor") -> torch.Tensor:
        t = t.detach().cpu().float().squeeze()
        if t.ndim != 1:
            raise ValueError(f"{name} should be 1D after squeeze, got shape {tuple(t.shape)}")
        return t

    # Convert inputs to 1D
    pg_1d = _to_1d(pg, "pg")
    pg_1d = pg_1d * normalization_factor #  CHANGE THIS TO N!!!!!!
    po_1d = _to_1d(po, "po")
    areas_1d = _to_1d(areas, "areas")
    err_1d = _to_1d(error_tensor, "error_tensor")  # no indexing, always original's errors

    # CONVERT TO SURFACE BRIGHTNESS , photon count / pc^2
    pg_1d = pg_1d / areas_1d
    po_1d = po_1d / areas_1d
    err_1d = err_1d / areas_1d

    # Safety check
    if not (pg_1d.shape[0] == po_1d.shape[0] == areas_1d.shape[0] == err_1d.shape[0]):
        raise ValueError("pg, po, areas, and error lengths must match.")

    nbins_val = pg_1d.shape[0]
    radius_arcsec = (np.arange(nbins_val) + 0.5) * pixel_scale  # bin centers


    if not plot:
        return torch.stack([pg_1d, po_1d], dim=0)

    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, img_file)

    mask_gen = pg_1d.numpy() > 0
    mask_org = po_1d.numpy() > 0

    fig, ax = plt.subplots(figsize=(6, 4))

    # Solid (generated) and dashed (original) in black
    ax.plot(radius_arcsec[mask_gen], pg_1d.numpy()[mask_gen],
            linestyle='--', linewidth=1.0, color='black', label="Diffusion Model")
    ax.plot(radius_arcsec[mask_org], po_1d.numpy()[mask_org],
            linestyle='-', linewidth=1.0, color='red', label="RefleX")

    # Error bars
    ax.errorbar(radius_arcsec[mask_org], po_1d.numpy()[mask_org],
                yerr=err_1d.numpy()[mask_org], fmt='none',
                ecolor='red', elinewidth=0.8, capsize=3)

    # Axes look & scales
    ax.set_yscale('log')
    ax.set_ylim(10**(-3.7), 10**(-1.2))
    ax.set_xlim(0, float(radius_arcsec.max()))
    ax.set_xlabel("Radial Distance (arcsec)")
    ax.set_ylabel("Surface Brightness (photon count / pc²)")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    ax.set_xlim(0, 1000)
    ax.set_title(f"$\\chi^2$ = {chi2:.2f}, N = {normalization_factor}", fontsize=12, pad=10)

    # Box on all sides, ticks inward on all sides (major+minor)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   length=4, width=0.8)
    ax.minorticks_on()
    ax.tick_params(which='minor', length=2, width=0.6)

    # No grid / legend for the clean IDL look
    # ax.legend().remove()  # (only if you had a legend)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    logging.info(f"Saved figure to {save_path}")
    plt.close(fig)

    return torch.stack([pg_1d, po_1d], dim=0)


def compute_polar_profile(img, n_rad=24, n_ang=12, center=None):
    """
    img: (B, 1, H, W) grayscale
    returns: (B, n_rad, n_ang) avg brightness per (radial, angular) sector
    """
    B, C, H, W = img.shape
    assert C == 1, "Expected single-channel images"

    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    y = torch.arange(H, device=img.device).view(H, 1).float() - cy
    x = torch.arange(W, device=img.device).view(1, W).float() - cx
    dx = x.expand(H, W).reshape(-1)
    dy = y.expand(H, W).reshape(-1)

    radius = torch.sqrt(dx * dx + dy * dy)  # (H*W,)
    angle = (torch.atan2(dy, dx) + 2 * torch.pi) % (2 * torch.pi)

    r_edges = torch.linspace(0.0, radius.max(), n_rad + 1, device=img.device)
    a_edges = torch.linspace(0.0, 2 * torch.pi, n_ang + 1, device=img.device)

    r_idx = torch.bucketize(radius, r_edges) - 1
    a_idx = torch.bucketize(angle, a_edges) - 1
    r_idx = r_idx.clamp(0, n_rad - 1)
    a_idx = a_idx.clamp(0, n_ang - 1)

    joint_idx = r_idx * n_ang + a_idx
    n_bins = n_rad * n_ang
    one_hot = F.one_hot(joint_idx, num_classes=n_bins).float()  # (H*W, n_bins)

    bin_counts = one_hot.sum(dim=0).clamp(min=1e-8)  # (n_bins,)
    flat = img.view(B, -1)                           # (B, H*W)
    bin_sums = flat @ one_hot                        # (B, n_bins)

    prof = (bin_sums / bin_counts).view(B, n_rad, n_ang)
    return prof


def compute_radial_profile_single(img, center=None, nbins=24):
    """
    Computes the average brightness per annulus for a single image.

    Args:
        img (torch.Tensor): A single image tensor of shape (H, W) or (1, H, W).
        center (tuple, optional): (cx, cy) of the profile center. 
                                  Defaults to the image center.
        nbins (int, optional): The number of radial bins. Defaults to 24.

    Returns:
        torch.Tensor: A 1D tensor of shape (nbins,) containing the average 
                      brightness for each annulus. Returns NaN for empty bins.
    """
    # --- 1. Input Validation and Setup ---
    if img.ndim == 3:
        # Squeeze the channel dimension if shape is (1, H, W)
        if img.shape[0] == 1:
            img = img.squeeze(0)
        else:
            raise ValueError(f"Input image must be 2D (H, W) or single-channel 3D (1, H, W), but got shape {img.shape}")
    elif img.ndim != 2:
        raise ValueError(f"Input image must be 2D (H, W), but got shape {img.shape}")

    H, W = img.shape

    if center is None:
        # Default to the geometric center of the image
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    # --- 2. Create a Radius Map ---
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing='ij'
    )
    # Calculate distance of each pixel from the center
    radius_map = torch.sqrt((x - cx)**2 + (y - cy)**2)

    # --- 3. Bin Pixels by Radius ---
    # Define the edges of the radial bins
    bin_edges = torch.linspace(0, radius_map.max(), nbins + 1, device=img.device)
    
    # Flatten both the image and the radius map for easier processing
    flat_img = img.flatten()
    flat_radius = radius_map.flatten()

    # Determine which bin each pixel falls into
    # `bucketize` finds the index of the bin for each radius value
    bin_idx = torch.bucketize(flat_radius, bin_edges, right=True) - 1
    # Ensure indices are within the valid range [0, nbins-1]
    bin_idx = bin_idx.clamp(0, nbins - 1)

    # --- 4. Calculate Profile ---
    # Create tensors to hold the sum of pixel values and the count of pixels in each bin
    bin_sums = torch.zeros(nbins, device=img.device)
    bin_counts = torch.zeros(nbins, device=img.device)

    # `index_add_` is a highly efficient way to sum values based on an index tensor.
    # It adds the values from `flat_img` to `bin_sums` at the indices specified by `bin_idx`.
    bin_sums.index_add_(0, bin_idx, flat_img)
    # We do the same with a tensor of ones to get the count of pixels in each bin.
    bin_counts.index_add_(0, bin_idx, torch.ones_like(flat_img))

    # Calculate the average brightness by dividing sums by counts.
    # Clamp counts to 1 to avoid division by zero for empty bins.
    # The result for an empty bin will be 0 / 1 = 0.
    # If you prefer NaN for empty bins, use: bin_counts[bin_counts == 0] = float('nan')
    average_profile = bin_sums / bin_counts.clamp(min=1)

    return average_profile

def compute_radial_profile(img, center=None, nbins=24):
    """
    Average brightness per annulus (not divided by geometric area).
    img: (B,1,H,W) → (B, nbins)
    """
    B, C, H, W = img.shape
    assert C == 1, "Single-channel only."

    if center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = center

    y = torch.arange(H, device=img.device).view(H, 1).float()
    x = torch.arange(W, device=img.device).view(1, W).float()
    radius_map = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    bin_edges = torch.linspace(0, radius_map.max(), nbins + 1, device=img.device)
    flat_img = img.view(B, -1)
    flat_radius = radius_map.view(-1)
    bin_idx = torch.bucketize(flat_radius, bin_edges) - 1
    bin_idx = bin_idx.clamp(0, nbins - 1)

    bin_sums = torch.zeros(B, nbins, device=img.device)
    bin_counts = torch.zeros(B, nbins, device=img.device)
    for b in range(B):
        vals = flat_img[b]
        bin_sums[b].index_add_(0, bin_idx, vals)
        ones = torch.ones_like(vals)
        bin_counts[b].index_add_(0, bin_idx, ones)

    return bin_sums / bin_counts


# ---------- Loss & stats ----------

def radial_profile_loss_by_area(pred_img, true_img, nbins=24, center=None, dmax = None, dmin = None):
    """MSE between area-normalized radial profiles."""
    rp_pred, areas = photon_counts_and_areas_of_annulus(pred_img, center=center, nbins=nbins, dmax = dmax, dmin = dmin)
    rp_true, areas = photon_counts_and_areas_of_annulus(true_img, center=center, nbins=nbins, dmax = dmax, dmin = dmin)
    return torch.mean((rp_pred - rp_true) ** 2)


def radial_profile_loss(pred_img, true_img, nbins=24, center=None):
    """MSE between average radial profiles (not area-normalized)."""
    rp_pred = compute_radial_profile(pred_img, center=center, nbins=nbins)
    rp_true = compute_radial_profile(true_img, center=center, nbins=nbins)
    return torch.mean((rp_pred - rp_true) ** 2)

def radial_profile_loss_single(pred_img, true_img, nbins=24, center=None):
    """MSE between average radial profiles (not area-normalized)."""
    rp_pred = compute_radial_profile_single(pred_img, center=center, nbins=nbins)
    rp_true = compute_radial_profile_single(true_img, center=center, nbins=nbins)
    return torch.mean((rp_pred - rp_true) ** 2)



def annulus_mape(g: torch.Tensor, o: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error (MAPE) across annuli.
    Denominator = prediction (o).

    Args:
        g (torch.Tensor): ground-truth per-annulus values (1D or 2D).
        o (torch.Tensor): predicted per-annulus values (same shape as g).
        eps (float): small constant to avoid division by zero.

    Returns:
        float: average percentage error across annuli.
    """
    ape = (g - o).abs() / (o.abs() + eps) * 100.0
    return float(ape.mean().item())


def bray_curtis_percent_noiseaware(
    g: torch.Tensor,
    o: torch.Tensor,
    sigma: float = 0.01,   # <-- hard-coded noise floor
    eps: float = 1e-12,
    reduction: str = "mean"
):
    """
    Noise-aware Bray–Curtis dissimilarity in percent.

    BC% = 100 * sum|g - o| / ( sum|g| + sum|o| + beta + eps )

    where beta = K * sigma * sqrt(2/pi), K = number of annuli.

    Args:
        g, o: tensors of shape (..., K), same shape.
        sigma: scalar noise std used as floor (hard-coded for now).
        eps: small constant to avoid division by zero.
        reduction: "mean" | "none" | "sum"

    Returns:
        float if reduction != "none", else Tensor of shape (...,).
    """
    if g.shape != o.shape:
        raise ValueError(f"Shapes must match, got {g.shape} vs {o.shape}")
    g = g.float(); o = o.float()

    K = g.shape[-1]
    num = (g - o).abs().sum(dim=-1)                         # (...,)
    den_raw = g.abs().sum(dim=-1) + o.abs().sum(dim=-1)     # (...,)

    # noise floor
    beta = K * sigma * math.sqrt(2.0 / math.pi)

    den = den_raw + beta + eps
    bc_percent = (num / den) * 100.0

    if reduction == "none":
        return bc_percent
    elif reduction == "mean":
        return float(bc_percent.mean().item())
    elif reduction == "sum":
        return float(bc_percent.sum().item())
    else:
        raise ValueError(f"Invalid reduction: {reduction!r}")
    

def percentage_error(g: torch.Tensor, o: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Normalized RMSE percentage difference between two images (0–100%).
    Handles cases where the dynamic range of the output is zero.
    """
    diff = (g - o).float()
    rmse = torch.sqrt(torch.mean(diff ** 2))

    # If RMSE is negligible, the images are identical.
    if rmse.item() < eps:
        return 0.0

    # Start with the output's dynamic range as the normalizer.
    normalizer = o.max() - o.min()

    # If the output's dynamic range is zero (it's a constant image),
    # fall back to the ground truth's dynamic range.
    if normalizer.item() < eps:
        normalizer = g.max() - g.min()

        # If the ground truth's dynamic range is ALSO zero, it means we have
        # two different constant images. The error is effectively 100%.
        if normalizer.item() < eps:
            return 100.0

    # Calculate the normalized error and convert to percentage.
    val = rmse / (normalizer + eps)
    
    # You might still want to clamp here if you never want values > 100
    # return min(val.item() * 100.0, 100.0)

    return float(val.item() * 100.0)

# -------------------------
# 1) Counts & Poisson errors (IDL: profc[0,*], profc[1,*])
# -------------------------
# def compute_annular_counts(img, center=None, nbins=24, k=1.0):
#     """
#     IDL-equivalent annular *counts* (NOT area-normalized).
#     Returns:
#       counts: (B, nbins)          == profc[0,*]
#       sigma:  (B, nbins)          == profc[1,*] = sqrt(counts)
#       edges:  (nbins+1,) bin edges (pixels)
#     Notes:
#       - If you quantized / scaled in IDL, reflect it with 'k' (e.g., img*255*k like your code).
#       - Works on (B,1,H,W) single-channel tensors.
#     """
#     B, C, H, W = img.shape
#     assert C == 1, "Single-channel images only."

#     # Choose center
#     if center is None:
#         cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
#     else:
#         cx, cy = center

#     # Radius map in pixels
#     y = torch.arange(H, device=img.device, dtype=torch.float32).view(H, 1)
#     x = torch.arange(W, device=img.device, dtype=torch.float32).view(1, W)
#     radius_map = torch.sqrt((x - cx).pow(2) + (y - cy).pow(2))

#     # Uniform annular bins in pixel radius
#     r_max = float(radius_map.max().item())
#     bin_edges = torch.linspace(0.0, r_max, nbins + 1, device=img.device)

#     # Flatten once
#     flat_radius = radius_map.view(-1)
#     bin_idx = torch.bucketize(flat_radius, bin_edges) - 1
#     bin_idx = bin_idx.clamp(0, nbins - 1)  # safety

#     flat_pixels = (img * 255.0 * k).view(B, -1)  # mirror your scaling
#     counts = torch.zeros((B, nbins), device=img.device)

#     for b in range(B):
#         counts[b].index_add_(0, bin_idx, flat_pixels[b])

#     # Poisson sigma = sqrt(counts). Avoid zero-division downstream by clamping minimally.
#     sigma = torch.sqrt(counts).clamp_min_(0.0)

#     return counts, sigma, bin_edges


# # -------------------------
# # 2) AI radial profile (no errors)  (IDL: profai)
# # -------------------------
# def compute_ai_profile_counts(ai_img, center=None, nbins=24, k=1.0):
#     """
#     Same annular *counts* for the AI image, but WITHOUT errors (profai).
#     Returns:
#       profai: (B, nbins)
#       edges:  (nbins+1,)
#     """
#     profai, _sigma_dummy, edges = compute_annular_counts(ai_img, center=center, nbins=nbins, k=k)
#     return profai, edges
 


# #3) Fit N and reduced chi^2 (IDL spec)

# def fit_normalization_and_chi2(profc_counts, profc_sigma, profai_counts, eps=1e-12):
#     """
#     Solve for N in profc = N * profai via chi^2 minimization
#       N = sum( m_i d_i / sigma_i^2 ) / sum( m_i^2 / sigma_i^2 )
#     and compute reduced chi^2:
#       chi2_red = (1/(nbins-1)) * sum( (d_i - N m_i)^2 / sigma_i^2 )

#     Inputs: (B, nbins) tensors
#       profc_counts  ≡ d_i
#       profc_sigma   ≡ sigma_i = sqrt(d_i)
#       profai_counts ≡ m_i
#     Returns:
#       N:         (B,) best-fit normalization for each item in batch
#       chi2_red:  (B,) reduced chi-square
#     """
#     assert profc_counts.shape == profai_counts.shape == profc_sigma.shape, "Shape mismatch."

#     valid = profc_counts > 0  # only use bins with positive counts
#     w = torch.zeros_like(profc_counts)
#     w[valid] = 1.0 / profc_counts[valid]  # σ² = counts


#     num = torch.sum(w * profai_counts * profc_counts, dim=1)
#     den = torch.sum(w * (profai_counts ** 2), dim=1).clamp_min(1e-12)
#     N = num / den

#     resid = profc_counts - N.unsqueeze(1) * profai_counts
#     chi2 = torch.sum(w * resid * resid, dim=1)
#     #nbins = profc_counts.shape[1]
#     dof = valid.sum(dim=1) - 1  # only count valid bins in DOF
#     chi2_red = chi2 / dof.clamp_min(1)

#     # Optional diagnostics
#     logging.info(f"[fit] N (mean±std): {N.mean().item():.6g} ± {N.std().item():.6g}")
#     logging.info(f"[fit] reduced chi2 (mean±std): {chi2_red.mean().item():.6g} ± {chi2_red.std().item():.6g}")

#     return N, chi2_red

# def fit_normalization_and_chi2(profc_counts, profc_sigma, profai_counts, eps=1e-12):
#     assert profc_counts.shape == profai_counts.shape == profc_sigma.shape

#     valid = (profc_counts > 0) & torch.isfinite(profc_sigma) & (profc_sigma > 0)
#     w = torch.zeros_like(profc_counts)
#     w[valid] = 1.0 / (profc_sigma[valid] ** 2)

#     num = torch.sum(w * profai_counts * profc_counts, dim=1)
#     den = torch.sum(w * (profai_counts ** 2), dim=1)

#     enough_bins = valid.sum(dim=1) > 1
#     N = torch.full_like(num, torch.nan)
#     N[enough_bins & (den > eps)] = num[enough_bins & (den > eps)] / den[enough_bins & (den > eps)]

#     resid = profc_counts - N.unsqueeze(1) * profai_counts
#     chi2 = torch.sum(w * resid * resid, dim=1)
#     dof = valid.sum(dim=1) - 1

#     chi2_red = torch.full_like(chi2, torch.nan)
#     chi2_red[enough_bins & (dof > 0)] = chi2[enough_bins & (dof > 0)] / dof[enough_bins & (dof > 0)]
#      # Optional diagnostics
#     logging.info(f"[fit] N (mean±std): {N.mean().item():.6g} ± {N.std().item():.6g}")
#     logging.info(f"[fit] reduced chi2 (mean±std): {chi2_red.mean().item():.6g} ± {chi2_red.std().item():.6g}")

#     return N, chi2_red

# def fit_normalization_and_chi2(profc_counts, profc_sigma, profai_counts, eps=1e-12, sigma_floor=0.0):
#     """
#     profc_counts  (B, nbins): observed ring counts (photons)
#     profc_sigma   (B, nbins): sqrt(counts) (photons). If you pass something else, set sigma_floor>0.
#     profai_counts (B, nbins): model ring counts (photons)
#     """
#     assert profc_counts.shape == profai_counts.shape == profc_sigma.shape

#     # Valid where observed has data & finite sigma
#     valid = (profc_counts > 0) & torch.isfinite(profc_sigma) & (profc_sigma > 0)

#     # Optional floor to avoid huge weights for tiny counts (keep 0.0 for pure Poisson)
#     if sigma_floor > 0:
#         profc_sigma = torch.clamp(profc_sigma, min=sigma_floor)

#     # Weights: w_i = 1 / sigma_i^2 (on valid bins only)
#     w = torch.zeros_like(profc_counts)
#     w[valid] = 1.0 / (profc_sigma[valid] ** 2)

#     # Sufficient bins to fit 1 parameter?
#     n_valid = valid.sum(dim=1)
#     enough_bins = n_valid > 1

#     # Compute N safely
#     num = torch.sum(w * profai_counts * profc_counts, dim=1)
#     den = torch.sum(w * (profai_counts ** 2),          dim=1)

#     N = torch.zeros_like(num)  # default to zero-model if den≈0 or not enough bins
#     good_fit = enough_bins & (den > eps)
#     N[good_fit] = num[good_fit] / den[good_fit]
#     N[~good_fit] = 0.0  # zero-model fallback, avoids NaNs

#     # Residuals & chi2 on valid bins only
#     resid = profc_counts - N.unsqueeze(1) * profai_counts
#     resid = torch.where(valid, resid, torch.zeros_like(resid))

#     chi2 = torch.sum(w * resid * resid, dim=1)
#     dof = (n_valid - 1).clamp_min(1)
#     chi2_red = chi2 / dof

#     # Log only finite stats; don't let a bad row poison the mean/std
#     finiteN = torch.isfinite(N)
#     finiteC = torch.isfinite(chi2_red)
#     N_mean = N[finiteN].mean().item() if finiteN.any() else float("nan")
#     N_std  = N[finiteN].std(unbiased=False).item() if finiteN.sum() > 1 else (0.0 if finiteN.any() else float("nan"))
#     C_mean = chi2_red[finiteC].mean().item() if finiteC.any() else float("nan")
#     C_std  = chi2_red[finiteC].std(unbiased=False).item() if finiteC.sum() > 1 else (0.0 if finiteC.any() else float("nan"))

#     if chi2_red > 50:
#         logging.info(f"[fit] N (mean±std): {N_mean:.6g} ± {N_std:.6g}")
#         logging.info(f"[fit] reduced chi2 (mean±std): {C_mean:.6g} ± {C_std:.6g}")

#         logging.info(f"sum counts (Chandra) = {profc_counts.sum(dim=1)[:5]}")
#         logging.info(f"sum counts (AI)  {profai_counts.sum(dim=1)[:5]}")
#         logging.info(f"n_valid bins = {(profc_counts > 0).sum(dim=1)[:5]}")
#         logging.info(f"any fractional < 1e-3 counts? {(profc_counts[(profc_counts>0) & (profc_counts<1e-3)]).numel() > 0}")
        
#     return N, chi2_red
# # -------------------------
# # 4) Convenience wrapper for one-shot workflow
# # -------------------------
# def chandra_vs_ai_fit(chandra_img, ai_img, nbins=24, center=None, k=1.0):
    """
    Full pipeline:
      - profc[0,*] & profc[1,*] from Chandra
      - profai from AI
      - Fit N and reduced chi^2
    Shapes: imgs are (B,1,H,W)
    Returns:
      profc_counts, profc_sigma, profai_counts, N, chi2_red
    """
    # profc_counts, profc_sigma, _edges = compute_annular_counts(chandra_img, center=center, nbins=nbins, k=k)
    # profai_counts, _ = compute_ai_profile_counts(ai_img, center=center, nbins=nbins, k=k)
    # N, chi2_red = fit_normalization_and_chi2(profc_counts, profc_sigma, profai_counts)
    # return profc_counts, profc_sigma, profai_counts, N, chi2_red