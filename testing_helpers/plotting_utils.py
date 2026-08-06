import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import matplotlib
import logging
from math import sqrt, ceil
import math
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


LABEL_FONT_SIZE = 30

# This helper function is needed by plot_comparison_grid
def _load_font(size):
    """Attempts to load a truetype font, falling back to the default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        return ImageFont.load_default()
    
def _tick_centers(ncol, nrow, img_size=48, pad=0):
    xs = [i * (img_size + pad) + img_size / 2 for i in range(ncol)]
    ys = [i * (img_size + pad) + img_size / 2 for i in range(nrow)]
    return xs, ys

def save_generated_grid(grid_list, ncol, out_dir, filename):
    """
    grid_list: list of (1,H,W) tensors in [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)
    grid_img = torchvision.utils.make_grid(grid_list, nrow=ncol, padding=0)
    fig, ax = plt.subplots(figsize=(ncol * 2, max(1, len(grid_list) // ncol) * 2))
    ax.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap="viridis", vmin=0, vmax=1)
    ax.set_axis_off()
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def save_side_by_side_comparison(
    gen_list, orig_list, ncol, nrow, x_raw, nH_raw, out_dir, filename,
    label_fontsize=14  # --- MODIFIED: Added font size parameter ---
):
    """
    Two-panel figure: generated vs originals, with tick labels reflecting raw params.
    """
    os.makedirs(out_dir, exist_ok=True)
    gen_grid = torchvision.utils.make_grid(gen_list, nrow=ncol, padding=0)
    orig_grid = torchvision.utils.make_grid(orig_list, nrow=ncol, padding=0)

    xs, ys = _tick_centers(ncol, nrow, img_size=48, pad=0)

    fig, (ax_gen, ax_orig) = plt.subplots(1, 2, figsize=(ncol * 3, nrow * 2.5))

    ax_gen.imshow(gen_grid.permute(1, 2, 0).squeeze(), cmap="viridis", vmin=0, vmax=1)
    ax_gen.set_title("Generated Samples", fontsize=16)
    ax_gen.set_xticks(xs); ax_gen.set_yticks(ys)
    # --- MODIFIED: Using the new parameter and added rotation ---
    ax_gen.set_xticklabels([str(v) for v in x_raw], fontsize=label_fontsize, rotation=45, ha='right')
    ax_gen.set_yticklabels([f"{v:.2e}" for v in nH_raw], fontsize=label_fontsize)
    ax_gen.set_xlabel("x (raw)"); ax_gen.set_ylabel("nH (raw)")

    ax_orig.imshow(orig_grid.permute(1, 2, 0).squeeze(), cmap="gray", vmin=0, vmax=1)
    ax_orig.set_title("Original Dataset", fontsize=16)
    ax_orig.set_xticks(xs); ax_orig.set_yticks(ys)
    # --- MODIFIED: Using the new parameter and added rotation ---
    ax_orig.set_xticklabels([str(v) for v in x_raw], fontsize=label_fontsize, rotation=45, ha='right')
    ax_orig.set_yticklabels([f"{v:.2e}" for v in nH_raw], fontsize=label_fontsize)
    ax_orig.set_xlabel("x (raw)"); ax_orig.set_ylabel("nH (raw)")

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def _heatmap(
    matrix, x_raw, nH_raw, title, cbar_label, cmap, out_dir, filename,
    vmin=None, vmax=None, origin="upper",
    label_fontsize=14 # --- MODIFIED: Added font size parameter ---
):
    os.makedirs(out_dir, exist_ok=True)
    nrow, ncol = matrix.shape
    fig, ax = plt.subplots(figsize=(ncol * 1.5, nrow * 1.2))
    im = ax.imshow(matrix, origin=origin, aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=vmin, vmax=vmax)
    
    avg_val = np.mean(matrix)
    title_str = f"{avg_val:.3f}" if 0 < abs(avg_val) < 10 else f"{avg_val:.3e}"
    ax.set_title(f"{title}: {title_str}", fontsize=16)

    # Axes ticks & labels
    ax.set_xticks(np.arange(len(x_raw)))
    ax.set_yticks(np.arange(len(nH_raw)))
    # --- MODIFIED: Using the new parameter and added rotation ---
    ax.set_xticklabels([str(v) for v in x_raw], fontsize=label_fontsize, rotation=45, ha='right')
    ax.set_yticklabels([f"{v:.2e}" for v in nH_raw], fontsize=label_fontsize)
    ax.set_xlabel("x (raw)", fontsize=14)
    ax.set_ylabel("nH (raw)", fontsize=14)

    norm = matplotlib.colors.Normalize(vmin=vmin if vmin is not None else np.nanmin(matrix),
                                     vmax=vmax if vmax is not None else np.nanmax(matrix))
    cmap_obj = plt.get_cmap(cmap)

    # Annotate each box
    for i in range(nrow):
        for j in range(ncol):
            val = matrix[i, j]
            text_str = f"{val:.3f}" if 0 < abs(val) < 10 else f"{val:.2f}"
            rgba = cmap_obj(norm(val))
            luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            text_color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, text_str, ha="center", va="center", color=text_color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=12, fontsize=14)

    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

# --- MODIFIED: Public functions now accept and pass down the font size parameter ---

def save_mse_heatmap(mse_matrix, x_raw, nH_raw, out_dir, filename, vmin=0.0, vmax=2e-2, label_fontsize=14):
    return _heatmap(mse_matrix, x_raw, nH_raw, "Average MSE", "MSE", cmap="viridis", 
                    out_dir=out_dir, filename=filename, vmin=vmin, vmax=vmax, origin="upper", label_fontsize=label_fontsize)

def save_sbp_heatmap(sbp_matrix, x_raw, nH_raw, out_dir, filename, vmin=0.0, vmax=2e-2, label_fontsize=14):
    return _heatmap(sbp_matrix, x_raw, nH_raw, "Average SBP Error", "SBP MSE", cmap="magma",
                    out_dir=out_dir, filename=filename, vmin=vmin, vmax=vmax, origin="upper", label_fontsize=label_fontsize)

def save_chi2_heatmap(chi2_matrix, x_raw, nH_raw, out_dir, filename, vmin=0.0, vmax=3, label_fontsize=14):
    return _heatmap(chi2_matrix, x_raw, nH_raw, "Average Polar-χ²", "Reduced χ²", cmap="magma",
                    out_dir=out_dir, filename=filename, vmin=vmin, vmax=vmax, origin="upper", label_fontsize=label_fontsize)

def save_percentage_heatmap(matrix, x_raw, nH_raw, out_dir, filename, vmin=0.0, vmax=100, label_fontsize=14):
    return _heatmap(matrix, x_raw, nH_raw, "Average % Difference", "% Difference", cmap="cividis",
                    out_dir=out_dir, filename=filename, vmin=vmin, vmax=vmax, origin="upper", label_fontsize=label_fontsize)


def save_generated_cells(gen_list, out_dir, prefix="cell", start_idx=0):
    """
    Save each generated sample as an individual PNG into <out_dir>/gen_grid/,
    auto-computing a near-square grid to label row/col in filenames.

    Args:
        gen_list (list[Tensor]): List of (1,H,W) or (C,H,W) tensors in [0,1].
        out_dir (str): Base output directory.
        prefix (str): Filename prefix (default: "cell").
        start_idx (int): Optional starting index for filenames.

    Returns:
        list[str]: Paths of the saved PNG files.
    """
    N = len(gen_list)
    if N == 0:
        return []

    # Choose a compact grid: ncol ≈ sqrt(N), nrow = ceil(N/ncol)
    ncol = int(ceil(sqrt(N)))
    nrow = int(ceil(N / ncol))

    cell_dir = os.path.join(out_dir, "gen_grid")
    os.makedirs(cell_dir, exist_ok=True)

    saved_paths = []
    for idx, img in enumerate(gen_list):
        # Ensure CHW with C in {1,3}
        if img.ndim == 2:
            img = img.unsqueeze(0)  # -> (1,H,W)
        elif img.ndim == 3 and img.size(0) not in (1, 3):
            img = img.mean(0, keepdim=True)  # collapse to grayscale

        r = idx // ncol
        c = idx % ncol

        fname = f"{prefix}_{(start_idx + idx):04d}_r{r:02d}_c{c:02d}.png"
        path = os.path.join(cell_dir, fname)
        torchvision.utils.save_image(torch.clamp(img, 0.0, 1.0), path)
        saved_paths.append(path)

    return saved_paths

def save_g_o_grids_side_by_side(
    g_list, o_list, out_dir, filename,
    nrow=None, pad=2, pad_value=0.5,
    cmap="gray", vmin=0.0, vmax=1.0, figsize=(12, 6)
):
    os.makedirs(out_dir, exist_ok=True)

    N = min(len(g_list), len(o_list))
    if N == 0:
        return None
    g_list, o_list = g_list[:N], o_list[:N]

    if nrow is None:
        nrow = max(1, int(math.ceil(math.sqrt(N))))

    def _prep(x):
        t = x.detach().float().cpu()
        if t.dim() == 2:
            t = t.unsqueeze(0)  # (1,H,W)
        return t.clamp(0, 1)

    g_tensor = torch.stack([_prep(x) for x in g_list], dim=0)  # (N,1,H,W)
    o_tensor = torch.stack([_prep(x) for x in o_list], dim=0)  # (N,1,H,W)

    grid_g = torchvision.utils.make_grid(g_tensor, nrow=nrow, padding=pad, pad_value=pad_value)
    grid_o = torchvision.utils.make_grid(o_tensor, nrow=nrow, padding=pad, pad_value=pad_value)

    H = max(grid_g.shape[1], grid_o.shape[1])
    def _pad_h(img, target_h):
        if img.shape[1] == target_h:
            return img
        diff = target_h - img.shape[1]
        return F.pad(img, (0, 0, 0, diff), value=pad_value)  # l,r,t,b

    grid_g = _pad_h(grid_g, H)
    grid_o = _pad_h(grid_o, H)

    spacer_w = pad * 4
    spacer = torch.full((grid_g.shape[0], H, spacer_w), pad_value)
    combined = torch.cat([grid_g, spacer, grid_o], dim=2)  # concat along width

    path = os.path.join(out_dir, filename)

    scale = 3
    dpi = 100 * scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(combined.permute(1, 2, 0).numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")

    # --- Hardcoded labels ---
    label_left  = "G — generated"
    label_right = "O — original"
    label_fontsize = 14
    label_bg_alpha = 0.55
    label_margin_px = 10

    left_x  = label_margin_px
    right_x = grid_g.shape[2] + spacer_w + label_margin_px
    top_y   = label_margin_px
    bbox = dict(facecolor="black", alpha=label_bg_alpha, boxstyle="round,pad=0.3")

    ax.text(left_x,  top_y, label_left,  ha="left", va="top", fontsize=label_fontsize,
            color="white", bbox=bbox)
    ax.text(right_x, top_y, label_right, ha="left", va="top", fontsize=label_fontsize,
            color="white", bbox=bbox)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Figure saved -> {os.path.abspath(path)}")
    return path


def save_photon_count_histogram(pg_, po_, tag, plots_dir):
    """
    Generates and saves a grouped bar chart comparing generated (pg) and
    original (po) photon counts for each annulus.

    Args:
        pg_ (torch.Tensor): A 1D tensor of generated photon counts per annulus.
        po_ (torch.Tensor): A 1D tensor of original photon counts per annulus.
        tag (str): A label for the plot, e.g., 'best', 'avg', 'worst'.
        plots_dir (str): The directory path where the plot image will be saved.
    """
    try:
        # Convert tensors to CPU numpy arrays and ensure they are 1D
        pg_counts = pg_.cpu().numpy().flatten()
        po_counts = po_.cpu().numpy().flatten()
        
        num_annuli = len(po_counts)
        if num_annuli == 0:
            logging.warning(f"No data to plot for '{tag}' case. Skipping bar chart.")
            return

        # Create x-axis positions for the bars
        indices = np.arange(num_annuli)
        bar_width = 0.35  # Width of each bar

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot the bars for 'Original' and 'Generated' counts side-by-side
        ax.bar(indices - bar_width/2, po_counts, bar_width, 
               label='Original (po)', color='skyblue', edgecolor='black')
        
        ax.bar(indices + bar_width/2, pg_counts, bar_width, 
               label='Generated (pg)', color='salmon', edgecolor='black')

        # --- Formatting the Plot ---
        ax.set_ylabel('Photon Count')
        ax.set_xlabel('Annulus Index')
        ax.set_title(f'Photon Count Comparison per Annulus ({tag.capitalize()} Case)')
        
        # Set the x-axis ticks to be in the middle of each group of bars
        ax.set_xticks(indices)
        ax.set_xticklabels(indices + 1)  # Label annuli as 1, 2, 3...
        
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        # Save the figure
        plot_path = os.path.join(plots_dir, f"{tag}_photon_counts_barchart.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"Successfully saved photon count bar chart to {plot_path}")

    except Exception as e:
        logging.error(f"Failed to create bar chart for '{tag}'. Error: {e}")


def plot_comparison_grid(
    g_target,
    g_target_x,
    g_target_nH,
    chi2_values,
    x_raw,
    nH_raw,
    grid_w,
    grid_h,
    output_path
):
    """
    Visualizes a pre-calculated list of chi-squared values in a grid.
    Left: generated image + its (x/1e4, nH).
    Right: grid of χ² values (target cell highlighted), and a Top-10 list of the lowest χ²
           with their corresponding (x/1e4, nH) from the grid.

    NOTE: All computations (sorting/matching) use original x_raw values.
          Only the *display* uses x_raw/1e4.
    """
    logging.info(f"Plotting single-g comparison grid to {output_path}...")
    chi2_matrix = np.array(chi2_values, dtype=float).reshape(grid_h, grid_w)

    # ---- scale for display only ----
    X_SCALE = 1e4
    x_disp = np.asarray(x_raw, dtype=float) / X_SCALE
    g_target_x_disp = float(g_target_x) / X_SCALE

    # --- Helper to convert image tensor to PIL ---
    def tensor_to_pil(tensor):
        img_tensor = tensor.detach().cpu().squeeze()
        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1) / 2
        img_tensor = img_tensor.clamp(0, 1)
        img_tensor_rgb = img_tensor.unsqueeze(0).repeat(3, 1, 1)
        return Image.fromarray(
            (img_tensor_rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )

    # --- Compute Top-K (lowest) χ² entries with their (r, c, x, nH) ---
    K = min(10, chi2_matrix.size)
    flat_order = np.argsort(chi2_matrix, axis=None)[:K]
    top_rows, top_cols = np.unravel_index(flat_order, chi2_matrix.shape)
    top_list = []
    for rank, (r, c) in enumerate(zip(top_rows, top_cols), start=1):
        val = float(chi2_matrix[r, c])
        x_val = float(x_raw[c])             # original (for logic)
        nH_val = float(nH_raw[r])
        x_val_disp = x_val / X_SCALE        # for display
        top_list.append({
            "rank": rank, "r": r, "c": c,
            "chi2": val, "x": x_val, "x_disp": x_val_disp, "nH": nH_val
        })

    # For quick membership checks when drawing the grid
    top_coords = {(e["r"], e["c"]): e["rank"] for e in top_list}

    # --- Setup canvas and drawing tools ---
    g_pil = tensor_to_pil(g_target)
    g_w, g_h = g_pil.size
    font         = _load_font(14)
    header_font  = _load_font(16)
    small_font   = _load_font(12)

    padding   = 20
    cell_w    = 100
    cell_h    = 40
    top_margin  = 60
    left_margin = 120
    panel_w     = 340  # side panel for Top-10 list

    grid_drawing_w = cell_w * grid_w
    grid_drawing_h = cell_h * grid_h

    # Canvas size now includes the right panel
    canvas_w = g_w + padding + left_margin + grid_drawing_w + padding + panel_w + padding
    canvas_h = max(g_h + 120, top_margin + grid_drawing_h + padding)
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)

    # --- Draw generated image and its info (x displayed divided by 1e4) ---
    canvas.paste(g_pil, (padding, top_margin))
    draw.text(
        (padding, top_margin + g_h + 10),
        f"Generated Image\nx: {g_target_x_disp:.4g}  (×10⁻⁴ of raw)\nnH: {g_target_nH:.2e}",
        fill='black',
        font=header_font
    )

    # --- Grid placement ---
    grid_start_x = padding + g_w + padding + left_margin
    grid_start_y = top_margin

    # Match uses ORIGINAL arrays (no scaling)
    match_row = next((i for i, val in enumerate(nH_raw) if math.isclose(val, g_target_nH)), -1)
    match_col = next((i for i, val in enumerate(x_raw)  if math.isclose(val, g_target_x)),  -1)

    # --- Draw the chi-squared grid ---
    for r in range(grid_h):
        for c in range(grid_w):
            x0 = grid_start_x + c * cell_w
            y0 = grid_start_y + r * cell_h

            # Highlight logic: target cell (yellow) > Top-10 (light green) > default
            if r == match_row and c == match_col:
                fill_color = "#FFEB99"   # yellow-ish
            elif (r, c) in top_coords:
                fill_color = "#CFF6D3"   # light green-ish
            else:
                fill_color = "#EEEEEE"

            # Cell rectangle
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline='black', fill=fill_color)

            # χ² text centered
            val_text = f"{chi2_matrix[r, c]:.3f}"
            text_bbox = draw.textbbox((0, 0), val_text, font=font)
            text_pos = (
                x0 + (cell_w - (text_bbox[2] - text_bbox[0])) / 2,
                y0 + (cell_h - (text_bbox[3] - text_bbox[1])) / 2
            )
            draw.text(text_pos, val_text, fill='black', font=font)

            # If in Top-10, add a small rank badge in the top-left corner
            if (r, c) in top_coords:
                rank = top_coords[(r, c)]
                badge_r = 12
                cx, cy = x0 + 12, y0 + 12
                draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r], fill="#2A7F62", outline="black")
                rank_text = str(rank)
                rb = draw.textbbox((0, 0), rank_text, font=small_font)
                draw.text(
                    (cx - (rb[2] - rb[0]) / 2, cy - (rb[3] - rb[1]) / 2),
                    rank_text,
                    fill='white',
                    font=small_font
                )

    # --- Headers around the grid ---
    draw.text(
        (grid_start_x + grid_drawing_w / 2, grid_start_y - 40),
        "x",
        fill='black',
        font=header_font,
        anchor="mm"
    )
    for c, x_val_disp in enumerate(x_disp):
        draw.text(
            (grid_start_x + (c + 0.5) * cell_w, grid_start_y - 15),
            f"{x_val_disp:.4g}",
            fill='black',
            font=font,
            anchor="mm"
        )
    draw.text(
        (grid_start_x - 80, grid_start_y + grid_drawing_h / 2),
        "nH",
        fill='black',
        font=header_font,
        anchor="mm"
    )
    for r, nH_val in enumerate(nH_raw):
        draw.text(
            (grid_start_x - 45, grid_start_y + (r + 0.5) * cell_h),
            f"{nH_val:.1e}",
            fill='black',
            font=font,
            anchor="mm"
        )

    # --- Right panel: Top-10 lowest χ² with (x/1e4, nH) ---
    panel_x = grid_start_x + grid_drawing_w + 20
    panel_y = top_margin
    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + grid_drawing_h],
        outline="#BBBBBB",
        fill="#F9F9F9"
    )
    draw.text((panel_x + 12, panel_y + 10), "Top 10 Lowest Chi2 (with x×10⁻⁴, nH)", fill='black', font=header_font)

    line_y = panel_y + 40
    line_step = 26
    for e in top_list:
        line = f"{e['rank']:>2}) Chi2={e['chi2']:.4g}   x={e['x_disp']:.4g}   nH={e['nH']:.3e}"
        draw.text((panel_x + 12, line_y), line, fill='black', font=font)
        line_y += line_step

    # --- Save the final image ---
    canvas.save(output_path)
    logging.info(f"Saved comparison grid with Top-10 list to {output_path}")