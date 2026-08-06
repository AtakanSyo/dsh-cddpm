
""" OLD BASELINE CHI2 FUNC NO V PARAM 
def radial_reduced_chi2_area2_bin_filtered(
    data,
    model,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 43.2,
    eps: float = 0.0,
    bin_filter: float = 10.0,
    return_bin_errors: bool = False,
    return_profiles: bool = False,
    return_chi2_contributions: bool = False,
    return_N_best: bool = False,
    plot_bins: bool = False,
    Fano_Data: float = 1,
    Fano_Model: float = 1,
):
    # --- 1. Setup Data and Geometry ---
    data = np.nan_to_num(np.squeeze(np.asarray(data, dtype=np.float64)))
    model = np.nan_to_num(np.squeeze(np.asarray(model, dtype=np.float64)))
    
    H, W = data.shape

    # print(f"DEBUG Function B Input Shape: {data.shape}")
    # print(f"DEBUG Function B Data Sum:    {np.sum(data):.4f}")
    cy, cx = (data.shape[0]-1)/2, (data.shape[1]-1)/2
    # print(f"DEBUG Function B Center Val:  {data[int(cy), int(cx)]:.4f}")

    # --- 2. Radial Bins ---
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    
    # Define edges to capture everything (corner to corner)
    edges = np.linspace(0.0, rr.max(), n_annuli + 1)
    bins = np.digitize(rr.ravel(), edges) - 1
    bins = np.clip(bins, 0, n_annuli - 1)

    # --- 3. Diagnostic Plotting (Inserted Here) ---
    if plot_bins:
        # Reshape the 1D bins back to 2D image shape for visualization
        bin_map = bins.reshape(H, W)

        # --- Create alternating B&W map ---
        # Calculate bin parity (0 for even, 1 for odd)
        bin_map_bw = bin_map % 2
        # Define a custom 2-color colormap
        bw_cmap = ListedColormap(['white', 'black'])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the heatmap using the parity map and custom colormap
        im = ax.imshow(bin_map_bw, origin='lower', cmap=bw_cmap, interpolation='nearest')
        
        # Create a custom colorbar with just two ticks for Black and White
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Even Bins (Black)', 'Odd Bins (White)'])
        
        # Overlay the calculated annular edges
        for i, radius in enumerate(edges):
            # Only draw if radius is within reasonable bounds of the image
            # Use a color that contrasts with both black and white, e.g., cyan or magenta
            circle = patches.Circle((cx, cy), radius, fill=False, edgecolor='cyan', 
                                    linestyle='--', alpha=0.7, linewidth=1.0)
            ax.add_patch(circle)
            
            # Optional: Label edges
            if i % 5 == 0 and i < len(edges) - 1:
                ax.text(cx + radius, cy, str(i), color='cyan', fontsize=9, weight='bold')

        ax.scatter([cx], [cy], color='red', marker='x', s=100, label='Center')
        ax.set_title(f"Diagnostic: Alternating Bin Assignment (N={n_annuli})")
        ax.legend()
        plt.show()

    # --- 3. Summation ---
    S_data = np.bincount(bins, weights=data.ravel(), minlength=n_annuli)
    S_model = np.bincount(bins, weights=model.ravel(), minlength=n_annuli)
    N_pixels = np.bincount(bins, minlength=n_annuli).astype(np.float64)

    # --- 4. Surface Brightness ---
    area_per_pixel = pixel_scale_arcsec ** 2
    area_bin = N_pixels * area_per_pixel

    # Inside Function B, right at the top:
    # print(f"DEBUG Function B Input Shape: {data.shape}")
    
    # Identify Valid Bins
    valid_bins = (S_data > bin_filter)
    
    mu_data = np.zeros_like(S_data)
    mu_model = np.zeros_like(S_model)
    weights = np.zeros_like(S_data)

    # Calculate mu on valid bins
    with np.errstate(divide='ignore', invalid='ignore'):
        mu_data[valid_bins] = S_data[valid_bins] / area_bin[valid_bins]
        mu_model[valid_bins] = S_model[valid_bins] / area_bin[valid_bins]
        # Weight = Area^2 / Counts
        weights[valid_bins] = (area_bin[valid_bins] ** 2) / (S_data[valid_bins]*Fano_Data + S_model[valid_bins]*Fano_Model)

    # --- 5. N_best Calculation ---
    w = weights[valid_bins]
    md = mu_data[valid_bins]
    mm = mu_model[valid_bins]
    
    if len(w) == 0:
        if return_bin_errors or return_profiles or return_chi2_contributions:
            extras = []
            if return_bin_errors:
                extras.append(np.zeros(n_annuli, dtype=np.float64).tolist())
            if return_profiles:
                extras.append(np.zeros(n_annuli, dtype=np.float64).tolist())
                extras.append(np.zeros(n_annuli, dtype=np.float64).tolist())
                extras.append(float("nan"))
            if return_chi2_contributions:
                extras.append(np.zeros(n_annuli, dtype=np.float64).tolist())
            n_best_slot = (float("nan"),) if return_N_best else ()
            return (float("nan"), *n_best_slot, *extras)
        if return_N_best:
            return (float("nan"), float("nan"))
        return float("nan")

    numerator = np.sum(w * md * mm)
    denominator = np.sum(w * mm * mm)
    N_best = numerator / denominator if denominator != 0 else 1.0

    # --- 6. Reduced Chi2 Calculation with N^2 model variance ---

    dof = len(w) - 1
    if dof <= 0:
        if return_N_best:
            return (float("nan"), float(N_best))
        return float("nan")

    sigma2_counts = (
        Fano_Data * S_data[valid_bins]
        + (N_best ** 2) * Fano_Model * S_model[valid_bins]
    )

    w_final = (area_bin[valid_bins] ** 2) / sigma2_counts

    residuals = md - (N_best * mm)
    chi2_contributions = w_final * (residuals ** 2)

    chi2 = np.sum(chi2_contributions)
    chi2_red = chi2 / dof


    # --- 7. Returns ---
    if return_bin_errors or return_profiles or return_chi2_contributions:
        extras = []
        if return_bin_errors:
            reflex_bin_variances = np.zeros(n_annuli, dtype=np.float64)
            reflex_bin_variances[valid_bins] = (
                S_data[valid_bins] / (area_bin[valid_bins] ** 2)
            )
            extras.append(reflex_bin_variances.tolist())
        if return_profiles:
            extras.append(mu_data.tolist())
            extras.append(mu_model.tolist())
            extras.append(float(N_best))
        if return_chi2_contributions:
            full_chi2_contributions = np.zeros(n_annuli, dtype=np.float64)
            full_chi2_contributions[valid_bins] = chi2_contributions
            extras.append(full_chi2_contributions.tolist())

        n_best_slot = (float(N_best),) if return_N_best else ()
        return (float(chi2_red), *n_best_slot, *extras)

    if return_N_best:
        return (float(chi2_red), float(N_best))
    return float(chi2_red) """
