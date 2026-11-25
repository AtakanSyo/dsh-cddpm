"""Poisson diagnostics for independent repeated images at ONE fixed condition.

Public entry point: plot_poisson_diagnostics(npz_path_or_array, ...).
Use raw count-domain output, not PNGs, normalized images, or pooled conditions.
A matching figure supports Poisson-like behavior; it cannot prove a joint law.
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


def _load(data):
    metadata = {}
    if isinstance(data, (str, Path)):
        with np.load(data, allow_pickle=False) as archive:
            values = np.asarray(archive['samples'], dtype=np.float64)
            if 'metadata_json' in archive:
                metadata = json.loads(str(archive['metadata_json'].item()))
            for key in ('x', 'nH'):
                if key in archive:
                    metadata[key] = float(archive[key].item())
    else:
        values = np.asarray(data, dtype=np.float64)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3 or values.shape[0] < 2 or min(values.shape[1:]) < 2:
        raise ValueError('Expected (realizations >= 2, height >= 2, width >= 2) at ONE fixed condition; do not pool conditions.')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Samples must contain finite, nonnegative raw count values.')
    return values, metadata


def _ratio(variance, mean):
    return np.divide(variance, mean, out=np.full_like(np.asarray(mean, dtype=float), np.nan), where=np.asarray(mean) > 0)


def _ks_statistic(values):
    """Discrete KS distance, evaluated on both sides of every observed jump."""
    unique, counts = np.unique(values, return_counts=True)
    empirical = np.cumsum(counts) / len(values)
    before = empirical - counts / len(values)
    rate = values.mean()
    return float(max(np.max(np.abs(empirical - poisson.cdf(unique, rate))),
                     np.max(np.abs(before - poisson.cdf(unique - 1, rate)))))


def _poisson_fit(values, integer_support, bootstrap, rng):
    if not integer_support:
        return {'statistic': None, 'p_value': None, 'reason': 'Noninteger raw pixel values; strict Poisson count support is not satisfied.'}
    # Only numerical deviations <= integer_tol are canonicalized for this test.
    values = np.rint(values)
    rate = float(values.mean())
    if rate == 0:
        return {'statistic': 0.0, 'p_value': None, 'reason': 'All zero: compatible with rate zero, but uninformative about nonzero-rate noise.'}
    observed = _ks_statistic(values)
    simulated = [_ks_statistic(rng.poisson(rate, size=len(values))) for _ in range(bootstrap)]
    return {'statistic': observed, 'p_value': float((1 + np.count_nonzero(np.asarray(simulated) >= observed)) / (bootstrap + 1)),
            'reason': 'Discrete KS, parametric bootstrap with the Poisson rate refitted in every replicate.'}


def _summary(values, n_annuli, pixel_scale, pixel, integer_tol):
    n, height, width = values.shape
    yy, xx = np.indices((height, width))
    radius = np.hypot(yy - (height - 1) / 2, xx - (width - 1) / 2)
    # Include the corners; disjoint bins partition the ENTIRE image.
    edges = np.linspace(0, radius.max() + 1e-10, n_annuli + 1)
    membership = np.clip(np.searchsorted(edges, radius, side='right') - 1, 0, n_annuli - 1)
    annular = np.stack([values[:, membership == k].sum(axis=1) for k in range(n_annuli)], axis=1)
    means, variances = values.mean(axis=0), values.var(axis=0, ddof=1)
    ring_mean, ring_var = annular.mean(axis=0), annular.var(axis=0, ddof=1)
    cov = np.cov(annular, rowvar=False)
    denominator = np.sqrt(ring_var[:, None] * ring_var[None, :])
    correlation = np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 0)
    np.fill_diagonal(correlation, np.nan)
    # Also check immediate neighbours; annular averaging can hide local dependence.
    residual = values - means
    standardized = np.divide(residual, np.sqrt(variances), out=np.zeros_like(residual), where=variances > 0)
    neighbour_r = []
    for a, b, valid in ((standardized[:, :, :-1], standardized[:, :, 1:], (variances[:, :-1] > 0) & (variances[:, 1:] > 0)),
                        (standardized[:, :-1, :], standardized[:, 1:, :], (variances[:-1, :] > 0) & (variances[1:, :] > 0))):
        neighbour_r.extend(((a * b).sum(axis=0) / (n - 1))[valid].tolist())
    total = values.sum(axis=(1, 2))
    deviations = np.abs(values - np.rint(values))
    fractional = float(np.mean(deviations > integer_tol))
    finite_correlations = np.abs(correlation[np.isfinite(correlation)])
    summary = {
        'n_samples': n, 'image_shape': [height, width], 'integer_tolerance': integer_tol,
        'fraction_noninteger_pixels': fractional, 'integer_count_support': fractional == 0.0,
        'max_distance_from_integer': float(deviations.max()),
        'total_mean': float(total.mean()), 'total_variance': float(total.var(ddof=1)),
        'total_fano': float(_ratio(total.var(ddof=1), total.mean())),
        'median_pixel_fano': float(np.nanmedian(_ratio(variances, means))) if np.any(means > 0) else None,
        'mean_absolute_zero_probability_error': float(np.mean(np.abs((values == 0).mean(axis=0) - np.exp(-means)))),
        'annular_radii_arcsec': ((edges[:-1] + edges[1:]) * 0.5 * pixel_scale).tolist(),
        'annular_mean': ring_mean.tolist(), 'annular_variance': ring_var.tolist(),
        'annular_fano': _ratio(ring_var, ring_mean).tolist(),
        'max_abs_annular_correlation': float(finite_correlations.max()) if finite_correlations.size else None,
        'mean_neighbour_correlation': float(np.mean(neighbour_r)) if neighbour_r else None,
        'fixed_pixel_row_col': list(pixel), 'fixed_pixel_mean': float(values[:, pixel[0], pixel[1]].mean()),
    }
    return summary, {'mean': means, 'variance': variances, 'annular': annular, 'correlation': correlation,
                     'total': total, 'pixel': values[:, pixel[0], pixel[1]], 'zero': (values == 0).mean(axis=0)}


def _cdf_plot(axis, data, label, color):
    unique, counts = np.unique(data, return_counts=True)
    rate = float(data.mean())
    low = min(float(data.min()), float(poisson.ppf(0.001, rate)))
    high = max(float(data.max()), float(poisson.ppf(0.999, rate)), low + 1)
    axis.step(np.r_[low - 1, unique, high + 1], np.r_[0, np.cumsum(counts) / len(data), 1],
              where='post', color=color, label=f'{label}: empirical')
    # A bounded grid keeps high photon-count totals inexpensive.
    grid = np.unique(np.floor(np.linspace(max(-1, low - 1), high + 1, min(2000, int(high - low) + 3))))
    axis.step(grid, poisson.cdf(grid, rate), where='post', color=color, linestyle='--',
              label=f'{label}: fitted Poisson')
    axis.set(xlabel='Raw counts', ylabel='Cumulative probability', ylim=(-0.02, 1.02))
    axis.legend(fontsize=8)


def _json_clean(value):
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def plot_poisson_diagnostics(samples, *, reference_samples=None, n_annuli=24,
                             pixel_scale_arcsec=None, pixel=None, bootstrap=199,
                             seed=42, integer_tol=1e-5, save_path=None, show=True,
                             title=None):
    """Plot six diagnostics and return ``(figure, summary_dict)``.

    Parameters
    ----------
    samples : path or array (N,H,W), also accepts (N,1,H,W)
        Independent count-domain images for ONE fixed physical condition. An NPZ
        from sample_poisson_ensemble.py carries condition/provenance metadata.
    reference_samples : optional path or array
        Repeated independent simulator images at the SAME condition, exposure,
        crop and pixel scale. A single simulator image is insufficient. Array
        callers are responsible for matching conditions and camera settings.
    n_annuli : int >= 2
        Disjoint radial regions, including corners, centred at the image centre.
    pixel : (row, column), optional
        A pixel chosen in advance; default is (H//2,W//2). Do not choose the pixel
        after inspecting p-values without accounting for that selection.
    bootstrap : int >= 19
        Number of Poisson simulations. Default 199 is exploratory; use >= 999
        for reported tests. Annular bands are pointwise fitted-null simulation
        envelopes, NOT simultaneous confidence bands or parameter intervals.
    integer_tol : float
        Absolute numerical tolerance in photon units. Continuous outputs are
        kept unchanged in plots. Strict count tests are omitted if any pixel
        lies farther than this tolerance from an integer.
    save_path : path, optional
        Save figure (PNG/PDF/etc.) and a neighbouring .json summary.

    Notes
    -----
    The null is repeated independent Poisson pixels with fixed, spatially varying
    rates. Rates are estimated separately per pixel/region; never pool counts at
    different radii or conditions. Total/ring tests additionally assume spatial
    independence. Common flux changes, fixed photon totals, or simulator packet
    weighting can violate this stronger null even if individual marginals fit.
    Nonrejection does not prove Poisson behavior. Two discrete KS bootstrap tests
    (total and preselected pixel) are Holm-adjusted within THIS function call;
    correction across additional conditions/checkpoints remains the user's job.
    No rates from a physical halo model are used: this diagnoses stochastic
    behavior, not correctness of the mean brightness profile.
    """
    if not isinstance(n_annuli, (int, np.integer)) or n_annuli < 2:
        raise ValueError('n_annuli must be an integer >= 2.')
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 19:
        raise ValueError('bootstrap must be an integer >= 19.')
    if not np.isfinite(integer_tol) or integer_tol < 0 or integer_tol >= 0.5:
        raise ValueError('integer_tol must be finite, nonnegative, and < 0.5 count.')
    values, metadata = _load(samples)
    if values.shape[0] < 100:
        warnings.warn('Fewer than 100 realizations: moment/correlation estimates are noisy and tests have limited power.', stacklevel=2)
    pixel_scale = float(pixel_scale_arcsec if pixel_scale_arcsec is not None else metadata.get('pixel_scale_arcsec', 45.0))
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError('pixel_scale_arcsec must be positive and finite.')
    pixel = tuple(pixel) if pixel is not None else (values.shape[1] // 2, values.shape[2] // 2)
    if len(pixel) != 2 or any(not isinstance(v, (int, np.integer)) or not 0 <= v < d for v, d in zip(pixel, values.shape[1:])):
        raise ValueError('pixel must be an in-bounds integer (row, column).')
    summary, measurements = _summary(values, n_annuli, pixel_scale, pixel, integer_tol)
    summary['metadata'] = metadata
    summary['bootstrap_replicates'] = int(bootstrap)
    summary['bootstrap_seed'] = seed
    rng = np.random.default_rng(seed)
    tests = {name: _poisson_fit(measurements[name], summary['integer_count_support'], bootstrap, rng)
             for name in ('total', 'pixel')}
    ranked = sorted((result['p_value'], name) for name, result in tests.items() if result['p_value'] is not None)
    previous = 0.0
    for index, (p, name) in enumerate(ranked):
        previous = max(previous, min(1.0, (len(ranked) - index) * p))
        tests[name]['p_value_holm_within_figure'] = previous
    summary['poisson_fit_tests'] = tests
    ring_mean = measurements['annular'].mean(axis=0)
    simulated_fano = np.full((bootstrap, n_annuli), np.nan)
    # Batch over bootstrap replicates to avoid bootstrap*N*K memory growth.
    for index in range(bootstrap):
        simulated = rng.poisson(ring_mean, size=(values.shape[0], n_annuli))
        simulated_fano[index] = _ratio(simulated.var(axis=0, ddof=1), simulated.mean(axis=0))
    lower, upper = np.full(n_annuli, np.nan), np.full(n_annuli, np.nan)
    for index in range(n_annuli):
        finite = simulated_fano[:, index][np.isfinite(simulated_fano[:, index])]
        if finite.size:
            lower[index], upper[index] = np.quantile(finite, (0.025, 0.975))
    summary['annular_fano_poisson_pointwise_envelope'] = {'lower': lower.tolist(), 'upper': upper.tolist()}
    summary['notes'] = [
        'Each archive/array must contain independent repeated images at one fixed condition.',
        'Mean-variance agreement is necessary, not sufficient, for independent Poisson counts.',
        'Continuous outputs can be Poisson-like in moments but are not exact Poisson counts.',
        'Annular 95% bands are pointwise fitted-null simulations; sparse all-zero replicates have undefined Fano factors.',
        'Total/radial tests assume independent pixels; annular correlations do not exhaust all possible spatial dependence.',
        'The two p-values are adjusted only within this figure; low p-values indicate incompatibility, large p-values do not establish equivalence.',
        'This assesses stochastic behavior, not accuracy of the predicted mean image.',
    ]
    datasets = [(values, summary, measurements, 'Generated', 'tab:blue')]
    if reference_samples is not None:
        reference, reference_metadata = _load(reference_samples)
        if reference.shape[1:] != values.shape[1:]:
            raise ValueError('Reference images must have the same crop and pixel dimensions as generated images.')
        for key in ('x', 'nH', 'pixel_scale_arcsec', 'source_distance_pc'):
            if key in metadata and key in reference_metadata and not np.isclose(metadata[key], reference_metadata[key], rtol=1e-8, atol=0):
                raise ValueError(f'Reference metadata mismatch for {key}. Use the SAME physical condition and camera.')
        ref_summary, ref_measurements = _summary(reference, n_annuli, pixel_scale, pixel, integer_tol)
        summary['reference'] = ref_summary
        datasets.append((reference, ref_summary, ref_measurements, 'Simulator', 'tab:orange'))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    maximum = max(float(max(d[2]['mean'].max(), d[2]['variance'].max())) for d in datasets)
    for _, info, measured, label, color in datasets:
        axes[0, 0].scatter(measured['mean'], measured['variance'], s=7, alpha=0.35, color=color, label=label, rasterized=True)
        axes[0, 1].plot(info['annular_radii_arcsec'], info['annular_fano'], '.-', color=color, label=label)
        axes[0, 2].scatter(np.exp(-measured['mean']), measured['zero'], s=8, alpha=0.35, color=color, label=label, rasterized=True)
        _cdf_plot(axes[1, 0], measured['total'], label, color)
        _cdf_plot(axes[1, 1], measured['pixel'], label, color)
    axes[0, 0].plot([0, max(maximum, 1)], [0, max(maximum, 1)], 'k--', linewidth=1, label='Poisson: variance = mean')
    axes[0, 0].set(xlabel='Pixel mean across realizations', ylabel='Pixel variance across realizations', title='Pixel mean–variance relation')
    axes[0, 0].set_xscale('symlog', linthresh=0.01)
    axes[0, 0].set_yscale('symlog', linthresh=0.01)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].fill_between(summary['annular_radii_arcsec'], lower, upper, alpha=0.2, color='gray', label='95% pointwise Poisson envelope')
    axes[0, 1].axhline(1, color='black', linestyle='--', linewidth=1)
    axes[0, 1].set(xlabel='Radius (arcsec)', ylabel='Variance / mean of annular total', title='Annular Fano factors (independent-pixel null)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 2].set(xlabel='Poisson prediction: exp(−pixel mean)', ylabel='Observed fraction of exact zeros', title='Zero-count probabilities', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1, 0].set_title(f"Whole-image total; Fano = {summary['total_fano']:.3g}")
    axes[1, 1].set_title(f'Preselected pixel {pixel}')
    corr = axes[1, 2].imshow(np.ma.masked_invalid(measurements['correlation']), vmin=-1, vmax=1, cmap='RdBu_r', origin='lower',
                           extent=(0.5, n_annuli + 0.5, 0.5, n_annuli + 0.5))
    axes[1, 2].set(xlabel='Annulus index', ylabel='Annulus index', title='Generated: correlations between annular totals')
    fig.colorbar(corr, ax=axes[1, 2], label='Pearson correlation (diagonal omitted)', shrink=0.85)
    condition = f"; x={metadata['x']:.4g}, NH={metadata['nH']:.3g}" if 'x' in metadata and 'nH' in metadata else ''
    subtitle = f"N={len(values)}{condition}; fractional pixel values={100 * summary['fraction_noninteger_pixels']:.2f}%"
    support_note = 'Count tests available in JSON summary' if summary['integer_count_support'] else 'Strict count tests omitted: raw outputs have noninteger support'
    fig.suptitle(f"{title or 'Poisson diagnostics at a fixed condition'}\n{subtitle}\n{support_note}", fontsize=13)
    summary = _json_clean(summary)
    if save_path is not None:
        destination = Path(save_path).expanduser()
        if destination.suffix.lower() == '.json':
            raise ValueError('save_path must be an image/PDF filename, not a JSON filename.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        destination.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if show:
        plt.show()
    return fig, summary
