"""One figure summarizing all fixed-condition ensembles in test_poisson_npz.

Usage, from analysis/:
    from utils.poisson_summary import plot_poisson_summary
    fig, summary = plot_poisson_summary(save_path="poisson_summary.pdf")

Requires the existing utils/poisson_diagnostics.py. No raw counts are rounded,
resampled, or pooled across conditions. Synthetic Poisson images are generated
ONLY as a comparison, not as replacements for the model output.
"""
import hashlib
import json
import logging
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .poisson_diagnostics import _load, _json_clean

_LOG = logging.getLogger(__name__)
_METRICS = ('pixel_fano_weighted', 'total_fano', 'mean_annular_correlation', 'zero_frequency_residual')
_TARGETS = (1.0, 1.0, 0.0, 0.0)
_RUN_KEYS = ('checkpoint_sha256', 'method', 'steps', 'inverse', 'source_distance_pc', 'pixel_scale_arcsec')


def _radial_groups(shape, n_annuli):
    yy, xx = np.indices(shape)
    radius = np.hypot(yy - (shape[0] - 1) / 2, xx - (shape[1] - 1) / 2)
    edges = np.linspace(0, radius.max() + 1e-10, n_annuli + 1)
    membership = np.clip(np.searchsorted(edges, radius.ravel(), side='right') - 1, 0, n_annuli - 1)
    order = np.argsort(membership, kind='stable')
    _, starts = np.unique(membership[order], return_index=True)
    return order, starts


def _metrics(flat, radial_groups):
    """Compute moments over realizations BEFORE summarizing over pixels."""
    means = flat.mean(axis=0)
    variance = flat.var(axis=0, ddof=1)
    variance[np.ptp(flat, axis=0) == 0] = 0.0
    mean_total = float(means.sum())
    pixel_fano = float(variance.sum() / mean_total) if mean_total > 0 else np.nan
    totals = flat.sum(axis=1)
    total_variance = float(totals.var(ddof=1)) if np.ptp(totals) > 0 else 0.0
    total_fano = total_variance / mean_total if mean_total > 0 else np.nan
    order, starts = radial_groups
    # Omit geometrically empty annuli; zero-variance occupied annuli are excluded below.
    annular = np.add.reduceat(flat[:, order], starts, axis=1)
    usable = np.ptp(annular, axis=0) > 0
    if np.count_nonzero(usable) >= 2:
        correlation = np.corrcoef(annular[:, usable], rowvar=False)
        off_diagonal = correlation[np.triu_indices(correlation.shape[0], k=1)]
        mean_corr = float(off_diagonal.mean())
    else:
        mean_corr = np.nan
    # The expected zero probability is fitted SEPARATELY at every pixel.
    zero_residual = float(np.mean((flat == 0).mean(axis=0) - np.exp(-means)))
    return np.array([pixel_fano, total_fano, mean_corr, zero_residual]), int(np.count_nonzero(usable))


def _condition(metadata, path):
    if 'x' not in metadata or 'nH' not in metadata:
        raise ValueError(f'{path.name}: missing scalar x/nH metadata; use archives from sample_poisson_ensemble.py.')
    x, nh = float(metadata['x']), float(metadata['nH'])
    if not (np.isfinite(x) and 0 < x < 1 and np.isfinite(nh) and nh > 0):
        raise ValueError(f'{path.name}: expected fractional 0 < x < 1 and finite nH > 0.')
    return x, nh


def _load_named(path):
    try:
        return _load(path)
    except (ValueError, KeyError, OSError) as error:
        raise ValueError(f'Cannot read ensemble {path}: {error}') from error


def _draw_poisson_summary(records, poisson_replicates, title=None):
    """Presentation only: consumes existing statistics without recomputing them."""
    if not records:
        raise ValueError('The summary contains no conditions.')
    count = len(records)
    panel_height = max(2.4, 0.34 * count + 0.6)
    fig = plt.figure(figsize=(13.2, 2 * panel_height + 3.0), facecolor='white')
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 2 * panel_height + 0.7, 0.8],
                            left=0.18, right=0.97, bottom=0.035, top=0.98, hspace=0.35)
    header = fig.add_subplot(outer[0])
    header.set_axis_off()
    header.text(0.5, 0.96, title or 'Noise statistics of generated images',
                transform=header.transAxes, ha='center', va='top', fontsize=18, fontweight='semibold')
    sample_sizes = sorted({record['n_samples'] for record in records})
    sampling_text = (f'{sample_sizes[0]} images per condition' if len(sample_sizes) == 1
                     else f'{sample_sizes[0]}–{sample_sizes[-1]} images per condition')
    header.text(0.5, 0.57, f'{count} conditions · {sampling_text}',
                transform=header.transAxes, ha='center', va='center', fontsize=11, color='#52606D')
    header.legend(handles=[
        Line2D([], [], marker='o', color='#1668A6', linestyle='None', markersize=7, label='Generated images'),
        Patch(facecolor='#CBD2D9', label='Poisson reference (95%)'),
        Line2D([], [], color='#52606D', linestyle='--', label='Poisson target'),
    ], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False,
       fontsize=10, handlelength=1.8, columnspacing=2.0)

    grid = outer[1].subgridspec(2, 2, wspace=0.27, hspace=0.80)
    axes = []
    titles = ('A  Pixel fluctuations', 'B  Total-flux fluctuations',
              'C  Correlations between regions', 'D  Zero-count frequency')
    explanations = ('Poisson = 1; lower values mean less local variation',
                    'Poisson = 1; higher values mean more total variation',
                    'Poisson = 0; positive values mean regions vary together',
                    'Poisson = 0; negative values mean too few zeros')
    xlabels = ('Variance / mean (flux-weighted)', 'Variance / mean of total counts',
               'Mean correlation between annuli', 'Observed − Poisson (percentage points)')
    labels = []
    for record in records:
        exponent = int(np.floor(np.log10(record['nH'])))
        mantissa = record['nH'] / 10.0**exponent
        separator = ',  ' if count > 5 else '\n'
        labels.append(f"$x={record['x']:.4g}$" + separator +
                      rf"$N_{{\rm H}}={mantissa:.3g}\times10^{{{exponent}}}$")

    for column, (name, target) in enumerate(zip(_METRICS, _TARGETS)):
        axis = fig.add_subplot(grid[column // 2, column % 2])
        axes.append(axis)
        multiplier = 100.0 if column == 3 else 1.0
        limits = [target * multiplier]
        for record in records:
            value = record['observed'][name]
            if value is not None and np.isfinite(value):
                limits.append(value * multiplier)
            reference = record['poisson_reference'][name]
            if reference['lower'] is not None:
                limits.extend([reference['lower'] * multiplier, reference['upper'] * multiplier])
        # Ordinary linear axes for the usual range; only extreme Fano factors
        # use a clearly labelled compressed scale, without clipping any points.
        if column < 2:
            upper = max(limits)
            if upper > 12:
                axis.set_xscale('symlog', linthresh=1, linscale=1)
                axis.set_xlim(-0.04, upper * 1.7)
                candidates = [0, 0.5, 1] + [10.0**power for power in range(1, int(np.ceil(np.log10(upper)))+1)]
                ticks = [value for value in candidates if value <= axis.get_xlim()[1]]
                axis.set_xticks(ticks, labels=[f'{value:g}' for value in ticks])
                axis.minorticks_off()
                xlabels_current = xlabels[column] + '\n(linear to 1; logarithmic above)'
            else:
                axis.set_xlim(-0.05 * max(1, upper), max(1.25, upper * 1.24))
                xlabels_current = xlabels[column]
        elif column == 2:
            axis.set_xlim(-1.05 if min(limits) < -0.15 else -0.15, 1.12)
            xlabels_current = xlabels[column]
        else:
            padding = max(0.5, (max(limits) - min(limits)) * 0.18)
            axis.set_xlim(min(limits)-padding, max(limits)+padding)
            xlabels_current = xlabels[column]
        for row, record in enumerate(records):
            reference = record['poisson_reference'][name]
            if reference['lower'] is not None:
                axis.plot([reference['lower'] * multiplier, reference['upper'] * multiplier], [row, row],
                          color='#CBD2D9', linewidth=10, solid_capstyle='butt', zorder=2)
            value = record['observed'][name]
            if value is not None and np.isfinite(value):
                displayed = value * multiplier
                axis.plot(displayed, row, 'o', color='#1668A6', markersize=7, zorder=4)
                axis.annotate(f'{displayed:.3g}', (displayed, row), xytext=(9, 0),
                              textcoords='offset points', ha='left', va='center',
                              color='#174F78', fontsize=9, zorder=5)
            else:
                axis.text(0.55, row, 'undefined', transform=axis.get_yaxis_transform(),
                          ha='center', va='center', fontsize=9, color='#7B8794')
        axis.axvline(target * multiplier, color='#52606D', linestyle=(0, (4, 4)), linewidth=1.1, zorder=1)
        axis.set_title(titles[column], loc='left', fontsize=13, fontweight='semibold', pad=29)
        axis.text(0, 1.045, explanations[column], transform=axis.transAxes,
                  fontsize=9, color='#52606D', ha='left', va='bottom')
        axis.set_xlabel(xlabels_current, fontsize=10, labelpad=9)
        axis.set_ylim(count-0.45, -0.6)
        axis.set_yticks(range(count))
        if column % 2 == 0:
            axis.set_yticklabels(labels, fontsize=10)
        else:
            axis.set_yticklabels([])
        axis.tick_params(axis='y', length=0, pad=12)
        axis.tick_params(axis='x', labelsize=10, colors='#52606D', length=3)
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#E9EDF1', linewidth=0.8)
        for side in ('top', 'right', 'left'):
            axis.spines[side].set_visible(False)
        axis.spines['bottom'].set_color('#CBD2D9')

    footer = fig.add_subplot(outer[2])
    footer.set_axis_off()
    fractions = [record['fraction_noninteger'] for record in records]
    fraction_text = (f'{100*min(fractions):.1f}%' if min(fractions) == max(fractions)
                     else f'{100*min(fractions):.1f}–{100*max(fractions):.1f}%')
    footer.text(0, 0.62,
                f'Grey bars: pointwise ranges from {poisson_replicates} Poisson ensembles per condition, matched in mean image and sample size.\n'
                f'Noninteger values: {fraction_text} across conditions. Count values are unchanged; each condition is analysed separately.\n'
                'These comparisons assess noise statistics, not mean-profile accuracy or a joint goodness-of-fit test.',
                transform=footer.transAxes, fontsize=9, color='#52606D', ha='left', va='center', linespacing=1.7)
    return fig


def plot_poisson_summary_from_json(summary_file, *, save_path=None, show=True, title=None):
    """Redraw a saved summary without repeating any Poisson simulations.

    Example:
        fig, summary = plot_poisson_summary_from_json(
            'poisson_summary.json', save_path='poisson_summary_clean.pdf')

    The JSON must be produced by plot_poisson_summary. Statistics and reference
    ranges are reused unchanged; only the layout and display units are updated.
    Panel D displays the saved fractional residual in percentage points.
    """
    source = Path(summary_file).expanduser()
    summary = json.loads(source.read_text())
    if not isinstance(summary, dict) or not summary.get('conditions') or 'poisson_replicates' not in summary:
        raise ValueError('Expected a JSON summary saved by plot_poisson_summary.')
    if save_path is not None and Path(save_path).suffix.lower() in ('', '.npz', '.json'):
        raise ValueError('save_path must be a figure filename, for example poisson_summary_clean.pdf.')
    fig = _draw_poisson_summary(summary['conditions'], summary['poisson_replicates'], title)
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches='tight')
        destination.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if show:
        plt.show()
    return fig, summary


def plot_poisson_summary(folder='test_poisson_npz', *, n_annuli=24,
                         poisson_replicates=99, seed=42, integer_tol=1e-5,
                         save_path=None, show=True, title=None):
    """Read all *.npz files in a folder and return (figure, summary_dict).

    One row per condition, sorted by x then nH; a 2x2 figure compares:

    1. SUM of individual pixel variances / SUM of pixel means. This is the
       flux-weighted pixel Fano factor, NOT the variance of pooled pixels.
    2. Variance of each realization's TOTAL counts / mean total counts.
    3. Mean signed Pearson correlation between distinct annular count totals.
       Zero-variance annuli have undefined correlations and are excluded.
    4. Mean over pixels of [observed zero fraction - exp(-pixel mean)].
       Negative values indicate fewer exact zeros than predicted by Poisson.

    References are independent Poisson image ensembles generated using each
    file's fitted pixel means, original shape and number of realizations.
    Every reference replicate recomputes means and all four statistics. The
    grey bars are 2.5th-97.5th percentile POINTWISE fitted-null reference ranges;
    they are NOT confidence intervals on the model, simultaneous bounds,
    formal p-values, or a claim of distributional equivalence. The fit uses the
    model's mean image, so this does not validate its physical brightness.

    poisson_replicates defaults to 99 for exploratory plotting; use 499 or more
    for a final figure. Work scales with replicates * realizations * pixels *
    conditions. Only one original and one reference ensemble are held at a time.
    Correlation and zero-frequency summaries can conceal local departures;
    use plot_poisson_diagnostics for the detailed per-condition plots.

    Each archive must contain samples (N,H,W) or (N,1,H,W), N>=2, x and nH;
    metadata_json from the sampler is also supported. Files are NOT concatenated.
    Duplicate conditions and differing recorded checkpoint/sampler/inverse or
    camera settings are rejected: keep each model/run comparison in its own
    folder. Missing provenance generates a warning, rather than inventing it.
    Fractions of noninteger values are computed over ALL pixels and realizations,
    including zeros; these are descriptive, not percentages of failed tests.
    Undefined statistics are shown as 'undefined' and saved as JSON null.

    Panel D displays residuals in percentage points; JSON retains fractions.
    Use plot_poisson_summary_from_json to redraw saved results without resampling.

    save_path optionally saves one figure (PDF/PNG/etc.) and an adjacent JSON
    with per-file statistics, reference ranges, provenance and interpretation.
    Existing files at save_path are replaced. Input NPZ files are never modified.
    """
    if not isinstance(n_annuli, (int, np.integer)) or n_annuli < 2:
        raise ValueError('n_annuli must be an integer >= 2.')
    if not isinstance(poisson_replicates, (int, np.integer)) or poisson_replicates < 19:
        raise ValueError('poisson_replicates must be an integer >= 19.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    if not np.isfinite(integer_tol) or not 0 <= integer_tol < 0.5:
        raise ValueError('integer_tol must be finite and satisfy 0 <= tolerance < 0.5.')
    n_annuli, poisson_replicates, integer_tol = int(n_annuli), int(poisson_replicates), float(integer_tol)
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f'Ensemble folder not found: {folder}')
    files = sorted(folder.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No .npz files found in {folder}')
    if save_path is not None and Path(save_path).suffix.lower() in ('', '.npz', '.json'):
        raise ValueError('save_path must be a figure filename, for example poisson_summary.pdf.')

    # Inspect every file before any expensive reference simulations.
    catalog, seen_conditions, provenance, shape, missing_provenance = [], {}, {}, None, []
    for path in files:
        values, metadata = _load_named(path)
        x, nh = _condition(metadata, path)
        if (x, nh) in seen_conditions:
            raise ValueError(f'Duplicate condition x={x}, nH={nh} in {seen_conditions[(x, nh)]} and {path.name}. '
                             'Keep one ensemble archive per condition; files are never pooled automatically.')
        seen_conditions[(x, nh)] = path.name
        if shape is not None and shape != values.shape[1:]:
            raise ValueError(f'{path.name}: image shape differs from the other archives; use matching camera/crop settings.')
        shape = values.shape[1:]
        absent = [key for key in _RUN_KEYS if metadata.get(key) is None]
        if absent:
            missing_provenance.append({'file': path.name, 'missing_keys': absent})
        for key in _RUN_KEYS:
            value = metadata.get(key)
            if value is not None:
                if key in provenance and provenance[key] != value:
                    raise ValueError(f'{path.name}: mixed {key} across archives. Use a separate folder for each model/sampling setup.')
                provenance[key] = value
        # Checkpoint statistics should also agree within one run when recorded.
        for key in ('condition_stats', 'anscombe_global_max'):
            if key in metadata:
                if key in provenance and provenance[key] != metadata[key]:
                    raise ValueError(f'{path.name}: inconsistent {key}; do not mix normalization settings.')
                provenance[key] = metadata[key]
        catalog.append((x, nh, path, metadata, values.shape[0]))
        del values
    catalog.sort(key=lambda item: (item[0], item[1]))
    if missing_provenance:
        warnings.warn('Some archives lack checkpoint/sampler/camera provenance; consistency cannot be fully verified. '
                      'See summary["missing_provenance"].', stacklevel=2)
    if any(item[4] < 100 for item in catalog):
        warnings.warn('Some ensembles have fewer than 100 realizations; estimates can be noisy and reference ranges broad.', stacklevel=2)

    groups = _radial_groups(shape, n_annuli)
    records = []
    for index, (x, nh, path, metadata, count) in enumerate(catalog):
        _LOG.info('Poisson summary: condition %d/%d, %s', index+1, len(catalog), path.name)
        values, _ = _load_named(path)
        flat = values.reshape(count, -1)
        means = flat.mean(axis=0)
        observed, usable = _metrics(flat, groups)
        fractional = float(np.mean(np.abs(flat - np.rint(flat)) > integer_tol))
        # Stream assignment depends on the condition, not file order or its name.
        digest = hashlib.sha256(f'{x:.17g}|{nh:.17g}'.encode()).digest()
        stream_words = np.frombuffer(digest[:16], dtype='<u4').tolist()
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), *stream_words]))
        null = np.full((poisson_replicates, len(_METRICS)), np.nan)
        for replicate in range(poisson_replicates):
            poisson_images = rng.poisson(means, size=flat.shape)
            null[replicate], _ = _metrics(poisson_images, groups)
            del poisson_images
        references = {}
        for column, name in enumerate(_METRICS):
            finite = null[:, column][np.isfinite(null[:, column])]
            references[name] = {
                'lower': float(np.quantile(finite, 0.025)) if finite.size else None,
                'median': float(np.median(finite)) if finite.size else None,
                'upper': float(np.quantile(finite, 0.975)) if finite.size else None,
                'finite_replicates': int(finite.size),
            }
        records.append({
            'file': str(path), 'x': x, 'nH': nh, 'n_samples': count,
            'image_shape': list(shape), 'total_mean': float(means.sum()),
            'fraction_noninteger': fractional, 'annuli_with_nonzero_variance': usable,
            'observed': dict(zip(_METRICS, observed.tolist())), 'poisson_reference': references,
            'metadata': metadata,
        })
        del values, flat, null

    fig = _draw_poisson_summary(records, poisson_replicates, title)
    summary = _json_clean({
        'folder': str(folder), 'n_conditions': len(records), 'n_annuli': n_annuli,
        'poisson_replicates': poisson_replicates, 'seed': int(seed), 'integer_tolerance': integer_tol,
        'common_provenance': provenance, 'missing_provenance': missing_provenance,
        'metric_definitions': {
            'pixel_fano_weighted': 'sum_i Var_across_samples(X_i) / sum_i Mean_across_samples(X_i)',
            'total_fano': 'Var_across_samples(sum_i X_i) / Mean_across_samples(sum_i X_i)',
            'mean_annular_correlation': 'mean signed Pearson correlation over distinct pairs of nonconstant annular totals',
            'zero_frequency_residual': 'mean_i [fraction_samples(X_i == 0) - exp(-mean_samples(X_i))]',
        },
        'reference_method': 'Independent Poisson pixels at fitted generated pixel means; same N and image shape. '
                            'All statistics are refitted in each replicate. 2.5th-97.5th quantiles are pointwise, not simultaneous.',
        'notes': ['No raw values are rounded or pooled across conditions.',
                  'Correlations exclude zero-variance annuli; undefined null replicates are omitted and their number recorded.',
                  'Signed averages and flux weighting can hide local deviations; inspect per-condition diagnostics too.',
                  'Matching these summaries does not prove a Poisson joint distribution or validate physical mean profiles.',
                  'Fractional values are counted over every pixel and sample, including zeros.'],
        'conditions': records,
    })
    if save_path is not None:
        destination = Path(save_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches='tight')
        destination.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if show:
        plt.show()
    return fig, summary
