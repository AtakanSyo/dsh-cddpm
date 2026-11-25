from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

def radial_reduced_chi2_area2_bin_filtered(
    data,
    model,
    V=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 43.2,
    bin_filter: float = 10.0,
    return_bin_errors: bool = False,
    return_profiles: bool = False,
    return_chi2_contributions: bool = False,
    return_N_best: bool = False,
    use_N: bool = True,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
):
    """
    Calculate the radial reduced chi-squared between a data image and a
    normalized model image.

    The images are reduced to radial surface-brightness profiles. A
    multiplicative model normalization, N_best, is fitted by directly
    minimizing

        chi2(N) = sum_i [
            (mu_data,i - N * mu_model,i)^2
            /
            (
                Fano_Data * S_data,i / A_i^2
                + N^2 * Fano_Model * V_model,i / A_i^2
            )
        ].

    Equivalently, in count-space variance notation,

        chi2(N) = sum_i [
            A_i^2 * (mu_data,i - N * mu_model,i)^2
            /
            (
                Fano_Data * S_data,i
                + N^2 * Fano_Model * V_model,i
            )
        ].

    Parameters
    ----------
    data : array_like
        Observed count image.

    model : array_like
        Model count image.

    V : array_like, optional
        Per-pixel variance of the model before normalization.

        For a deterministic bilinear interpolant in
        (x, log10(nH)),

            V = sum_k w_k^2 I_k.

        If V is None, the model is treated as a Poisson realization and
        its per-pixel variance is taken to be equal to its counts.

    n_annuli : int
        Number of radial annuli.

    pixel_scale_arcsec : float
        Pixel width in arcseconds.

    bin_filter : float
        Only annuli with S_data > bin_filter are included.

    return_bin_errors : bool
        Return the total one-sigma surface-brightness uncertainty per
        annulus, evaluated at N_best.

    return_profiles : bool
        Return the data and unnormalized model radial profiles, along
        with N_best.

    return_chi2_contributions : bool
        Return each annulus's contribution to chi-squared.

    return_N_best : bool
        Return the fitted model normalization.

    Fano_Data : float
        Multiplicative variance factor for the data.

    Fano_Model : float
        Multiplicative variance factor for the model.

    Returns
    -------
    float or tuple
        Reduced chi-squared, optionally followed by N_best and requested
        diagnostic arrays.

    Notes
    -----
    The degrees of freedom are

        dof = number of valid annuli - 1,

    because one normalization parameter is fitted.

    This minimizes the explicitly defined chi-squared statistic. It is
    not a full Gaussian maximum-likelihood fit because the variance
    depends on N and no logarithmic variance term is included.
    """

    # ------------------------------------------------------------------
    # 2. Validate images.
    # ------------------------------------------------------------------
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    model = np.squeeze(np.asarray(model, dtype=np.float64))

    if data.shape != model.shape:
        raise ValueError(
            f"data shape {data.shape} must match model shape {model.shape}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError("data contains NaN or infinite values.")

    if not np.all(np.isfinite(model)):
        raise ValueError("model contains NaN or infinite values.")

    if np.any(data < 0.0):
        raise ValueError("data contains negative counts.")

    if np.any(model < 0.0):
        raise ValueError("model contains negative counts.")

    if V is not None:
        V = np.squeeze(np.asarray(V, dtype=np.float64))

        if V.ndim != 2:
            raise ValueError(
                f"V must be 2-D after squeezing; got shape {V.shape}."
            )

        if V.shape != model.shape:
            raise ValueError(
                f"V shape {V.shape} must match model shape {model.shape}."
            )

        if not np.all(np.isfinite(V)):
            raise ValueError("V contains NaN or infinite values.")

        if np.any(V < 0.0):
            raise ValueError("V contains negative variance values.")

    height, width = data.shape

    # ------------------------------------------------------------------
    # 3. Construct radial bins.
    # ------------------------------------------------------------------
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    yy, xx = np.indices((height, width))

    radius_pixels = np.sqrt(
        (yy - cy) ** 2
        + (xx - cx) ** 2
    )

    edges = np.linspace(
        0.0,
        radius_pixels.max(),
        n_annuli + 1,
    )

    bins = np.digitize(
        radius_pixels.ravel(),
        edges,
        right=False,
    ) - 1

    bins = np.clip(
        bins,
        0,
        n_annuli - 1,
    )

    # ------------------------------------------------------------------
    # 4. Sum counts and model variance within annuli.
    # ------------------------------------------------------------------
    S_data = np.bincount(
        bins,
        weights=data.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    S_model = np.bincount(
        bins,
        weights=model.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    N_pixels = np.bincount(
        bins,
        minlength=n_annuli,
    ).astype(np.float64)

    if V is None:
        V_model = S_model.copy()
    else:
        V_model = np.bincount(
            bins,
            weights=V.ravel(),
            minlength=n_annuli,
        ).astype(np.float64)
    # ------------------------------------------------------------------
    # 6. Convert counts to surface brightness.
    # ------------------------------------------------------------------
    area_per_pixel = pixel_scale_arcsec**2
    area_bin = N_pixels * area_per_pixel

    valid_bins = (
        (S_data > bin_filter)
        & (area_bin > 0.0)
        & np.isfinite(S_data)
        & np.isfinite(S_model)
        & np.isfinite(V_model)
    )

    valid_indices = np.flatnonzero(valid_bins)

    def empty_result():
        extras = []

        if return_bin_errors:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )

        if return_profiles:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )
            extras.append(float("nan"))

        if return_chi2_contributions:
            extras.append(
                np.zeros(n_annuli, dtype=np.float64).tolist()
            )

        n_best_slot = (
            (float("nan"),)
            if return_N_best
            else ()
        )

        if extras or return_N_best:
            return (
                float("nan"),
                *n_best_slot,
                *extras,
            )

        return float("nan")

    if len(valid_indices) == 0:
        return empty_result()

    mu_data = np.zeros(n_annuli, dtype=np.float64)
    mu_model = np.zeros(n_annuli, dtype=np.float64)

    mu_data[valid_bins] = (
        S_data[valid_bins]
        / area_bin[valid_bins]
    )

    mu_model[valid_bins] = (
        S_model[valid_bins]
        / area_bin[valid_bins]
    )

    md = mu_data[valid_bins]
    mm = mu_model[valid_bins]

    Sd = S_data[valid_bins]
    Vm = V_model[valid_bins]
    Ab = area_bin[valid_bins]

# ------------------------------------------------------------------
    # 7. Determine model normalization.
    #    If use_N is True, fit N_best by minimizing the reported chi2.
    #    If use_N is False, fix N = 1 (no parameter fitted), which is
    #    appropriate when data and model share the same absolute photon
    #    normalization (e.g. cDDPM / interpolation / RefleX at the same
    #    photon budget). Fixing N avoids the small-nu bias that the
    #    fitted normalization introduces into the reduced chi2.
    # ------------------------------------------------------------------
    def chi2_for_normalization(
        normalization: float,
    ) -> float:
        if (
            not np.isfinite(normalization)
            or normalization < 0.0
        ):
            return float("inf")

        variance_counts = (
            Fano_Data * Sd
            + normalization**2 * Fano_Model * Vm
        )

        positive_variance = variance_counts > 0.0

        if not np.all(positive_variance):
            return float("inf")

        residuals = (
            md
            - normalization * mm
        )

        contributions = (
            Ab**2
            * residuals**2
            / variance_counts
        )

        value = np.sum(contributions)

        if not np.isfinite(value):
            return float("inf")

        return float(value)

    if use_N:
        # Obtain a stable scale estimate for the optimization bounds.
        model_sum = float(np.sum(S_model[valid_bins]))
        data_sum = float(np.sum(S_data[valid_bins]))

        if model_sum > 0.0:
            normalization_scale = data_sum / model_sum
        else:
            normalization_scale = 1.0

        normalization_scale = max(
            normalization_scale,
            1.0,
        )

        upper_bound = max(
            10.0,
            10.0 * normalization_scale,
        )

        optimization = minimize_scalar(
            chi2_for_normalization,
            bounds=(0.0, upper_bound),
            method="bounded",
            options={
                "xatol": 1e-10,
                "maxiter": 1000,
            },
        )

        if (
            not optimization.success
            or not np.isfinite(optimization.fun)
        ):
            raise RuntimeError(
                "Could not determine N_best. "
                f"Optimizer message: {optimization.message}"
            )

        N_best = float(optimization.x)

        # Warn through an exception if the optimum appears constrained by
        # the automatically selected upper bound.
        if N_best >= 0.999 * upper_bound:
            raise RuntimeError(
                "The fitted normalization reached the optimization upper "
                f"bound ({upper_bound:.6g}). Increase the bound or inspect "
                "the data/model scaling."
            )
    else:
        # No normalization fitted: fix N = 1.
        N_best = 1.0

    # ------------------------------------------------------------------
    # 8. Calculate final chi-squared contributions.
    # ------------------------------------------------------------------
    sigma2_counts = (
        Fano_Data * Sd
        + N_best**2 * Fano_Model * Vm
    )

    if np.any(sigma2_counts <= 0.0):
        raise ValueError(
            "One or more valid annuli have non-positive total variance."
        )

    residuals = (
        md
        - N_best * mm
    )

    chi2_contributions_valid = (
        Ab**2
        * residuals**2
        / sigma2_counts
    )

    chi2 = float(
        np.sum(chi2_contributions_valid)
    )

    # Degrees of freedom: subtract one only if a normalization was fitted.
    if use_N:
        dof = len(valid_indices) - 1
    else:
        dof = len(valid_indices)

    if dof <= 0:
        chi2_red = float("nan")
    else:
        chi2_red = chi2 / dof

    # ------------------------------------------------------------------
    # 9. Assemble optional outputs.
    # ------------------------------------------------------------------
    extras = []

    if return_bin_errors:
        total_profile_variance = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        total_profile_variance[valid_bins] = (
            sigma2_counts
            / Ab**2
        )

        total_profile_errors = np.sqrt(
            total_profile_variance
        )

        extras.append(
            total_profile_errors.tolist()
        )

    if return_profiles:
        extras.append(mu_data.tolist())
        extras.append(mu_model.tolist())
        extras.append(float(N_best))

    if return_chi2_contributions:
        full_contributions = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        full_contributions[valid_bins] = (
            chi2_contributions_valid
        )

        extras.append(
            full_contributions.tolist()
        )

    n_best_slot = (
        (float(N_best),)
        if return_N_best
        else ()
    )

    if extras or return_N_best:
        return (
            float(chi2_red),
            *n_best_slot,
            *extras,
        )

    return float(chi2_red)


def _call_chi2_pair(mat_a, mat_b, *, V, n_annuli, pixel_scale_arcsec, bin_filter, need_contributions, Fano_Data=1.0, Fano_Model=1.0, use_N=True):
    """Call radial_reduced_chi2_area2_bin_filtered and return a normalized result tuple."""
    val, errors, mu_data, mu_model, N_best, *extra = radial_reduced_chi2_area2_bin_filtered(
        mat_a, mat_b,
        V=V,
        n_annuli=n_annuli,
        pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter,
        return_bin_errors=True,
        return_profiles=True,
        return_chi2_contributions=need_contributions,
        Fano_Data=Fano_Data,
        Fano_Model=Fano_Model,
        use_N=use_N,
    )
    contrib = np.asarray(extra[0], dtype=np.float64) if need_contributions and extra else np.zeros(n_annuli, dtype=np.float64)
    valid = np.asarray(errors, dtype=np.float64) > 0.0
    dof = max(int(np.count_nonzero(valid)) - 1, 1)
    summary_str = f"N_best={N_best:.4f}"
    helper_data = (np.asarray(errors, dtype=np.float64), np.asarray(mu_data, dtype=np.float64), np.asarray(mu_model, dtype=np.float64), float(N_best))
    return val, contrib, dof, summary_str, helper_data


def radial_reduced_poisson_deviance_abs(
    data,
    model,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 43.2,
    bin_filter: float = 0.0,
    return_profiles: bool = False,
    return_deviance_contributions: bool = False,
    return_expected_counts: bool = False,
    return_counts: bool = False,
    use_N: bool = False,
    return_N_best: bool = False,
    return_dof: bool = False,
):
    """
    Calculate the radial absolute two-sample Poisson deviance between
    a RefleX count image and a generated count image.

    Both images are reduced to radial annular count profiles.

    For annulus i, let

        R_i = RefleX counts,
        G_i = generated counts.

    Under the null hypothesis that both images are independent Poisson
    realizations of the same underlying expected count lambda_i,

        R_i ~ Poisson(lambda_i),
        G_i ~ Poisson(lambda_i),

    the maximum-likelihood estimate of the common expectation is

        lambda_hat_i = (R_i + G_i) / 2.

    The two-sample Poisson deviance is

        D = 2 * sum_i [
            R_i * log(R_i / lambda_hat_i)
            +
            G_i * log(G_i / lambda_hat_i)
        ]

    or equivalently

        D = 2 * sum_i [
            R_i * log(2 R_i / (R_i + G_i))
            +
            G_i * log(2 G_i / (R_i + G_i))
        ].

    The convention

        0 * log(0 / x) = 0

    is used. If R_i = G_i = 0, that annulus contributes zero.

    Because no overall normalization parameter is fitted, the nominal
    degrees of freedom are

        dof = K,

    where K is the number of included annuli with R_i > 0. Annuli with
    no RefleX/data photons are excluded, regardless of the model count.

    A descriptive reduced deviance is therefore

        D_red = D / K.

    Optional multiplicative normalization (``use_N``)
    ---------------------------------------------------
    When ``use_N=True``, a multiplicative normalization N is fitted so
    that the RefleX side is compared at N times the generated side's
    rate, i.e. R_i ~ Poisson(N * lambda_i), G_i ~ Poisson(lambda_i), for
    a shared per-annulus shape lambda_i. This generalizes the pooled
    estimate to

        lambda_hat_i(N) = (R_i + G_i) / (N + 1),

    which reduces to the original lambda_hat_i = (R_i + G_i) / 2 when
    N = 1 (the fixed, unnormalized case). The deviance generalizes to

        D(N) = 2 * sum_i [
            R_i * log(R_i / (N * lambda_hat_i(N)))
            +
            G_i * log(G_i / lambda_hat_i(N))
        ].

    The maximum-likelihood N has the closed form

        N_best = sum_i(R_i) / sum_i(G_i),

    i.e. the ratio of total RefleX counts to total generated counts
    over the valid annuli. This is the direct Poisson-deviance analogue
    of the ``N`` fitted by ``radial_reduced_chi2_area2_bin_filtered``,
    which likewise scales the generated side to match the data's
    absolute photon normalization before comparing morphology.

    If the total generated counts are zero (sum_i(G_i) = 0), N_best is
    not identifiable from the data and falls back to N_best = 1.0 (no
    normalization).

    Because fitting N consumes one degree of freedom, the reduced
    deviance becomes

        D_red = D(N_best) / (K - 1)

    when ``use_N=True`` (K - 1 <= 0 yields NaN, as for the K <= 0 case).

    Parameters
    ----------
    data : array_like
        RefleX count image.

    model : array_like
        Generated count image.

        For the strict two-sample Poisson interpretation, these values
        should represent a Poisson count realization. Continuous
        predicted intensities do not strictly satisfy the statistical
        assumptions of this two-sample deviance.

    n_annuli : int
        Number of radial annuli.

    pixel_scale_arcsec : float
        Pixel width in arcseconds. This does not affect the deviance,
        which is calculated in count space, but is used when returning
        radial surface-brightness profiles.

    bin_filter : float
        Only annuli with R >= bin_filter (RefleX/data counts) are
        included. The default, 0.0, imposes no additional restriction
        beyond N_pixels > 0 and the requirement that R > 0. Thus an
        annulus is excluded whenever the RefleX/data count is zero,
        regardless of the model count. Passing a positive value
        (e.g. 10.0) additionally excludes low-count annuli from both the
        deviance sum and dof, mirroring ``bin_filter`` in
        ``radial_reduced_chi2_area2_bin_filtered``.

    return_profiles : bool
        If True, return the RefleX and generated radial
        surface-brightness profiles.

    return_deviance_contributions : bool
        If True, return the contribution of every annulus to the total
        Poisson deviance.

    return_expected_counts : bool
        If True, return the pooled maximum-likelihood expected counts
        lambda_hat_i (see ``use_N`` below; this is the generated-side
        expected count -- the RefleX-side expected count is
        N_best * lambda_hat_i).

    return_counts : bool
        If True, return the annular RefleX and generated counts.

    use_N : bool
        If True, fit a multiplicative normalization N_best that scales
        the RefleX side relative to the generated side before computing
        the deviance (see "Optional multiplicative normalization"
        below), and subtract one degree of freedom for it. If False
        (default), N is fixed at 1 and no normalization is fitted --
        matching the original absolute, unnormalized comparison.

    return_N_best : bool
        If True, return the fitted (or fixed, when ``use_N=False``)
        normalization N_best.

    return_dof : bool
        If True, return the degrees of freedom used to calculate the
        reduced deviance.

    Returns
    -------
    float or tuple
        Reduced Poisson deviance, optionally followed by requested
        diagnostic arrays.

        The ordering is

            D_red,
            [profiles],
            [deviance contributions],
            [expected counts],
            [annular counts],
            [N_best],
            [dof]

        according to the requested options.

    Notes
    -----
    By default (``use_N=False``) this is an absolute comparison: no
    multiplicative normalization is fitted, so differences in both
    total photon yield and radial morphology contribute to the
    statistic. With ``use_N=True``, a single multiplicative
    normalization is fitted (see above), leaving only the residual
    morphology (and any yield mismatch not explained by a uniform
    rescaling) in the statistic.

    Unlike the previous chi-squared statistic:

    - by default (``bin_filter=0.0``), no positive minimum-count threshold
      is applied, but bins with zero RefleX/data counts are excluded;
    - no observed-count variance denominator is constructed;
    - no Fano factors are required;
    - by default, no model normalization is fitted.

    The asymptotic chi-squared interpretation can become inaccurate when
    counts are very sparse. In that regime, the deviance remains a useful
    discrepancy statistic, but formal p-values should preferably be
    calibrated using Monte Carlo / parametric bootstrap simulations.
    """

    # ------------------------------------------------------------------
    # 1. Validate inputs.
    # ------------------------------------------------------------------
    data = np.squeeze(np.asarray(data, dtype=np.float64))
    model = np.squeeze(np.asarray(model, dtype=np.float64))

    if data.ndim != 2:
        raise ValueError(
            f"data must be 2-D after squeezing; got shape {data.shape}."
        )

    if model.ndim != 2:
        raise ValueError(
            f"model must be 2-D after squeezing; got shape {model.shape}."
        )

    if data.shape != model.shape:
        raise ValueError(
            f"data shape {data.shape} must match model shape {model.shape}."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "data contains NaN or infinite values."
        )

    if not np.all(np.isfinite(model)):
        raise ValueError(
            "model contains NaN or infinite values."
        )

    if np.any(data < 0.0):
        raise ValueError(
            "data contains negative counts."
        )

    if np.any(model < 0.0):
        raise ValueError(
            "model contains negative counts."
        )

    if n_annuli <= 0:
        raise ValueError(
            "n_annuli must be a positive integer."
        )

    if pixel_scale_arcsec <= 0.0:
        raise ValueError(
            "pixel_scale_arcsec must be positive."
        )

    if bin_filter < 0.0 or not np.isfinite(bin_filter):
        raise ValueError(
            "bin_filter must be finite and non-negative."
        )

    height, width = data.shape

    # ------------------------------------------------------------------
    # 2. Construct radial bins.
    # ------------------------------------------------------------------
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    yy, xx = np.indices(
        (height, width)
    )

    radius_pixels = np.sqrt(
        (yy - cy) ** 2
        + (xx - cx) ** 2
    )

    edges = np.linspace(
        0.0,
        radius_pixels.max(),
        n_annuli + 1,
    )

    bins = np.digitize(
        radius_pixels.ravel(),
        edges,
        right=False,
    ) - 1

    bins = np.clip(
        bins,
        0,
        n_annuli - 1,
    )

    # ------------------------------------------------------------------
    # 3. Sum RefleX and generated counts within each annulus.
    # ------------------------------------------------------------------
    R = np.bincount(
        bins,
        weights=data.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    G = np.bincount(
        bins,
        weights=model.ravel(),
        minlength=n_annuli,
    ).astype(np.float64)

    N_pixels = np.bincount(
        bins,
        minlength=n_annuli,
    ).astype(np.float64)

    # ------------------------------------------------------------------
    # 4. Determine which annuli contain RefleX/data photons.
    #
    #    No minimum-count threshold is applied by default
    #    (bin_filter=0.0), but an annulus with R == 0 is excluded from
    #    both the deviance and dof, regardless of G.
    #    Passing a positive bin_filter additionally excludes annuli with
    #    fewer than that many RefleX (data) counts, mirroring bin_filter
    #    in radial_reduced_chi2_area2_bin_filtered (which uses S_data as
    #    the threshold quantity; here R plays that role).
    # ------------------------------------------------------------------
    valid_bins = (
        (N_pixels > 0.0)
        & (R >= bin_filter)
        & (R > 0.0)
        & np.isfinite(R)
        & np.isfinite(G)
    )

    valid_indices = np.flatnonzero(
        valid_bins
    )

    if len(valid_indices) == 0:
        extras = []

        if return_profiles:
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )

        if return_deviance_contributions:
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )

        if return_expected_counts:
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )

        if return_counts:
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )
            extras.append(
                np.zeros(
                    n_annuli,
                    dtype=np.float64,
                ).tolist()
            )

        if return_N_best:
            extras.append(float("nan"))

        if return_dof:
            extras.append(0)

        if extras:
            return (
                float("nan"),
                *extras,
            )

        return float("nan")

    # Restrict calculations to valid annuli.
    Rv = R[valid_bins]
    Gv = G[valid_bins]

    # ------------------------------------------------------------------
    # 5. Optional multiplicative normalization N_best.
    #
    #    R_i ~ Poisson(N * lambda_i), G_i ~ Poisson(lambda_i).
    #
    #    The closed-form maximum-likelihood estimate is
    #
    #        N_best = sum_i(R_i) / sum_i(G_i).
    #
    #    If use_N is False, N is fixed at 1 (no parameter fitted),
    #    which reproduces the original, unnormalized comparison.
    # ------------------------------------------------------------------
    if use_N:
        total_R = float(np.sum(Rv))
        total_G = float(np.sum(Gv))

        # N is not identifiable when the generated side has no counts
        # at all; fall back to the unnormalized case.
        N_best = (
            total_R / total_G
            if total_G > 0.0
            else 1.0
        )
    else:
        N_best = 1.0

    # ------------------------------------------------------------------
    # 6. Maximum-likelihood pooled expectation under H0.
    #
    #       lambda_hat_i = (R_i + G_i) / (N_best + 1)
    #
    #    which reduces to (R_i + G_i) / 2 when N_best = 1.
    # ------------------------------------------------------------------
    lambda_hat = (
        Rv + Gv
    ) / (N_best + 1.0)

    # ------------------------------------------------------------------
    # 7. Calculate two-sample Poisson deviance contributions.
    #
    #    For each annulus:
    #
    #        D_i = 2 [
    #            R_i log(R_i / (N_best * lambda_hat_i))
    #            +
    #            G_i log(G_i / lambda_hat_i)
    #        ].
    #
    #    Terms with zero observed counts are defined to be zero.
    #
    #    If R_i = G_i = 0, then lambda_hat_i = 0 and the full annulus
    #    contributes exactly zero.
    # ------------------------------------------------------------------
    deviance_contributions_valid = np.zeros(
        len(valid_indices),
        dtype=np.float64,
    )

    # RefleX term.
    mask_R = (
        Rv > 0.0
    )

    deviance_contributions_valid[mask_R] += (
        2.0
        * Rv[mask_R]
        * np.log(
            Rv[mask_R]
            / (N_best * lambda_hat[mask_R])
        )
    )

    # Generated-image term.
    mask_G = (
        Gv > 0.0
    )

    deviance_contributions_valid[mask_G] += (
        2.0
        * Gv[mask_G]
        * np.log(
            Gv[mask_G]
            / lambda_hat[mask_G]
        )
    )

    # Numerical roundoff can theoretically create values infinitesimally
    # below zero, although the deviance is mathematically non-negative.
    deviance_contributions_valid = np.maximum(
        deviance_contributions_valid,
        0.0,
    )

    deviance = float(
        np.sum(
            deviance_contributions_valid
        )
    )

    # ------------------------------------------------------------------
    # 8. Degrees of freedom and reduced deviance.
    #
    #    No normalization parameter is fitted by default, so nominally:
    #
    #        dof = K.
    #
    #    Here K is the number of spatial annuli with R > 0.
    #    When use_N is True, one degree of freedom is subtracted for
    #    the fitted normalization N_best.
    # ------------------------------------------------------------------
    dof = (
        len(valid_indices) - 1
        if use_N
        else len(valid_indices)
    )

    if dof <= 0:
        deviance_red = float("nan")
    else:
        deviance_red = (
            deviance / dof
        )

    # ------------------------------------------------------------------
    # 8. Construct surface-brightness profiles for optional output.
    #
    #    These profiles are NOT used to calculate the deviance.
    #    The deviance is calculated directly from annular counts.
    # ------------------------------------------------------------------
    area_per_pixel = (
        pixel_scale_arcsec**2
    )

    area_bin = (
        N_pixels
        * area_per_pixel
    )

    mu_data = np.zeros(
        n_annuli,
        dtype=np.float64,
    )

    mu_model = np.zeros(
        n_annuli,
        dtype=np.float64,
    )

    mu_data[valid_bins] = (
        R[valid_bins]
        / area_bin[valid_bins]
    )

    mu_model[valid_bins] = (
        G[valid_bins]
        / area_bin[valid_bins]
    )

    # ------------------------------------------------------------------
    # 9. Assemble optional outputs.
    # ------------------------------------------------------------------
    extras = []

    if return_profiles:
        extras.append(
            mu_data.tolist()
        )
        extras.append(
            mu_model.tolist()
        )

    if return_deviance_contributions:
        full_contributions = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        full_contributions[valid_bins] = (
            deviance_contributions_valid
        )

        extras.append(
            full_contributions.tolist()
        )

    if return_expected_counts:
        full_expected = np.zeros(
            n_annuli,
            dtype=np.float64,
        )

        full_expected[valid_bins] = (
            lambda_hat
        )

        extras.append(
            full_expected.tolist()
        )

    if return_counts:
        extras.append(
            R.tolist()
        )
        extras.append(
            G.tolist()
        )

    if return_N_best:
        extras.append(
            float(N_best)
        )

    if return_dof:
        extras.append(
            int(dof)
        )

    if extras:
        return (
            float(deviance_red),
            *extras,
        )

    return float(deviance_red)


def _call_poisson_deviance_pair(
    mat_a,
    mat_b,
    *,
    n_annuli,
    pixel_scale_arcsec,
    need_contributions,
    bin_filter=0.0,
    use_N=False,
):
    """Return the Poisson-deviance metric in the comparison-helper format."""
    result = radial_reduced_poisson_deviance_abs(
        mat_a,
        mat_b,
        n_annuli=n_annuli,
        pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter,
        return_profiles=True,
        return_deviance_contributions=need_contributions,
        use_N=use_N,
        return_N_best=True,
        return_dof=True,
    )

    val, mu_a, mu_b, *extra = result
    if need_contributions:
        contributions, N_best, dof = extra
    else:
        N_best, dof = extra
        contributions = np.zeros(n_annuli, dtype=np.float64)

    # ``dof`` comes from the metric itself so the reported value and the
    # divisor used to calculate ``val`` cannot drift apart.
    # Matches the chi2 helper's (errors, mu_data, mu_model, N_best) shape
    # so the shared profile-plotting code in _compare_matrices can treat
    # both metrics uniformly.
    helper_data = (
        np.asarray(contributions, dtype=np.float64),
        np.asarray(mu_a, dtype=np.float64),
        np.asarray(mu_b, dtype=np.float64),
        float(N_best),
    )
    summary_str = (
        ("N-normalized" if use_N else "absolute")
        + (
            f"; bin_filter={bin_filter:g}"
            if bin_filter > 0.0
            else "; data counts > 0"
        )
        + (
            f"; fitted per pair, N={float(N_best):.4g}"
            if use_N
            else "; no fitted normalization"
        )
    )
    return (
        float(val),
        np.asarray(contributions, dtype=np.float64),
        int(dof),
        summary_str,
        helper_data,
    )

def compare_npz_by_params(
    x: float,
    nH: float,
    npz_paths,
    *,
    labels=None,
    decimals: int = 6,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
):
    """
    Like compare_npy_matrices, but pulls the (x, nH) matrix directly out of
    saved .npz archives (as produced by save_npy_to_npz, keyed by "xs"/"nHs"/
    "mats" - the same archives error_heatmap_between_npz reads) instead of
    loading individual .npy files.

    x, nH : float
        Parameter values to look up in every archive.
    npz_paths : sequence of str/Path
        One .npz archive per matrix to compare. As with compare_npy_matrices,
        the LAST archive's matrix is treated as the reference "data".
    labels : list of str, optional
        Display labels. Defaults to each archive's filename stem.
    decimals : int
        Rounding applied to each archive's xs/nHs before matching against
        x, nH (guards against float rounding drift between archives).
    """
    if len(npz_paths) < 1:
        raise ValueError("Provide at least 1 .npz file path.")

    n = len(npz_paths)
    x_r = round(float(x), decimals)
    nH_r = round(float(nH), decimals)
    print(f"x={x} -> matched x={x_r}")
    print(f"nH={nH} -> matched nH={nH_r}")

    matrices = []
    model_variances = []
    fallback_labels = []
    for p in npz_paths:
        with np.load(p) as data:
            xs_arr = np.round(data["xs"].astype(float), decimals)
            nHs_arr = np.round(data["nHs"].astype(float), decimals)
            match = np.where((xs_arr == x_r) & (nHs_arr == nH_r))[0]
            if len(match) == 0:
                raise ValueError(
                    f"No matrix for x={x} (rounded {x_r}), nH={nH} (rounded {nH_r}) found in {p}"
                )
            match_index = int(match[0])
            arr = np.squeeze(np.asarray(data["mats"][match_index], dtype=np.float64))
            variance = (
                np.squeeze(np.asarray(data["Vs"][match_index], dtype=np.float64))
                if "Vs" in data.files
                else None
            )
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected 2D matrix after squeeze, got {arr.shape}")
        matrices.append(arr)
        model_variances.append(variance)
        fallback_labels.append(Path(p).stem)

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < n:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[i] for i in range(n)]
    x = str(x).replace(".", "_")
    nH = str(nH).replace("+", "_").replace(".", "_")
    png_filename = f"x_{x}_nH_{nH}.png"
    print(png_filename)
    return _compare_matrices(
        matrices, labels,
        model_variances=model_variances,
        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter, Fano_Data=Fano_Data, Fano_Model=Fano_Model,
        cmaps=cmaps, a=a, center=center, figsize=figsize, font_size=font_size,
        x_lim=x_lim, y_lim=y_lim, show_bin_contributions=show_bin_contributions,
        show_mean_bin_contribution_plot=show_mean_bin_contribution_plot,
        show_chi2_label=False,
    )


def compare_npz_by_params_poisson_deviance(
    x: float,
    nH: float,
    npz_paths,
    *,
    labels=None,
    decimals: int = 6,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 0.0,
    use_N: bool = False,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
):
    """Compare matching NPZ matrices with radial Poisson deviance.

    This is the Poisson-deviance counterpart of ``compare_npz_by_params``.
    It uses ``radial_reduced_poisson_deviance_abs`` as the pairwise metric,
    so it does not use variance arrays or Fano factors. The last archive
    remains the reference matrix for plotting and reference-summary
    purposes.

    bin_filter : float
        Forwarded to ``radial_reduced_poisson_deviance_abs``. Only
        annuli with at least ``bin_filter`` RefleX (data-side) counts are
        included in the deviance sum and dof. The default, 0.0, matches
        the original minimum-count behavior: no positive threshold is
        imposed. Annuli with zero RefleX/data counts are nevertheless
        excluded from the deviance sum and dof, regardless of the model
        count. Passing a positive value (e.g. 10.0, as in
        ``compare_npz_by_params``) additionally excludes low-count annuli,
        mirroring that function's ``bin_filter``.

    use_N : bool
        Forwarded to ``radial_reduced_poisson_deviance_abs``. If False
        (default), no model normalization is fitted -- the original
        absolute comparison. If True, a multiplicative normalization is
        fitted per pair (one degree of freedom subtracted accordingly).

    The returned mapping has the same shape as ``compare_npz_by_params``:
    ``{(i, j): (value_ij, value_ji)}``. With ``use_N=False`` and
    ``bin_filter=0.0`` (the defaults), the metric is directional because
    only annuli with positive counts on the data side are included. Thus
    the two values in each tuple can differ. ``use_N=True`` and a positive
    ``bin_filter`` introduce further direction dependence.
    """
    if len(npz_paths) < 1:
        raise ValueError("Provide at least 1 .npz file path.")

    n = len(npz_paths)
    x_r = round(float(x), decimals)
    nH_r = round(float(nH), decimals)
    print(f"x={x} -> matched x={x_r}")
    print(f"nH={nH} -> matched nH={nH_r}")

    matrices = []
    fallback_labels = []
    for p in npz_paths:
        with np.load(p) as data:
            xs_arr = np.round(data["xs"].astype(float), decimals)
            nHs_arr = np.round(data["nHs"].astype(float), decimals)
            match = np.where((xs_arr == x_r) & (nHs_arr == nH_r))[0]
            if len(match) == 0:
                raise ValueError(
                    f"No matrix for x={x} (rounded {x_r}), nH={nH} "
                    f"(rounded {nH_r}) found in {p}"
                )
            arr = np.squeeze(
                np.asarray(data["mats"][int(match[0])], dtype=np.float64)
            )
        if arr.ndim != 2:
            raise ValueError(
                f"{p}: expected 2D matrix after squeeze, got {arr.shape}"
            )
        matrices.append(arr)
        fallback_labels.append(Path(p).stem)

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < n:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[i] for i in range(n)]

    x_tag = str(x).replace(".", "_")
    nH_tag = str(nH).replace("+", "_").replace(".", "_")
    print(f"x_{x_tag}_nH_{nH_tag}_poisson_deviance.png")

    return _compare_matrices(
        matrices,
        labels,
        metric="poisson_deviance_abs",
        n_annuli=n_annuli,
        pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter,
        use_N=use_N,
        cmaps=cmaps,
        a=a,
        center=center,
        figsize=figsize,
        font_size=font_size,
        x_lim=x_lim,
        y_lim=y_lim,
        show_bin_contributions=show_bin_contributions,
        show_mean_bin_contribution_plot=show_mean_bin_contribution_plot,
        show_chi2_label=False,
    )


def compare_npz_by_param_list(
    xs: list,
    nHs: list,
    npz_path,
    *,
    labels=None,
    decimals: int = 6,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
    show_matrices: bool = True,
    red_gradient: bool = True,
):
    """
    Like compare_npz_by_params, but pulls MULTIPLE (x, nH) matrices out of a
    SINGLE .npz archive, instead of one matrix each from multiple archives.

    xs, nHs : sequence of float
        Parallel lists of parameter values to look up in the archive; the
        i-th matrix compared is the one matching (xs[i], nHs[i]). As with
        compare_npz_by_params, the LAST pair is treated as the reference
        "data".
    npz_path : str/Path
        Single .npz archive (as produced by save_npy_to_npz, keyed by
        "xs"/"nHs"/"mats") to pull every matrix from.
    labels : list of str, optional
        Display labels. Defaults to "x=.., nH=.." per matched pair.
    decimals : int
        Rounding applied to the archive's xs/nHs before matching against
        each (x, nH) pair (guards against float rounding drift).
    show_matrices : bool
        Whether to display the top-row matrix images above the radial
        profile panel. When False, only the profile panel is shown.
    red_gradient : bool
        Colour the (non-reference) profile lines red instead of the default
        colourblind-safe palette, with intensity deepening from the first
        line to the last in xs/nHs (pale red -> saturated red). The
        reference (last pair) still plots as black.
    """
    if len(xs) != len(nHs):
        raise ValueError("xs and nHs must have the same length.")
    if len(xs) < 1:
        raise ValueError("Provide at least 1 (x, nH) pair.")

    n = len(xs)
    matrices = []
    model_variances = []
    fallback_labels = []
    with np.load(npz_path) as data:
        xs_arr = np.round(data["xs"].astype(float), decimals)
        nHs_arr = np.round(data["nHs"].astype(float), decimals)
        mats = data["mats"]
        has_variance = "Vs" in data.files
        vs = data["Vs"] if has_variance else None

        for x, nH in zip(xs, nHs):
            x_r = round(float(x), decimals)
            nH_r = round(float(nH), decimals)
            print(f"x={x} -> matched x={x_r}")
            print(f"nH={nH} -> matched nH={nH_r}")
            match = np.where((xs_arr == x_r) & (nHs_arr == nH_r))[0]
            if len(match) == 0:
                raise ValueError(
                    f"No matrix for x={x} (rounded {x_r}), nH={nH} (rounded {nH_r}) found in {npz_path}"
                )
            match_index = int(match[0])
            arr = np.squeeze(np.asarray(mats[match_index], dtype=np.float64))
            variance = (
                np.squeeze(np.asarray(vs[match_index], dtype=np.float64))
                if has_variance
                else None
            )
            if arr.ndim != 2:
                raise ValueError(f"{npz_path}: expected 2D matrix after squeeze, got {arr.shape}")
            matrices.append(arr)
            model_variances.append(variance)
            fallback_labels.append(f"x={x:g}, nH={nH:.3g}")

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < n:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[i] for i in range(n)]

    param_tag = "_".join(
        f"x{str(x).replace('.', '_')}_nH{str(nH).replace('+', '_').replace('.', '_')}"
        for x, nH in zip(xs, nHs)
    )
    png_filename = f"{Path(npz_path).stem}_{param_tag}.png"
    print(png_filename)
    return _compare_matrices(
        matrices, labels,
        model_variances=model_variances,
        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter, Fano_Data=Fano_Data, Fano_Model=Fano_Model,
        cmaps=cmaps, a=a, center=center, figsize=figsize, font_size=font_size,
        x_lim=x_lim, y_lim=y_lim, show_bin_contributions=show_bin_contributions,
        show_mean_bin_contribution_plot=show_mean_bin_contribution_plot,
        show_chi2_label=False, show_matrices=show_matrices,
    )


def compare_npz_by_param_pairs(
    npz_paths: list,
    xs: list,
    nHs: list,
    *,
    labels=None,
    decimals: int = 6,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
):
    """
    Like compare_npz_by_param_list, but each (x, nH) pair is pulled from its
    OWN .npz archive instead of a single shared archive.

    npz_paths, xs, nHs : parallel sequences
        The i-th matrix compared is matched (xs[i], nHs[i]) inside
        npz_paths[i]. As with compare_npz_by_params, the LAST triple is
        treated as the reference "data".
    labels : list of str, optional
        Display labels. Defaults to "<archive stem>: x=.., nH=.." per pair.
    decimals : int
        Rounding applied to each archive's xs/nHs before matching against
        the corresponding (x, nH) pair (guards against float rounding drift
        between archives).
    """
    if len(npz_paths) != len(xs) or len(xs) != len(nHs):
        raise ValueError("npz_paths, xs, and nHs must all have the same length.")
    if len(npz_paths) < 1:
        raise ValueError("Provide at least 1 (npz_path, x, nH) triple.")

    n = len(npz_paths)
    matrices = []
    model_variances = []
    fallback_labels = []
    for p, x, nH in zip(npz_paths, xs, nHs):
        x_r = round(float(x), decimals)
        nH_r = round(float(nH), decimals)
        print(f"x={x} -> matched x={x_r}")
        print(f"nH={nH} -> matched nH={nH_r}")
        with np.load(p) as data:
            xs_arr = np.round(data["xs"].astype(float), decimals)
            nHs_arr = np.round(data["nHs"].astype(float), decimals)
            match = np.where((xs_arr == x_r) & (nHs_arr == nH_r))[0]
            if len(match) == 0:
                raise ValueError(
                    f"No matrix for x={x} (rounded {x_r}), nH={nH} (rounded {nH_r}) found in {p}"
                )
            match_index = int(match[0])
            arr = np.squeeze(np.asarray(data["mats"][match_index], dtype=np.float64))
            variance = (
                np.squeeze(np.asarray(data["Vs"][match_index], dtype=np.float64))
                if "Vs" in data.files
                else None
            )
        if arr.ndim != 2:
            raise ValueError(f"{p}: expected 2D matrix after squeeze, got {arr.shape}")
        matrices.append(arr)
        model_variances.append(variance)
        fallback_labels.append(f"{Path(p).stem}: x={x:g}, nH={nH:.3g}")

    base_labels = fallback_labels if labels is None else list(labels)
    if len(base_labels) == 0:
        base_labels = fallback_labels
    if len(base_labels) < n:
        base_labels = base_labels + fallback_labels[len(base_labels):]
    labels = [base_labels[i] for i in range(n)]

    param_tag = "_".join(
        f"{Path(p).stem}_x{str(x).replace('.', '_')}_nH{str(nH).replace('+', '_').replace('.', '_')}"
        for p, x, nH in zip(npz_paths, xs, nHs)
    )
    png_filename = f"{param_tag}.png"
    print(png_filename)
    return _compare_matrices(
        matrices, labels,
        model_variances=model_variances,
        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
        bin_filter=bin_filter, Fano_Data=Fano_Data, Fano_Model=Fano_Model,
        cmaps=cmaps, a=a, center=center, figsize=figsize, font_size=font_size,
        x_lim=x_lim, y_lim=y_lim, show_bin_contributions=show_bin_contributions,
        show_mean_bin_contribution_plot=show_mean_bin_contribution_plot,
        show_chi2_label=False,
    )


import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# Colourblind-safe palette (Wong/Okabe-Ito-adjacent); reference is black.
_CB_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"]
_LINESTYLES = ["-", "--", "-.", ":"]

_MNRAS_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _compare_matrices(
    matrices,
    labels,
    *,
    metric: str = "chi2",
    model_variances=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    use_N: bool = False,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 9,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
    show_chi2_label: bool = True,
    show_matrices: bool = True,      # display the top-row matrix images
    red_gradient: bool = False,      # colour model lines red, deepening in list order
    # ---- NEW styling parameters ----
    journal_style: bool = True,      # apply MNRAS-like rc styling locally
    ref_as_points: bool = True,      # reference (last matrix) as black points
    log_x: bool = False,             # log-scale radius axis
    scalebar_arcsec: float | None = 500.0,  # scale bar on image panels; None disables
    panel_labels: bool = True,       # (a), (b), (c) corner labels
    save_path=None,                  # save figure (pdf recommended)
    show: bool = True,
):
    """
    Shared implementation behind the NPZ/matrix comparison functions:
    display already-loaded 2D matrices in log scale with a shared radial
    surface-brightness profile panel below, and compute a pairwise
    goodness-of-fit metric for every pair.

    ``metric`` may be ``"chi2"`` or ``"poisson_deviance_abs"``. The latter
    compares annular counts without Fano factors or model variances. It is
    directional because only annuli with positive data-side counts are
    included (and a positive ``bin_filter`` further thresholds that side).
    ``use_N`` is forwarded to ``radial_reduced_poisson_deviance_abs`` for
    the ``poisson_deviance_abs`` metric only (it has no effect on
    ``metric="chi2"``, which fits its own normalization independently);
    when True, a multiplicative normalization is fitted per pair and per
    direction, so the two ``results[(i, j)]`` values generally differ.
    ``bin_filter`` is shared with the chi2 metric (``S_data > bin_filter``
    there) but defaults to 0.0 here -- unlike chi2's 10.0 default, this
    keeps ``poisson_deviance_abs`` unfiltered unless a threshold is
    explicitly requested.

    Styling follows MNRAS figure conventions when journal_style=True:
    serif/STIX fonts, inward ticks on all four sides with minor ticks,
    no grid, frameless legend, reference profile drawn as black data
    points with error bars and models as colourblind-safe lines, a
    shared logarithmic colour scale across image panels with a scale
    bar and (a)/(b)/(c) panel labels.
    """
    from .plotting import normalize_and_log_scale
    from matplotlib.ticker import MaxNLocator  # type: ignore[import]

    if metric not in {"chi2", "poisson_deviance_abs"}:
        raise ValueError(
            "metric must be 'chi2' or 'poisson_deviance_abs'."
        )

    n = len(matrices)
    if model_variances is None:
        model_variances = [None] * n
    elif len(model_variances) != n:
        raise ValueError(
            "model_variances must have the same length as matrices."
        )

    validated_variances = []
    for index, (matrix, variance) in enumerate(zip(matrices, model_variances)):
        if variance is None:
            validated_variances.append(None)
            continue

        variance = np.squeeze(np.asarray(variance, dtype=np.float64))
        if variance.shape != matrix.shape:
            raise ValueError(
                f"Variance shape mismatch at index {index}: "
                f"V={variance.shape}, matrix={matrix.shape}."
            )
        if not np.all(np.isfinite(variance)):
            raise ValueError(
                f"Variance matrix at index {index} contains NaN or infinite values."
            )
        if np.any(variance < 0.0):
            raise ValueError(
                f"Variance matrix at index {index} contains negative values."
            )
        validated_variances.append(variance)

    model_variances = validated_variances

    # Image panels: one perceptually-uniform map shared by all panels is
    # the journal-preferred choice; per-panel maps remain available by
    # passing an explicit cmaps list.
    if cmaps is None:
        cmaps = ["magma"] * n
    else:
        base = list(cmaps) or ["magma"]
        cmaps = [base[i % len(base)] for i in range(n)]

    # --- Shared colour range ---
    shared_min = float(min(m.min() for m in matrices))
    shared_max = float(max(m.max() for m in matrices))

    rc = _MNRAS_RC if journal_style else {}

    with plt.rc_context(rc):
        # --- Figure layout ---
        if figsize is None:
            # Sized for a two-column (page-width) MNRAS figure; use
            # figsize=(3.32, ...) explicitly for single-column.
            figsize = (2.3 * n, 4.8) if show_matrices else (2.3 * n, 3.4)
        fig = plt.figure(figsize=figsize)

        if show_matrices:
            gs = fig.add_gridspec(2, n, height_ratios=[1, 1.35], hspace=0.12)
            img_axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
            ax_prof = fig.add_subplot(gs[1, :])

            # --- Top row: log-scaled images ---
            for k, (ax, arr, label, cmap) in enumerate(
                zip(img_axes, matrices, labels, cmaps)
            ):
                scaled, _, _ = normalize_and_log_scale(
                    arr,
                    a,
                    data_min=shared_min,
                    data_max=shared_max,
                )
                ax.imshow(
                    scaled,
                    cmap=cmap,
                    origin="lower",
                    interpolation="nearest",   # show the 48x48 pixels as pixels
                )
                ax.set_title(label, fontsize=font_size)
                ax.set_xticks([])
                ax.set_yticks([])

                if panel_labels:
                    ax.text(
                        0.04, 0.96, f"({chr(ord('a') + k)})",
                        transform=ax.transAxes,
                        ha="left", va="top",
                        fontsize=font_size, color="white",
                    )

                # Scale bar on the last (reference) panel only, to avoid clutter.
                if scalebar_arcsec is not None and k == n - 1:
                    bar_pix = scalebar_arcsec / pixel_scale_arcsec
                    H_img, W_img = arr.shape
                    x0 = 0.95 * W_img - bar_pix
                    y0 = 0.06 * H_img
                    ax.plot([x0, x0 + bar_pix], [y0, y0],
                            color="white", lw=1.5, solid_capstyle="butt")
                    ax.text(x0 + bar_pix / 2.0, y0 + 0.03 * H_img,
                            rf"${scalebar_arcsec:.0f}''$",
                            ha="center", va="bottom",
                            fontsize=font_size - 1, color="white")
        else:
            gs = fig.add_gridspec(1, 1)
            ax_prof = fig.add_subplot(gs[0, 0])

        # --- Radial geometry (last matrix as template) ---
        template = matrices[-1]
        H, W = template.shape
        yy, xx = np.indices((H, W))
        if center == "peak":
            cy, cx = np.unravel_index(np.nanargmax(template), template.shape)
            cy, cx = float(cy), float(cx)
        else:
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = rr.max()
        edges_pix = np.linspace(0.0, r_max, n_annuli + 1)
        bins = np.clip(np.digitize(rr.ravel(), edges_pix) - 1, 0, n_annuli - 1)
        bin_centers_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
        bin_centers_arc = bin_centers_pix * pixel_scale_arcsec

        N_pixels = np.bincount(bins, minlength=n_annuli).astype(np.float64)
        area_per_pix = pixel_scale_arcsec ** 2
        area_bin = N_pixels * area_per_pix

        def _radial_sum(arr):
            return np.bincount(
                bins, weights=np.nan_to_num(arr.ravel()), minlength=n_annuli
            )

        sums = [_radial_sum(m) for m in matrices]
        variance_sums = [
            _radial_sum(variance) if variance is not None else sums[index]
            for index, variance in enumerate(model_variances)
        ]
        sb_profiles = [np.zeros(n_annuli, dtype=np.float64) for _ in matrices]
        for sb, s in zip(sb_profiles, sums):
            valid_sb = (
                N_pixels > 0.0
                if metric == "poisson_deviance_abs"
                else s > bin_filter
            )
            sb[valid_sb] = s[valid_sb] / area_bin[valid_sb]

        # --- Pairwise metric computation (unchanged) ---
        results = {}

        ref_idx = n - 1
        ref_sum = sums[ref_idx]
        valid_ref = (
            (N_pixels > 0.0) & (ref_sum >= bin_filter)
            if metric == "poisson_deviance_abs"
            else ref_sum > bin_filter
        )

        err_ref = np.zeros(n_annuli, dtype=np.float64)
        err_ref[valid_ref] = (
            np.sqrt(
                ref_sum[valid_ref]
                if metric == "poisson_deviance_abs"
                else Fano_Data * ref_sum[valid_ref]
            )
            / area_bin[valid_ref]
        )

        helper_profiles = {}
        pairwise_red_contributions = []
        pairwise_mean_values = []
        ref_metric_values = []

        is_poisson = metric == "poisson_deviance_abs"
        metric_label = "D_red" if is_poisson else "chi2_red"
        metric_name = (
            ("N-normalized" if use_N else "absolute")
            + " Poisson deviance"
            if is_poisson
            else "chi2"
        )
        print(f"\nPairwise {metric_label} ({metric_name}):")
        if is_poisson:
            print(
                f"  n_annuli={n_annuli}, pixel_scale={pixel_scale_arcsec}\"; "
                + (
                    "fitted multiplicative normalization N per pair "
                    "and direction; "
                    if use_N
                    else "no normalization; "
                )
                + (
                    f"bin_filter={bin_filter:g}"
                    if bin_filter > 0.0
                    else "data counts > 0"
                )
            )
        else:
            print(
                f"  n_annuli={n_annuli}, bin_filter={bin_filter}, "
                f"Fano Data={Fano_Data}, Fano Model={Fano_Model}, "
                f"pixel_scale={pixel_scale_arcsec}\""
            )
        print("-" * 55)
        need_pairwise_contributions = (
            show_bin_contributions or show_mean_bin_contribution_plot
        )
        if n == 1:
            print(f"  Single input matrix provided; pairwise {metric_label} skipped.")
        else:
            for (i, j) in combinations(range(n), 2):
                if is_poisson:
                    (
                        val_ij,
                        contrib_ij,
                        dof_ij,
                        summary_str_ij,
                        hdata_ij,
                    ) = _call_poisson_deviance_pair(
                        matrices[i],
                        matrices[j],
                        n_annuli=n_annuli,
                        pixel_scale_arcsec=pixel_scale_arcsec,
                        bin_filter=bin_filter,
                        need_contributions=need_pairwise_contributions,
                        use_N=use_N,
                    )

                    # The metric is directional even without normalization
                    # or a positive bin_filter: valid annuli are selected
                    # from the data side only (data counts > 0).
                    (
                        val_ji,
                        contrib_ji,
                        dof_ji,
                        summary_str_ji,
                        hdata_ji,
                    ) = _call_poisson_deviance_pair(
                        matrices[j],
                        matrices[i],
                        n_annuli=n_annuli,
                        pixel_scale_arcsec=pixel_scale_arcsec,
                        bin_filter=bin_filter,
                        need_contributions=need_pairwise_contributions,
                        use_N=use_N,
                    )
                else:
                    val_ij, contrib_ij, dof_ij, summary_str_ij, hdata_ij = _call_chi2_pair(
                        matrices[i], matrices[j],
                        V=model_variances[j],
                        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                        bin_filter=bin_filter,
                        need_contributions=need_pairwise_contributions,
                        Fano_Data=Fano_Data, Fano_Model=Fano_Model,
                    )
                    val_ji, contrib_ji, dof_ji, summary_str_ji, hdata_ji = _call_chi2_pair(
                        matrices[j], matrices[i],
                        V=model_variances[i],
                        n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                        bin_filter=bin_filter,
                        need_contributions=need_pairwise_contributions,
                        Fano_Data=Fano_Data, Fano_Model=Fano_Model,
                    )
                helper_profiles[(i, j)] = hdata_ij
                helper_profiles[(j, i)] = hdata_ji

                pair_mean = 0.5 * (val_ij + val_ji)
                pairwise_mean_values.append(pair_mean)
                if j == ref_idx:
                    ref_metric_values.append(val_ji)
                elif i == ref_idx:
                    ref_metric_values.append(val_ij)
                results[(i, j)] = (val_ij, val_ji)
                print(f"  {labels[i]} vs {labels[j]}")
                dof_report_ij = f"dof={dof_ij}   " if is_poisson else ""
                dof_report_ji = f"dof={dof_ji}   " if is_poisson else ""
                print(
                    f"    {metric_label} ({labels[i]} as data) = {val_ij:.4f}   "
                    f"{dof_report_ij}{summary_str_ij}"
                )
                print(
                    f"    {metric_label} ({labels[j]} as data) = {val_ji:.4f}   "
                    f"{dof_report_ji}{summary_str_ji}"
                )

                if need_pairwise_contributions:
                    red_contrib_ij = (
                        contrib_ij / float(dof_ij)
                        if dof_ij > 0
                        else np.full_like(contrib_ij, np.nan, dtype=np.float64)
                    )
                    red_contrib_ji = (
                        contrib_ji / float(dof_ji)
                        if dof_ji > 0
                        else np.full_like(contrib_ji, np.nan, dtype=np.float64)
                    )
                    pairwise_red_contributions.append(red_contrib_ij)
                    pairwise_red_contributions.append(red_contrib_ji)

                    if show_bin_contributions:
                        label_i = str(labels[i])
                        label_j = str(labels[j])
                        print(f"    per-annulus reduced {metric_label} contributions:")
                        print(
                            "      bin | radius[arcsec] | area[arcsec^2] | "
                            f"counts({label_i}) | counts({label_j}) | "
                            f"{label_i}->{label_j} | {label_j}->{label_i}"
                        )
                        for b in range(n_annuli):
                            print(
                                f"      {b+1:>3d} | {bin_centers_arc[b]:>13.3f} | "
                                f"{area_bin[b]:>13.3f} | "
                                f"{float(sums[i][b]):>12.3f} | "
                                f"{float(sums[j][b]):>13.3f} | "
                                f"{red_contrib_ij[b]:>11.4e} | "
                                f"{red_contrib_ji[b]:>11.4e}"
                            )

            if pairwise_mean_values:
                mean_all = float(np.mean(pairwise_mean_values))
                std_all = float(np.std(pairwise_mean_values))
                print(f"  Mean {metric_label} (all pairs, both directions) = {mean_all:.4f} ± {std_all:.4f}")
            if ref_metric_values:
                mean_ref = float(np.mean(ref_metric_values))
                std_ref = float(np.std(ref_metric_values))
                relation = (
                    "absolute comparison with reference"
                    if is_poisson
                    else "model fit to reference"
                )
                print(
                    f"  Mean {metric_label} ({relation} "
                    f"'{labels[ref_idx]}') = {mean_ref:.4f} ± {std_ref:.4f}"
                )
            if results:
                print(f"  Pairwise {metric_label} summary (mean of both directions):")
                for (i, j), (v_ij, v_ji) in results.items():
                    print(f"    {labels[i]} x {labels[j]} = {0.5 * (v_ij + v_ji):.4f}")
        print("-" * 55)

        # --- Radial profile panel ---
        x_vals = bin_centers_arc
        valid_bins_plot = valid_ref
        valid_bins_by_profile = [valid_ref.copy() for _ in range(n)]
        profiles_to_plot = [np.asarray(p, dtype=np.float64) for p in sb_profiles]
        ref_errors_plot = np.asarray(err_ref, dtype=np.float64)
        errors_plot = [np.zeros(n_annuli, dtype=np.float64) for _ in range(n)]
        errors_plot[ref_idx] = ref_errors_plot

        if is_poisson:
            ref_filter_mask = (
                (N_pixels > 0.0)
                & (ref_sum >= bin_filter)
                & (ref_sum > 0.0)
            )

            if n >= 2:
                # Match the metric's data-side validity rule. The last
                # matrix is the RefleX/data side for the displayed radial
                # comparisons, so only its positive-count annuli appear.
                for idx in range(n):
                    if idx == ref_idx:
                        continue
                    valid_bins_by_profile[idx] = ref_filter_mask.copy()
                valid_bins_by_profile[ref_idx] = ref_filter_mask.copy()
                valid_bins_plot = ref_filter_mask
            else:
                valid_bins_plot = ref_filter_mask
                valid_bins_by_profile[ref_idx] = valid_bins_plot

            for idx, s in enumerate(sums):
                valid = valid_bins_by_profile[idx]
                errors_plot[idx][valid] = (
                    np.sqrt(s[valid]) / area_bin[valid]
                )

            # When a normalization was fitted (use_N=True), rescale every
            # non-reference profile by its fitted N_best so the plotted
            # curves match what the reported D_red actually compared --
            # otherwise the plot still shows raw, un-rescaled profiles
            # even though the metric accounted for the flux mismatch.
            if use_N and n >= 2:
                ref_profile = None
                display_profiles = [None] * n

                for idx in range(n - 1):
                    helper_key = (ref_idx, idx)
                    if helper_key not in helper_profiles:
                        continue

                    _, mu_ref, mu_other, n_best = helper_profiles[helper_key]
                    if ref_profile is None:
                        ref_profile = mu_ref

                    model_profile = np.zeros_like(mu_ref, dtype=np.float64)
                    pair_plot_mask = valid_bins_by_profile[idx]
                    model_profile[pair_plot_mask] = (
                        n_best * mu_other[pair_plot_mask]
                    )
                    display_profiles[idx] = model_profile

                    errors_plot[idx] = errors_plot[idx] * n_best

                if ref_profile is not None:
                    display_profiles[ref_idx] = np.asarray(
                        ref_profile, dtype=np.float64
                    )
                    profiles_to_plot = [
                        np.asarray(p, dtype=np.float64)
                        if p is not None
                        else np.zeros_like(ref_profile, dtype=np.float64)
                        for p in display_profiles
                    ]
        elif n >= 2:
            ref_profile = None
            display_profiles = [None] * n

            for idx in range(n - 1):
                helper_key = (ref_idx, idx)
                if helper_key not in helper_profiles:
                    continue

                _, mu_ref, mu_other, n_best = helper_profiles[helper_key]
                if ref_profile is None:
                    ref_profile = mu_ref

                model_profile = np.zeros_like(mu_ref, dtype=np.float64)
                valid_bins = ref_sum > bin_filter
                model_profile[valid_bins] = n_best * mu_other[valid_bins]
                display_profiles[idx] = model_profile

                model_error = np.zeros_like(mu_ref, dtype=np.float64)
                model_error[valid_bins] = (
                    n_best
                    * np.sqrt(Fano_Model * variance_sums[idx][valid_bins])
                    / area_bin[valid_bins]
                )
                errors_plot[idx] = model_error

            if ref_profile is not None:
                valid_bins = ref_sum > bin_filter
                display_profiles[ref_idx] = np.asarray(ref_profile, dtype=np.float64)
                valid_bins_plot = valid_bins
                profiles_to_plot = [
                    np.asarray(p, dtype=np.float64)
                    if p is not None
                    else np.zeros_like(ref_profile, dtype=np.float64)
                    for p in display_profiles
                ]

        # Reference (ground truth) drawn as black data points; models as
        # colourblind-safe lines with distinct linestyles. Legend ordered
        # reference-first.
        handles, handle_labels = [], []

        if ref_as_points:
            ref_plot_mask = valid_bins_by_profile[ref_idx]
            h = ax_prof.errorbar(
                x_vals[ref_plot_mask],
                profiles_to_plot[ref_idx][ref_plot_mask],
                yerr=errors_plot[ref_idx][ref_plot_mask],
                fmt="o", ms=3.0, color="black",
                elinewidth=0.8, capsize=0, zorder=5,
            )
            handles.append(h)
            handle_labels.append(labels[ref_idx])
        model_indices = [k for k in range(n) if k != ref_idx or not ref_as_points]

        # Red gradient: first line in the list is a pale red, each
        # subsequent line deepens toward a saturated red.
        if red_gradient:
            n_lines = len(model_indices)
            shades = (
                np.linspace(0.35, 0.95, n_lines) if n_lines > 1
                else np.array([0.85])
            )
            red_colors = [plt.cm.Reds(s) for s in shades]

        for slot, k in enumerate(model_indices):
            if k == ref_idx:
                color = "black"
            elif red_gradient:
                color = red_colors[slot]
            else:
                color = _CB_COLORS[slot % len(_CB_COLORS)]
            ls = _LINESTYLES[slot % len(_LINESTYLES)]
            profile_plot_mask = valid_bins_by_profile[k]
            xv = x_vals[profile_plot_mask]
            (h,) = ax_prof.plot(
                xv, profiles_to_plot[k][profile_plot_mask],
                ls, color=color, lw=1.2, label=labels[k],
            )
            ax_prof.errorbar(
                xv,
                profiles_to_plot[k][profile_plot_mask],
                yerr=errors_plot[k][profile_plot_mask],
                fmt="none", ecolor=color,
                elinewidth=0.8, alpha=0.8, capsize=0,
            )
            handles.append(h)
            handle_labels.append(labels[k])

        ax_prof.set_xlabel(r"Radius [arcsec]", fontsize=font_size)
        ax_prof.set_ylabel(
            r"Surface brightness [photons arcsec$^{-2}$]",
            fontsize=font_size,
        )
        ax_prof.set_yscale("log")
        if log_x:
            ax_prof.set_xscale("log")
        else:
            ax_prof.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_prof.tick_params(axis="both", labelsize=font_size)
        ax_prof.legend(
            handles, handle_labels, fontsize=font_size, frameon=False,
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=min(len(handles), 4),
        )

        if x_lim is not None:
            ax_prof.set_xlim(x_lim)
        if y_lim is not None:
            ax_prof.set_ylim(y_lim)

        if show_chi2_label and ref_metric_values:
            mean_ref = float(np.mean(ref_metric_values))
            metric_tex = (
                r"D_{\mathrm{red}}"
                if is_poisson
                else r"\chi^2_{\mathrm{red}}"
            )
            ax_prof.text(
                0.97, 0.95,
                rf"$\langle {metric_tex}\rangle = {mean_ref:.2f}$",
                transform=ax_prof.transAxes,
                ha="right", va="top",
                fontsize=font_size,
            )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.24)   # room for the legend below the axes

        if save_path is not None:
            fig.savefig(save_path)
            print(f"Saved figure to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return results
""" 
def _compare_matrices(
    matrices,
    labels,
    *,
    model_variances=None,
    n_annuli: int = 24,
    pixel_scale_arcsec: float = 45.0,
    bin_filter: float = 10.0,
    Fano_Data: float = 1.0,
    Fano_Model: float = 1.0,
    cmaps=None,
    a: float = 1000.0,
    center: str = "geom",
    figsize=None,
    font_size: int = 12,
    x_lim=None,
    y_lim=None,
    show_bin_contributions: bool = True,
    show_mean_bin_contribution_plot: bool = True,
    show_chi2_label: bool = True,
):

    from .plotting import normalize_and_log_scale
    from matplotlib.ticker import MaxNLocator  # type: ignore[import]
    plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'axes.grid': False, 'legend.frameon': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
    n = len(matrices)
    if model_variances is None:
        model_variances = [None] * n
    elif len(model_variances) != n:
        raise ValueError(
            "model_variances must have the same length as matrices."
        )

    validated_variances = []
    for index, (matrix, variance) in enumerate(zip(matrices, model_variances)):
        if variance is None:
            validated_variances.append(None)
            continue

        variance = np.squeeze(np.asarray(variance, dtype=np.float64))
        if variance.shape != matrix.shape:
            raise ValueError(
                f"Variance shape mismatch at index {index}: "
                f"V={variance.shape}, matrix={matrix.shape}."
            )
        if not np.all(np.isfinite(variance)):
            raise ValueError(
                f"Variance matrix at index {index} contains NaN or infinite values."
            )
        if np.any(variance < 0.0):
            raise ValueError(
                f"Variance matrix at index {index} contains negative values."
            )
        validated_variances.append(variance)

    model_variances = validated_variances
    default_cmaps = ["magma", "viridis", "plasma", "cividis", "inferno", "cubehelix"]

    base_cmaps = default_cmaps if cmaps is None else list(cmaps)
    if len(base_cmaps) == 0:
        base_cmaps = default_cmaps
    cmaps = [base_cmaps[i % len(base_cmaps)] for i in range(n)]

    # --- Shared colour range ---
    shared_min = float(min(m.min() for m in matrices))
    shared_max = float(max(m.max() for m in matrices))

    # --- Figure layout ---
    if figsize is None:
        figsize = (5 * n, 10)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.3])

    img_axes = [fig.add_subplot(gs[0, k]) for k in range(n)]
    ax_prof = fig.add_subplot(gs[1, :])

    # --- Top row: log-scaled images, matching plot_matrices_with_radial_overlay_log_three ---
    for ax, arr, label, cmap in zip(img_axes, matrices, labels, cmaps):
        scaled, _, _ = normalize_and_log_scale(
            arr,
            a,
            data_min=shared_min,
            data_max=shared_max,
        )
        ax.imshow(scaled, cmap=cmap, origin="lower")
        ax.set_title(label, fontsize=font_size)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Radial geometry (use last matrix as template, same as original) ---
    template = matrices[-1]
    H, W = template.shape
    yy, xx = np.indices((H, W))
    if center == "peak":
        cy, cx = np.unravel_index(np.nanargmax(template), template.shape)
        cy, cx = float(cy), float(cx)
    else:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = rr.max()
    edges_pix = np.linspace(0.0, r_max, n_annuli + 1)
    bins = np.clip(np.digitize(rr.ravel(), edges_pix) - 1, 0, n_annuli - 1)
    bin_centers_pix = 0.5 * (edges_pix[:-1] + edges_pix[1:])
    bin_centers_arc = bin_centers_pix * pixel_scale_arcsec

    N_pixels = np.bincount(bins, minlength=n_annuli).astype(np.float64)
    area_per_pix = pixel_scale_arcsec ** 2
    area_bin = N_pixels * area_per_pix

    def _radial_sum(arr):
        return np.bincount(bins, weights=np.nan_to_num(arr.ravel()), minlength=n_annuli)

    sums = [_radial_sum(m) for m in matrices]
    variance_sums = [
        _radial_sum(variance) if variance is not None else sums[index]
        for index, variance in enumerate(model_variances)
    ]
    sb_profiles = [np.zeros(n_annuli, dtype=np.float64) for _ in matrices]
    for sb, s in zip(sb_profiles, sums):
        valid_sb = s > bin_filter
        sb[valid_sb] = s[valid_sb] / area_bin[valid_sb]

    # --- Pairwise metric computation ---
    tab10 = plt.colormaps.get_cmap("tab10")
    profile_colors = [tab10(i % tab10.N) for i in range(n)]
    results = {}

    ref_idx = n - 1
    ref_sum = sums[ref_idx]
    valid_ref = ref_sum > bin_filter

    # One-sigma surface-brightness errors for the reference data.
    err_ref = np.zeros(n_annuli, dtype=np.float64)
    err_ref[valid_ref] = (
        np.sqrt(Fano_Data * ref_sum[valid_ref])
        / area_bin[valid_ref]
    )

    helper_profiles = {}
    pairwise_red_contributions = []
    pairwise_mean_values = []
    ref_chi2_values = []  # chi2 with the reference (last / "data") matrix as data, model fit to it

    metric_label = "χ²_red"
    print(f"\nPairwise {metric_label} (chi2):")
    print(f"  n_annuli={n_annuli}, bin_filter={bin_filter}, Fano Data ={Fano_Data}, Fano Model ={Fano_Model}, pixel_scale={pixel_scale_arcsec}\"")
    print("-" * 55)
    need_pairwise_contributions = show_bin_contributions or show_mean_bin_contribution_plot
    if n == 1:
        print(f"  Single input matrix provided; pairwise {metric_label} skipped.")
    else:
        for (i, j) in combinations(range(n), 2):
            val_ij, contrib_ij, dof_ij, summary_str_ij, hdata_ij = _call_chi2_pair(
                matrices[i], matrices[j],
                V=model_variances[j],
                n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                bin_filter=bin_filter, need_contributions=need_pairwise_contributions,
                Fano_Data=Fano_Data, Fano_Model=Fano_Model,
            )
            val_ji, contrib_ji, dof_ji, summary_str_ji, hdata_ji = _call_chi2_pair(
                matrices[j], matrices[i],
                V=model_variances[i],
                n_annuli=n_annuli, pixel_scale_arcsec=pixel_scale_arcsec,
                bin_filter=bin_filter, need_contributions=need_pairwise_contributions,
                Fano_Data=Fano_Data, Fano_Model=Fano_Model,
            )
            helper_profiles[(i, j)] = hdata_ij
            helper_profiles[(j, i)] = hdata_ji

            pair_mean = 0.5 * (val_ij + val_ji)
            pairwise_mean_values.append(pair_mean)
            if j == ref_idx:
                ref_chi2_values.append(val_ji)  # data=matrices[j] (reference), model=matrices[i] fit to it
            elif i == ref_idx:
                ref_chi2_values.append(val_ij)  # data=matrices[i] (reference), model=matrices[j] fit to it
            results[(i, j)] = (val_ij, val_ji)
            print(f"  {labels[i]} vs {labels[j]}")
            print(f"    {metric_label} ({labels[i]} as data) = {val_ij:.4f}   {summary_str_ij}")
            print(f"    {metric_label} ({labels[j]} as data) = {val_ji:.4f}   {summary_str_ji}")

            if need_pairwise_contributions:
                red_contrib_ij = contrib_ij / float(dof_ij)
                red_contrib_ji = contrib_ji / float(dof_ji)
                pairwise_red_contributions.append(red_contrib_ij)
                pairwise_red_contributions.append(red_contrib_ji)

                if show_bin_contributions:
                    label_i = str(labels[i])
                    label_j = str(labels[j])
                    print(f"    per-annulus reduced {metric_label} contributions:")
                    print(
                        "      bin | radius[arcsec] | area[arcsec^2] | "
                        f"counts({label_i}) | counts({label_j}) | "
                        f"{label_i}->{label_j} | {label_j}->{label_i}"
                    )
                    for b in range(n_annuli):
                        print(
                            f"      {b+1:>3d} | {bin_centers_arc[b]:>13.3f} | {area_bin[b]:>13.3f} | "
                            f"{float(sums[i][b]):>12.3f} | {float(sums[j][b]):>13.3f} | "
                            f"{red_contrib_ij[b]:>11.4e} | {red_contrib_ji[b]:>11.4e}"
                        )

        if pairwise_mean_values:
            mean_all = float(np.mean(pairwise_mean_values))
            std_all = float(np.std(pairwise_mean_values))
            print(f"  Mean {metric_label} (all pairs, both directions) = {mean_all:.4f} ± {std_all:.4f}")
        if ref_chi2_values:
            mean_ref = float(np.mean(ref_chi2_values))
            std_ref = float(np.std(ref_chi2_values))
            print(f"  Mean {metric_label} (model fit to reference '{labels[ref_idx]}' as data) = {mean_ref:.4f} ± {std_ref:.4f}")
        if results:
            print(f"  Pairwise {metric_label} summary (mean of both directions):")
            for (i, j), (v_ij, v_ji) in results.items():
                print(f"    {labels[i]} x {labels[j]} = {0.5 * (v_ij + v_ji):.4f}")
    print("-" * 55)

    # --- Radial profile plots (log-line + linear-bars) ---
    x_vals = bin_centers_arc
    valid_bins_plot = valid_ref
    profiles_to_plot = [np.asarray(p, dtype=np.float64) for p in sb_profiles]
    ref_errors_plot = np.asarray(err_ref, dtype=np.float64)
    errors_plot = [np.zeros(n_annuli, dtype=np.float64) for _ in range(n)]
    errors_plot[ref_idx] = ref_errors_plot

    if n >= 2:
        ref_profile = None
        display_profiles = [None] * n

        for idx in range(n - 1):
            helper_key = (ref_idx, idx)
            if helper_key not in helper_profiles:
                continue

            _, mu_ref, mu_other, n_best = helper_profiles[helper_key]
            if ref_profile is None:
                ref_profile = mu_ref

            model_profile = np.zeros_like(mu_ref, dtype=np.float64)
            valid_bins = ref_sum > bin_filter
            model_profile[valid_bins] = n_best * mu_other[valid_bins]
            display_profiles[idx] = model_profile

            # One-sigma model surface-brightness errors use the propagated
            # variance when supplied, or Poisson variance as a fallback.
            model_error = np.zeros_like(mu_ref, dtype=np.float64)
            model_error[valid_bins] = (
                n_best
                * np.sqrt(Fano_Model * variance_sums[idx][valid_bins])
                / area_bin[valid_bins]
            )
            errors_plot[idx] = model_error

        if ref_profile is not None:
            valid_bins = ref_sum > bin_filter
            display_profiles[ref_idx] = np.asarray(ref_profile, dtype=np.float64)
            valid_bins_plot = valid_bins
            profiles_to_plot = [
                np.asarray(p, dtype=np.float64) if p is not None else np.zeros_like(ref_profile, dtype=np.float64)
                for p in display_profiles
            ]

    xv = x_vals[valid_bins_plot]
    for profile, err, label, color in zip(profiles_to_plot, errors_plot, labels, profile_colors):
        ax_prof.plot(xv, profile[valid_bins_plot], "-", color=color, label=label)
        ax_prof.errorbar(
            xv,
            profile[valid_bins_plot],
            yerr=err[valid_bins_plot],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            alpha=0.8,
            capsize=0,
        )

    ax_prof.set_xlabel(r"Radius $\left[\mathrm{arcsec}\right]$", fontsize=font_size)
    ax_prof.set_ylabel(
        r"Surface Brightness $\left[\mathrm{photons}\,\mathrm{arcsec}^{-2}\right]$",
        fontsize=font_size,
    )
    ax_prof.set_yscale("log")
    ax_prof.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_prof.tick_params(axis="both", labelsize=font_size)
    ax_prof.legend(fontsize=font_size)
    ax_prof.grid(True, which="both", linestyle="--", alpha=0.3)

    if x_lim is not None:
        ax_prof.set_xlim(x_lim)
    if y_lim is not None:
        ax_prof.set_ylim(y_lim)

    if show_chi2_label and ref_chi2_values:
        mean_ref = float(np.mean(ref_chi2_values))
        ax_prof.text(
            0.5, -0.12,
            rf"Mean $\chi^2_{{\mathrm{{red}}}}$ (model fit to data) = {mean_ref:.4f}",
            transform=ax_prof.transAxes,
            ha="center", va="top",
            fontsize=font_size,
        )

    fig.tight_layout()
    plt.show()

    return results

 """
