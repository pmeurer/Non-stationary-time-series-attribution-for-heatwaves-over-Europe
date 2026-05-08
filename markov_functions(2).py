"""
markov_functions.py
-------------------
Core functions and model classes for non-stationary Markov chain attribution
of heatwave time series.

Assumptions (simplified from full version):
- Markov chain order 1 (bivariate copula only)
- No seasonal cycle
- No ensemble bootstrapping
"""

import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy.polynomial.legendre import Legendre
from scipy.stats import genpareto
from statsmodels.base.model import GenericLikelihoodModel
import statsmodels.api as sm

plt.style.use("ggplot")


# ── Legendre helpers ───────────────────────────────────────────────────────────

def legendre_matrix(data, degree):
    """Build a design matrix of Legendre polynomials up to `degree`.

    Parameters
    ----------
    data:
        1-D array whose length sets the number of evaluation points.
    degree:
        Maximum polynomial degree (0–5).

    Returns
    -------
    numpy.ndarray of shape (len(data), degree + 2) including a constant column.
    """
    n = len(data)
    bases = np.array([Legendre.basis(k, [-1, 1]).linspace(n)[1] for k in range(6)]).T
    return sm.add_constant(bases[:, : degree + 1])


def _build_legendre_matrix(n_points, degree):
    """Convenience wrapper: build Legendre matrix for n_points evaluation points."""
    return legendre_matrix(np.ones(n_points), degree)


# ── Quantile and logistic regression ──────────────────────────────────────────

def quantile_model(data, degree, q, start_year, end_year):
    """Fit a quantile regression model using Legendre polynomials.

    Returns
    -------
    tuple: (annual_mean_series, summary, design_matrix, fitted_values, params, bic)
    """
    lm = legendre_matrix(data, degree)
    fit = sm.QuantReg(data, lm).fit(q=q)
    fitted = fit.predict(lm)

    idx = pd.date_range(start=f"{start_year}-01-01",
                        end=f"{end_year}-12-31",
                        periods=len(data))
    series = pd.Series(fitted, index=idx)
    annual_mean = series.groupby(series.index.year).mean()

    diff = data - fitted
    rho = np.where(diff < 0, (q - 1) * diff, q * diff)
    bic = np.sum(rho * diff) + lm.shape[1] * np.log(len(data))

    return annual_mean, fit.summary(), lm, fitted, fit.params, bic


def logistic_model(data, all_data, degree):
    """Fit a logistic regression model using Legendre polynomials.

    Parameters
    ----------
    data:
        Binary response vector (exceedance indicators) used for fitting.
    all_data:
        Full time series used for prediction (may be longer than `data`).
    degree:
        Legendre polynomial degree.

    Returns
    -------
    tuple: (fitted_model, predicted_probabilities)
    """
    lm_fit = legendre_matrix(data, degree)
    lm_pred = legendre_matrix(all_data, degree)
    fit = sm.Logit(data, lm_fit).fit()
    return fit, fit.predict(lm_pred)


def run_logistic_regression(data, all_data, q, max_degree, scenario):
    """Select optimal degree logistic regression by BIC.

    Returns
    -------
    tuple: best (fitted_model, predicted_probabilities)
    """
    binary = create_binary_vector(data, q)
    candidates = [logistic_model(binary, all_data, d) for d in range(1, max_degree + 1)]
    opt = int(np.argmin([c[0].bic for c in candidates]))
    print(f"Optimal degree logistic regression ({scenario}): {opt}")
    return candidates[opt]


def run_quantile_regression(data, q, max_degree, scenario, start_year, end_year):
    """Select optimal degree quantile regression by BIC.

    Returns
    -------
    tuple: best quantile_model output
    """
    bics = [quantile_model(data, d, q, start_year, end_year)[-1]
            for d in range(1, max_degree + 1)]
    opt = int(np.argmin(bics))
    print(f"Optimal degree quantile regression ({scenario}): {opt}")
    return quantile_model(data, opt + 1, q, start_year, end_year)


# ── Utility functions ──────────────────────────────────────────────────────────

def create_binary_vector(data, q):
    """Return 1 where data exceeds its q-th quantile, else 0."""
    return np.where(data > np.nanquantile(data, q), 1, 0)


def transform_z(x, threshold, scale, shape):
    """Transform exceedances to exponential scale via GPD probability integral."""
    return -np.log((1 + shape / scale * (x - threshold)) ** (-1 / shape))


def link_func(ts, params, degree):
    """Logistic link function evaluated on a Legendre basis."""
    return 1 / (1 + np.exp(-np.dot(legendre_matrix(ts, degree), params)))


def get_pairwise_timeseries_from_univariate(ts):
    """Convert a 1-D time series into consecutive pairs (t, t+1)."""
    n = len(ts)
    pairs = np.full((n - 1, 2), np.nan)
    for i in range(n - 1):
        pairs[i] = [ts[i], ts[i + 1]]
    return pairs


def interpolate_nan(arr):
    """Linearly interpolate NaN values in a 1-D array."""
    valid = ~np.isnan(arr)
    return np.interp(np.arange(len(arr)), np.arange(len(arr))[valid], arr[valid])


# ── I/O helpers ────────────────────────────────────────────────────────────────

def read_in_file_ensemble(file_path, scenario, folders):
    """Load EPI ensemble data from NetCDF files.

    Returns
    -------
    tuple: (dataset, num_years, start_year, end_year, num_members,
            flat_array [, flat_array_no_nan])  # no-nan only for hist
    """
    if not folders and scenario == "hist":
        ds = xr.open_mfdataset(file_path, combine="nested",
                               concat_dim="members", engine="netcdf4")
    else:
        nc_files = glob.glob(os.path.join(file_path, "*/*.nc"))
        ds = xr.open_mfdataset(nc_files, combine="nested",
                               concat_dim="members", engine="netcdf4")

    num_years = len(ds.EPI.groupby("time.year"))
    start_year = int(ds.time.dt.year.min())
    end_year = int(ds.time.dt.year.max())
    num_mem = len(ds.members)
    days_per_year = len(ds.sel(time=ds.time.dt.year == int(ds.time.dt.year[0])).time)

    blocks = np.split(ds.EPI.values, num_years, axis=1)
    flat = np.concatenate(
        np.array(blocks).reshape(num_years, num_mem * days_per_year)
    )

    if scenario == "hist":
        return ds, num_years, start_year, end_year, num_mem, flat, flat[~np.isnan(flat)]
    return ds, num_years, start_year, end_year, num_mem, flat


def read_in_file(file_path):
    """Load a single-member EPI NetCDF file (e.g. ERA5).

    Returns
    -------
    tuple: (dataset, num_years, start_year, end_year, epi_values)
    """
    ds = xr.open_dataset(file_path, engine="netcdf4")
    num_years = len(ds.EPI.groupby("time.year"))
    start_year = int(ds.time.dt.year.min())
    end_year = int(ds.time.dt.year.max())
    return ds, num_years, start_year, end_year, ds.EPI.values


# ── GPD parameter helpers ──────────────────────────────────────────────────────

def get_parameter(num_years, days_per_year, degree_scale, degree_shape,
                  params, num_members, type_of_par):
    """Evaluate fitted GPD scale or shape parameter for all time steps.

    Parameters
    ----------
    num_years, days_per_year, num_members:
        Data dimensions.
    degree_scale, degree_shape:
        Optimal polynomial degrees for scale and shape.
    params:
        Fitted GPD parameter vector.
    type_of_par:
        ``"scale"`` or ``"shape"``.

    Returns
    -------
    numpy.ndarray of length num_years * days_per_year * num_members.
    """
    n_total = num_years * days_per_year * num_members
    lm = _build_legendre_matrix(n_total, degree_scale if type_of_par == "scale" else degree_shape)

    if type_of_par == "scale":
        par = np.dot(lm, params[: degree_scale + 1])
        return np.exp(par)
    else:
        par = np.dot(lm, params[degree_scale + 1: degree_scale + degree_shape + 2])
        return par


# ── Extreme-value model fitting ────────────────────────────────────────────────

def find_extremes_model1(dataset, dataset_flat, q, start_year, end_year,
                         num_years, max_degree, scenario, output_dir, model_label):

    threshold = np.nanquantile(dataset_flat, q)

    if scenario == "era5":
        extremes = dataset.EPI.to_pandas()
        extremes = extremes[extremes > threshold]
    else:
        parts = [dataset.sel(members=i).EPI.to_pandas() for i in range(len(dataset.members))]
        extremes = pd.concat(parts).sort_index()
        extremes = extremes[extremes > threshold]

    lm_full, _ = _build_extreme_design_matrix(
        extremes, start_year, end_year, num_years, max_degree
    )
    gpd_fits = _fit_gpd_grid(extremes.to_numpy() - threshold, lm_full, max_degree)
    bics = np.array([f.bic for f in gpd_fits])

    # ── BIC matrix plot with actual values ────────────────────────────────────
    plot_bic_matrix(bics, max_degree, scenario, output_dir, model_label, q)

    best, deg_scale, deg_shape = _select_best_gpd(gpd_fits, max_degree)

    indices = np.where(np.isin(dataset_flat, extremes.values))[0]
    sc = np.exp(np.dot(legendre_matrix(dataset_flat, deg_scale),
                       best.params[: deg_scale + 1])[indices])
    sh = np.dot(legendre_matrix(dataset_flat, deg_shape),
                best.params[deg_scale + 1: deg_scale + deg_shape + 2])[indices]
    min_len = min(len(extremes.values), len(indices))
    _diag_plot(
        transform_z(extremes.values[:min_len], threshold, sc[:min_len], sh[:min_len]),
        scenario, output_dir, model_label, q
    )

    print(f"  GPD scale degree ({scenario}): {deg_scale}")
    print(f"  GPD shape degree ({scenario}): {deg_shape}")
    return best, threshold, deg_scale, deg_shape, best.params


def find_extremes_model2(dataset, dataset_flat, q, start_year, end_year,
                         num_years, max_degree, scenario, output_dir,
                         model_label, qm):

    if scenario == "era5":
        merged = dataset.EPI.to_pandas()
    else:
        parts = [dataset.sel(members=i).EPI.to_pandas() for i in range(len(dataset.members))]
        merged = pd.concat(parts).sort_index()

    exceedances_per_year = [
        merged[[d.year == start_year + i for d in merged.index]] - qm[0].iloc[i]
        for i in range(num_years)
    ]
    extremes = pd.concat(exceedances_per_year).sort_index()
    extremes = extremes[extremes >= 0]

    lm_full, _ = _build_extreme_design_matrix(
        extremes, start_year, end_year, num_years, max_degree
    )
    gpd_fits = _fit_gpd_grid(extremes.to_numpy(), lm_full, max_degree)
    bics = np.array([f.bic for f in gpd_fits])

    # ── BIC matrix plot with actual values ────────────────────────────────────
    plot_bic_matrix(bics, max_degree, scenario, output_dir, model_label, q)

    best, deg_scale, deg_shape = _select_best_gpd(gpd_fits, max_degree)

    indices = np.where(
        np.isin(pd.concat(exceedances_per_year).sort_index().to_numpy(), extremes.values)
    )[0]
    sc = np.exp(
        np.dot(legendre_matrix(dataset_flat, deg_scale),
               best.params[: deg_scale + 1])[indices]
    )
    sh = np.dot(
        legendre_matrix(dataset_flat, deg_shape),
        best.params[deg_scale + 1: deg_scale + deg_shape + 2]
    )[indices]
    min_len = min(len(extremes.values), len(indices))
    _diag_plot(
        transform_z(extremes.values[:min_len], 0, sc[:min_len], sh[:min_len]),
        scenario, output_dir, model_label, q
    )

    print(f"  GPD scale degree ({scenario}): {deg_scale}")
    print(f"  GPD shape degree ({scenario}): {deg_shape}")
    return best, extremes, deg_scale, deg_shape, best.params


# ── Internal helpers for extreme-value fitting ─────────────────────────────────

def _build_extreme_design_matrix(extremes, start_year, end_year, num_years, max_degree):
    """Build Legendre design matrix repeated for each extreme observation."""
    lm_annual = legendre_matrix(np.arange(num_years), max_degree)

    counts = []
    for yr in range(start_year, end_year + 1):
        mask = [d.year == yr for d in extremes.index]
        counts.append(int(np.sum(mask)) if np.sum(mask) > 0 else 0)

    rows = []
    for i, cnt in enumerate(counts):
        if cnt > 0:
            rows.append(np.tile(lm_annual[i], (cnt, 1)))
    legendre_rep = np.concatenate(rows)
    legendre_rep = legendre_rep[~np.all(legendre_rep == 0, axis=1)]
    return legendre_rep, counts


def _fit_gpd_grid(endog, lm, max_degree):
    """Fit GPD over all (degree_scale, degree_shape) combinations."""
    fits = []
    dummy_exog2 = np.zeros((len(endog), 2))  # placeholder, not used without seasonal cycle
    for i in range(max_degree + 1):
        for j in range(i + 1):
            fits.append(GPD(endog, lm[:, : 1 + max(i, j)], dummy_exog2, i, j).fit())
    return fits


def _select_best_gpd(fits, max_degree):
    """Return best GPD fit and its (degree_scale, degree_shape) by BIC."""
    bics = np.array([f.bic for f in fits])
    scale_degrees = [i for i in range(max_degree + 1) for j in range(i + 1)]
    shape_degrees = [j for i in range(max_degree + 1) for j in range(i + 1)]
    best_idx = int(np.argmin(bics))
    return fits[best_idx], scale_degrees[best_idx], shape_degrees[best_idx]




# ── Plot helpers ───────────────────────────────────────────────────────────────

def _diag_plot(z, scenario, output_dir, model, q):
    """Residual probability and quantile diagnostic plots for GPD fit."""
    n = len(z)
    x = np.arange(1, n + 1) / (n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.scatter(x, 1 - np.exp(-np.sort(z)), color="black", label="Model")
    ax1.plot([0, 1], [0, 1], color="blue")
    ax1.set_xlabel("Residual Empirical Probabilities")
    ax1.set_ylabel("Residual Model Probabilities")
    ax1.set_title(f"Residual Probability Plot – {scenario}")

    ax2.scatter(-np.log(1 - x), np.sort(z), color="black", label="Empirical")
    ax2.axline((1, 1), slope=1, color="blue")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("(Standardized) Residual Quantiles")
    ax2.set_ylabel("Empirical Residual Quantiles")
    ax2.set_title(f"Residual Quantile Plot – {scenario}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/diag_{scenario}_{model}_{q}.png",
                dpi=600, bbox_inches="tight")
    plt.close()

# keep public alias for backward compatibility
diag = _diag_plot


def plot_bic_matrix(values, max_degree, scenario, output_dir, model, q):
    """Plot BIC values as a heat map over (degree_scale, degree_shape) grid."""
    matrix = np.full((max_degree + 1, max_degree + 1), np.nan)
    for (i, j), v in zip(
        [(i, j) for i in range(max_degree + 1) for j in range(i + 1)], values
    ):
        matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix, cmap="summer", interpolation="nearest")
    best = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    ax.scatter(best[1], best[0], color="red", s=40, marker="x")
    plt.colorbar(im)
    ax.set_title(f"BIC GPD-fit – {scenario}")
    ticks = list(range(max_degree + 1))
    ax.set_xticks(ticks, [rf"$\varsigma_{{\xi,{k}}}$" for k in ticks])
    ax.set_yticks(ticks, [rf"$\varsigma_{{\sigma,{k}}}$" for k in ticks])
    plt.savefig(f"{output_dir}/bic_matrix_{scenario}_{model}_{q}.png",
                dpi=600, bbox_inches="tight")
    plt.close()


def parameter_plot(phi_hist, phi_histnat, scale_hist, scale_histnat,
                   shape_hist, shape_histnat, alpha_hist, alpha_histnat):
    """Plot time series of fitted Markov-GPD parameters (phi, sigma, xi, alpha)."""
    days_per_year = 92
    num_years = len(phi_hist) // days_per_year
    years = np.arange(1940, 1940 + num_years)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    c_hist, c_histnat = "orangered", "deepskyblue"

    def _plot(ax, h, hn, ylabel, ylim):
        ax.plot(years, h.reshape(num_years, days_per_year).mean(axis=1),
                color=c_hist, label="hist")
        ax.plot(years, hn.reshape(num_years, days_per_year).mean(axis=1),
                color=c_histnat, label="histnat")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    _plot(axs[0, 0], phi_hist, phi_histnat, r"$\phi_t$", (0, 0.2))
    _plot(axs[0, 1], scale_hist, scale_histnat, r"$\sigma$", (0, 3))
    _plot(axs[1, 0], shape_hist, shape_histnat, r"$\xi$", (-1, 1))
    _plot(axs[1, 1], alpha_hist, alpha_histnat, r"$\alpha$", (0, 1))

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


# ── Likelihood model classes ───────────────────────────────────────────────────

class GPD(GenericLikelihoodModel):
    """Generalised Pareto Distribution with Legendre-polynomial scale and shape."""

    def __init__(self, endog, exog, exog2, degree_scale, degree_shape, **kwds):
        self.degree_scale = degree_scale
        self.degree_shape = degree_shape
        super().__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        beta = params[: self.degree_scale + 1]
        gamma = params[self.degree_scale + 1: self.degree_scale + self.degree_shape + 2]
        sigma = np.exp(np.dot(self.exog[:, : self.degree_scale + 1], beta))
        xi = np.dot(self.exog[:, : self.degree_shape + 1], gamma)
        return -genpareto.logpdf(self.endog, c=xi, scale=sigma)

    def fit(self, start_params=None, maxiter=10000, **kwds):
        if start_params is None:
            start_params = np.zeros(self.degree_scale + self.degree_shape + 2)
            start_params[0] = np.sqrt(6 * np.var(self.endog)) / np.pi
            start_params[self.degree_scale + 1] = 1e-8
        return super().fit(start_params=start_params, method="BFGS",
                           maxiter=maxiter, **kwds)


class FitMCGPD(GenericLikelihoodModel):
    """Bivariate Markov-GPD copula model (first-order, constant threshold)."""

    def __init__(self, endog, exog, sigma, xi, phi, thres, degree_alpha, **kwds):
        super().__init__(endog, exog, **kwds)
        self.degree_alpha = degree_alpha
        self.phi1 = self.phi2 = phi
        self.thres1 = self.thres2 = thres
        self.sigma = sigma
        self.xi = xi

    # ── Pickands dependence function derivatives ───────────────────────────────
    def _V(self, z1, z2, a):
        return (z1 ** (-1 / a) + z2 ** (-1 / a)) ** a

    def _V1(self, z1, z2, a):
        return -(z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 1) * z1 ** (-1 / a - 1)

    def _V2(self, z1, z2, a):
        return -(z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 1) * z2 ** (-1 / a - 1)

    def _V12(self, z1, z2, a):
        return ((a - 1) / a * (z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 2)
                * z1 ** (-1 / a - 1) * z2 ** (-1 / a - 1))

    def nloglikeobs(self, params):
        sigma1 = sigma2 = self.sigma
        xi1 = xi2 = self.xi
        alpha = 1 / (1 + np.exp(-np.dot(
            self.exog[:, : self.degree_alpha + 1], params[: self.degree_alpha + 1]
        )))

        r1 = -1 / np.log(1 - self.phi1)
        r2 = -1 / np.log(1 - self.phi2)

        t1 = (1 + xi1 * (self.endog[:, 0] - self.thres1) / sigma1) ** (-1 / xi1)
        t2 = (1 + xi2 * (self.endog[:, 1] - self.thres2) / sigma2) ** (-1 / xi2)
        z1 = -1 / np.log(1 - self.phi1 * t1)
        z2 = -1 / np.log(1 - self.phi2 * t2)

        K1 = -self.phi1 / sigma1 * t1 ** (1 + xi1) * z1 ** 2 * np.exp(1 / z1)
        K2 = -self.phi2 / sigma2 * t2 ** (1 + xi2) * z2 ** 2 * np.exp(1 / z2)

        term_both = np.exp(-self._V(z1, z2, alpha)) * (
            self._V1(z1, z2, alpha) * self._V2(z1, z2, alpha)
            - self._V12(z1, z2, alpha)
        ) * K1 * K2
        term_only2 = np.exp(-self._V(r1, z2, alpha)) * self._V2(r1, z2, alpha) * K2
        term_only1 = np.exp(-self._V(z1, r2, alpha)) * self._V1(z1, r2, alpha) * K1
        term_none = np.exp(-self._V(r1, r2, alpha))

        exc1 = (self.endog[:, 0] > self.thres1).astype(int)
        exc2 = (self.endog[:, 1] > self.thres2).astype(int)
        ll = np.where(
            (exc1 == 1) & (exc2 == 1), term_both,
            np.where(exc2 == 1, term_only2,
                     np.where(exc1 == 1, term_only1, term_none))
        )

        uv = np.where(
            self.endog[:, 0] > self.thres1,
            self.phi1 * genpareto.pdf(self.endog[:, 0], c=xi1,
                                      loc=self.thres1, scale=sigma1),
            1 - self.phi1,
        )
        ll_ready = -(np.log(ll[: len(uv)]) - np.log(uv))
        return ll_ready[np.isfinite(ll_ready)]

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = np.zeros(1 + self.degree_alpha)
            start_params[0] = 0.75
        return super().fit(start_params=start_params, method="bfgs",
                           maxiter=maxiter, **kwds)


class FitMCGPD_joint(GenericLikelihoodModel):
    """Joint Markov-GPD model fitting scale, shape, and alpha simultaneously."""

    def __init__(self, endog, exog, degree_scale, degree_shape, degree_alpha,
                 phi, thres, start_vec, **kwds):
        super().__init__(endog, exog, **kwds)
        self.degree_scale = degree_scale
        self.degree_shape = degree_shape
        self.degree_alpha = degree_alpha
        self.phi1 = self.phi2 = phi
        self.thres1 = self.thres2 = thres
        self.start_vec = start_vec

    def _V(self, z1, z2, a):
        return (z1 ** (-1 / a) + z2 ** (-1 / a)) ** a

    def _V1(self, z1, z2, a):
        return -(z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 1) * z1 ** (-1 / a - 1)

    def _V2(self, z1, z2, a):
        return -(z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 1) * z2 ** (-1 / a - 1)

    def _V12(self, z1, z2, a):
        return ((self.degree_alpha - 1 if False else (
            (z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 2)
            * z1 ** (-1 / a - 1) * z2 ** (-1 / a - 1) * (a - 1) / a
        )))

    def _V12(self, z1, z2, a):
        return ((a - 1) / a * (z1 ** (-1 / a) + z2 ** (-1 / a)) ** (a - 2)
                * z1 ** (-1 / a - 1) * z2 ** (-1 / a - 1))

    def nloglikeobs(self, params):
        ds, dsh = self.degree_scale, self.degree_shape
        beta = params[: ds + 1]
        gamma = params[ds + 1: ds + dsh + 2]
        alpha_params = params[ds + dsh + 2:]

        sigma = np.exp(np.dot(self.exog[:, : ds + 1], beta))
        xi = np.dot(self.exog[:, : dsh + 1], gamma)
        alpha = 1 / (1 + np.exp(-np.dot(
            self.exog[:, : self.degree_alpha + 1], alpha_params
        )))

        r1 = -1 / np.log(1 - self.phi1)
        r2 = -1 / np.log(1 - self.phi2)

        t1 = (1 + xi * (self.endog[:, 0] - self.thres1) / sigma) ** (-1 / xi)
        t2 = (1 + xi * (self.endog[:, 1] - self.thres2) / sigma) ** (-1 / xi)
        z1 = -1 / np.log(1 - self.phi1 * t1)
        z2 = -1 / np.log(1 - self.phi2 * t2)

        K1 = -self.phi1 / sigma * t1 ** (1 + xi) * z1 ** 2 * np.exp(1 / z1)
        K2 = -self.phi2 / sigma * t2 ** (1 + xi) * z2 ** 2 * np.exp(1 / z2)

        term_both = np.exp(-self._V(z1, z2, alpha)) * (
            self._V1(z1, z2, alpha) * self._V2(z1, z2, alpha)
            - self._V12(z1, z2, alpha)
        ) * K1 * K2
        term_only2 = np.exp(-self._V(r1, z2, alpha)) * self._V2(r1, z2, alpha) * K2
        term_only1 = np.exp(-self._V(z1, r2, alpha)) * self._V1(z1, r2, alpha) * K1
        term_none = np.exp(-self._V(r1, r2, alpha))

        exc1 = (self.endog[:, 0] > self.thres1).astype(int)
        exc2 = (self.endog[:, 1] > self.thres2).astype(int)
        ll = np.where(
            (exc1 == 1) & (exc2 == 1), term_both,
            np.where(exc2 == 1, term_only2,
                     np.where(exc1 == 1, term_only1, term_none))
        )

        uv = np.where(
            self.endog[:, 0] > self.thres1,
            self.phi1 * genpareto.pdf(self.endog[:, 0], c=xi,
                                      loc=self.thres1, scale=sigma),
            1 - self.phi1,
        )
        ll_ready = -(np.log(ll) - np.log(uv[: len(ll)]))
        return ll_ready[np.isfinite(ll_ready)]

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwds):
        if start_params is None:
            start_params = self.start_vec
        return super().fit(start_params=start_params, method="bfgs",
                           maxiter=maxiter, **kwds)


# ── Main analysis class ────────────────────────────────────────────────────────

class attribution_analysis:
    """Non-stationary first-order Markov chain attribution analysis.

    Parameters
    ----------
    main_folder_hist, main_folder_histnat:
        Paths to folders containing historical and hist-nat EPI NetCDF files.
    q:
        Exceedance probability threshold (e.g. 0.95).
    model_name:
        CMIP6 model identifier string.
    max_degree:
        Maximum Legendre polynomial degree for GPD fitting.
    max_degree_alpha:
        Maximum Legendre polynomial degree for copula fitting.
    folders_hist, folders_histnat:
        Whether input data are stored in subdirectories.
    output_dir:
        Path for writing result files.
    num_of_years_ssp:
        Number of SSP years appended to historical run.
    era5_file_path:
        Path to the ERA5 EPI NetCDF file.
    """

    def __init__(self, main_folder_hist, main_folder_histnat, q, model_name,
                 max_degree, max_degree_alpha, folders_hist, folders_histnat,
                 output_dir, num_of_years_ssp, era5_file_path):
        self.main_folder_hist = main_folder_hist
        self.main_folder_histnat = main_folder_histnat
        self.q = q
        self.model_name = model_name
        self.max_degree = max_degree
        self.max_degree_alpha = max_degree_alpha
        self.folders_hist = folders_hist
        self.folders_histnat = folders_histnat
        self.output_dir = output_dir
        self.num_of_years_ssp = num_of_years_ssp
        self.era5_file_path = era5_file_path

    def _load_data(self):
        """Load hist, histnat, and ERA5 datasets."""
        data_hist = read_in_file_ensemble(
            self.main_folder_hist, "hist", self.folders_hist
        )
        data_histnat = read_in_file_ensemble(
            self.main_folder_histnat, "histnat", self.folders_histnat
        )
        data_era5 = read_in_file(self.era5_file_path)
        return data_hist, data_histnat, data_era5

    def _fit_copula(self, ts, scale, shape, phi, thres):
        """Fit FitMCGPD for degrees 0..max_degree_alpha; return list of fits."""
        pairs = get_pairwise_timeseries_from_univariate(ts)
        fits = []
        for deg in range(self.max_degree_alpha + 1):
            lm = legendre_matrix(ts, deg)
            fits.append(
                FitMCGPD(pairs, lm[:-1], scale[:-1], shape[:-1],
                         phi[:-1], thres[:-1], deg).fit()
            )
        return fits

    def _fit_joint(self, ts, lm, deg_scale, deg_shape, deg_alpha, phi, thres, par):
        """Fit FitMCGPD_joint given optimal degrees and initial parameters."""
        pairs = get_pairwise_timeseries_from_univariate(ts)
        max_deg = max(deg_scale, deg_shape, deg_alpha)
        lm_joint = legendre_matrix(ts, max_deg)
        return FitMCGPD_joint(
            pairs[:-1] if len(pairs) > len(lm_joint) - 1 else pairs,
            lm_joint[:-1],
            deg_scale, deg_shape, deg_alpha,
            phi[:-1], thres[:-1], par
        ).fit()

    def _save_results(self, ds, copula_hist, copula_histnat,
                      joint_hist, joint_histnat,
                      opt_alpha_hist, opt_alpha_histnat, model_tag):
        """Write result dataset and model objects to output directory."""
        tag = f"{self.model_name}_{self.q}"
        ds.to_netcdf(f"{self.output_dir}/{model_tag}_{tag}.nc", engine="netcdf4")
        joint_hist.save(f"{self.output_dir}/{model_tag}_hist_{tag}.pkl")
        joint_histnat.save(f"{self.output_dir}/{model_tag}_histnat_{tag}.pkl")

    def _build_result_dataset(self, scale_hist, scale_histnat, shape_hist, shape_histnat,
                               alpha_hist, alpha_histnat, phi_hist, phi_histnat,
                               time_coord, model_type):
        """Assemble xr.Dataset of fitted parameters evaluated on ERA5 time axis."""
        coords = {"time": time_coord}
        ds = xr.Dataset(
            {
                "scale_hist":     xr.DataArray(scale_hist,    coords=coords, dims="time"),
                "scale_histnat":  xr.DataArray(scale_histnat, coords=coords, dims="time"),
                "shape_hist":     xr.DataArray(shape_hist,    coords=coords, dims="time"),
                "shape_histnat":  xr.DataArray(shape_histnat, coords=coords, dims="time"),
                "alpha_hist":     xr.DataArray(alpha_hist,    coords=coords, dims="time"),
                "alpha_histnat":  xr.DataArray(alpha_histnat, coords=coords, dims="time"),
                "phi_hist":       xr.DataArray(phi_hist,      coords=coords, dims="time"),
                "phi_histnat":    xr.DataArray(phi_histnat,   coords=coords, dims="time"),
            },
            attrs={
                "description":       "Fitted Markov chain model",
                "climate_model":     self.model_name,
                "experiment_hist":   "historical",
                "experiment_histnat": "hist-nat",
                "quantile":          str(self.q),
                "time_period":       "1940–2025",
                "created_on":        str(datetime.date.today()),
                "model_type":        model_type,
            },
        )
        return ds

    def run_analysis_model1(self):
        """Fit Model 1 (constant threshold) and write results to output_dir."""
        print(f"\n[{self.model_name}] Starting Model 1 analysis...")
        data_hist, data_histnat, data_era5 = self._load_data()

        days_per_year = len(
            data_hist[0].sel(time=data_hist[0].time.dt.year == int(data_hist[0].time.dt.year[0])).time
        )
        indices_hist = ~np.isnan(data_hist[-2])

        # ── Logistic regression for phi ───────────────────────────────────────
        lm_histnat = run_logistic_regression(
            data_histnat[-1], data_histnat[-1], self.q, self.max_degree, "histnat"
        )
        lm_hist = run_logistic_regression(
            data_hist[-1], data_hist[-2], self.q, self.max_degree, "hist"
        )

        # ── GPD fit ───────────────────────────────────────────────────────────
        hist_ext = find_extremes_model1(
            data_hist[0], data_hist[-2], self.q,
            data_hist[2], data_hist[3], data_hist[1],
            self.max_degree, "hist", self.output_dir, "model1"
        )
        histnat_ext = find_extremes_model1(
            data_histnat[0], data_histnat[-1], self.q,
            data_histnat[2], data_histnat[3], data_histnat[1],
            self.max_degree, "histnat", self.output_dir, "model1"
        )

        # ── Scale / shape parameters ──────────────────────────────────────────
        scale_hist = get_parameter(
            data_hist[1], days_per_year, hist_ext[2], hist_ext[3],
            hist_ext[0].params, data_hist[4], "scale"
        )
        shape_hist = get_parameter(
            data_hist[1], days_per_year, hist_ext[2], hist_ext[3],
            hist_ext[0].params, data_hist[4], "shape"
        )
        scale_histnat = get_parameter(
            data_histnat[1], days_per_year, histnat_ext[2], histnat_ext[3],
            histnat_ext[0].params, data_histnat[4], "scale"
        )
        shape_histnat = get_parameter(
            data_histnat[1], days_per_year, histnat_ext[2], histnat_ext[3],
            histnat_ext[0].params, data_histnat[4], "shape"
        )

        phi_hist_arr = lm_hist[1]
        phi_histnat_arr = lm_histnat[1]
        thres_hist = np.repeat(hist_ext[1], len(data_hist[-1]))
        thres_histnat = np.repeat(histnat_ext[1],
                                  data_histnat[1] * data_histnat[-2] * days_per_year)

        # ── Copula (stage 1) ──────────────────────────────────────────────────
        copula_histnat = self._fit_copula(
            data_histnat[-1], scale_histnat, shape_histnat,
            phi_histnat_arr, thres_histnat
        )
        copula_hist = self._fit_copula(
            data_hist[-2][indices_hist], scale_hist[indices_hist],
            shape_hist[indices_hist], phi_hist_arr[indices_hist],
            thres_hist                 # ← kein [indices_hist], passt schon
        )

        opt_alpha_hist = int(np.argmin([f.bic for f in copula_hist]))
        opt_alpha_histnat = int(np.argmin([f.bic for f in copula_histnat]))

        deg_scale_hist, deg_shape_hist = hist_ext[2], hist_ext[3]
        deg_scale_histnat, deg_shape_histnat = histnat_ext[2], histnat_ext[3]
        max_deg_hist = max(deg_scale_hist, deg_shape_hist, opt_alpha_hist)
        max_deg_histnat = max(deg_scale_histnat, deg_shape_histnat, opt_alpha_histnat)

        par_hist = np.concatenate((hist_ext[-1], copula_hist[opt_alpha_hist].params))
        par_histnat = np.concatenate((histnat_ext[-1], copula_histnat[opt_alpha_histnat].params))

        # ── Joint fit (stage 2) ───────────────────────────────────────────────
        lm_joint_hist = legendre_matrix(data_hist[-2], max_deg_hist)
        joint_hist = FitMCGPD_joint(
            get_pairwise_timeseries_from_univariate(data_hist[-2][indices_hist]),
            lm_joint_hist[indices_hist][:-1],
            deg_scale_hist, deg_shape_hist, opt_alpha_hist,
            phi_hist_arr[indices_hist][:-1], thres_hist[:-1], par_hist
        ).fit()

        lm_joint_histnat = legendre_matrix(data_histnat[-1], max_deg_histnat)
        joint_histnat = FitMCGPD_joint(
            get_pairwise_timeseries_from_univariate(data_histnat[-1]),
            lm_joint_histnat[:-1],
            deg_scale_histnat, deg_shape_histnat, opt_alpha_histnat,
            phi_histnat_arr[:-1], thres_histnat[:-1], par_histnat
        ).fit()

        params_hist = joint_hist.params
        params_histnat = joint_histnat.params

        # ── Evaluate on ERA5 ──────────────────────────────────────────────────
        era5_ts = data_era5[-1]
        scale_h = np.exp(np.dot(legendre_matrix(era5_ts, deg_scale_hist),
                                params_hist[: deg_scale_hist + 1]))
        shape_h = np.dot(legendre_matrix(era5_ts, deg_shape_hist),
                         params_hist[deg_scale_hist + 1: deg_scale_hist + deg_shape_hist + 2])
        alpha_h = link_func(era5_ts, params_hist[deg_scale_hist + deg_shape_hist + 2:],
                            opt_alpha_hist)

        scale_hn = np.exp(np.dot(legendre_matrix(era5_ts, deg_scale_histnat),
                                 params_histnat[: deg_scale_histnat + 1]))
        shape_hn = np.dot(legendre_matrix(era5_ts, deg_shape_histnat),
                          params_histnat[deg_scale_histnat + 1: deg_scale_histnat + deg_shape_histnat + 2])
        alpha_hn = link_func(era5_ts, params_histnat[deg_scale_histnat + deg_shape_histnat + 2:],
                             opt_alpha_histnat)

        phi_h = lm_hist[0].predict(legendre_matrix(data_era5[0].EPI.values,
                                                    len(lm_hist[0].params) - 1))
        phi_hn = lm_histnat[0].predict(legendre_matrix(data_era5[0].EPI.values,
                                                        len(lm_histnat[0].params) - 1))

        ds = self._build_result_dataset(
            scale_h, scale_hn, shape_h, shape_hn, alpha_h, alpha_hn,
            phi_h, phi_hn, data_era5[0].time, "constant threshold"
        )
        parameter_plot(phi_h, phi_hn, scale_h, scale_hn, shape_h, shape_hn, alpha_h, alpha_hn)

        tag = f"{self.model_name}_{self.q}"
        ds.to_netcdf(f"{self.output_dir}/results_model1_{tag}.nc", engine="netcdf4")
        joint_hist.save(f"{self.output_dir}/model1_hist_{tag}.pkl")
        joint_histnat.save(f"{self.output_dir}/model1_histnat_{tag}.pkl")
        print(f"[{self.model_name}] Model 1 analysis finished.")

    def run_analysis_model2(self):
        """Fit Model 2 (varying threshold) and write results to output_dir."""
        print(f"\n[{self.model_name}] Starting Model 2 analysis...")
        data_hist, data_histnat, data_era5 = self._load_data()

        days_per_year = len(
            data_hist[0].sel(time=data_hist[0].time.dt.year == 1960).time
        )
        indices_hist = ~np.isnan(data_hist[-2])

        # ── Quantile regression for varying threshold ─────────────────────────
        qm_histnat = run_quantile_regression(
            data_histnat[-1], self.q, self.max_degree,
            "histnat", data_histnat[2], data_histnat[3]
        )
        qm_hist = run_quantile_regression(
            data_hist[-1], self.q, self.max_degree,
            "hist", data_hist[2], data_hist[3]
        )

        # ── GPD fit ───────────────────────────────────────────────────────────
        hist_ext = find_extremes_model2(
            data_hist[0], data_hist[-2], self.q,
            data_hist[2], data_hist[3], data_hist[1],
            self.max_degree, "hist", self.output_dir, "model2", qm_hist
        )
        histnat_ext = find_extremes_model2(
            data_histnat[0], data_histnat[-1], self.q,
            data_histnat[2], data_histnat[3], data_histnat[1],
            self.max_degree, "histnat", self.output_dir, "model2", qm_histnat
        )

        # ── Scale / shape parameters ──────────────────────────────────────────
        scale_hist = get_parameter(
            data_hist[1], days_per_year, hist_ext[2], hist_ext[3],
            hist_ext[0].params, data_hist[4], "scale"
        )
        shape_hist = get_parameter(
            data_hist[1], days_per_year, hist_ext[2], hist_ext[3],
            hist_ext[0].params, data_hist[4], "shape"
        )
        scale_histnat = get_parameter(
            data_histnat[1], days_per_year, histnat_ext[2], histnat_ext[3],
            histnat_ext[0].params, data_histnat[4], "scale"
        )
        shape_histnat = get_parameter(
            data_histnat[1], days_per_year, histnat_ext[2], histnat_ext[3],
            histnat_ext[0].params, data_histnat[4], "shape"
        )

        phi_histnat_arr = np.repeat(1 - self.q,
                                    data_histnat[1] * data_histnat[-2] * days_per_year)
        thres_histnat = np.repeat(qm_histnat[0].values, data_histnat[-2] * days_per_year)
        phi_hist_arr = np.repeat(1 - self.q, len(data_hist[-2]))
        thres_hist = np.repeat(qm_hist[0].values, data_hist[-3] * days_per_year)

        # ── Copula (stage 1) ──────────────────────────────────────────────────
        copula_histnat = self._fit_copula(
            data_histnat[-1], scale_histnat, shape_histnat,
            phi_histnat_arr, thres_histnat
        )
        copula_hist = self._fit_copula(
            data_hist[-2][indices_hist], scale_hist[indices_hist],
            shape_hist[indices_hist], phi_hist_arr[indices_hist],
            thres_hist[indices_hist]
        )

        opt_alpha_hist = int(np.argmin([f.bic for f in copula_hist]))
        opt_alpha_histnat = int(np.argmin([f.bic for f in copula_histnat]))

        deg_scale_hist, deg_shape_hist = hist_ext[2], hist_ext[3]
        deg_scale_histnat, deg_shape_histnat = histnat_ext[2], histnat_ext[3]
        max_deg_hist = max(deg_scale_hist, deg_shape_hist, opt_alpha_hist)
        max_deg_histnat = max(deg_scale_histnat, deg_shape_histnat, opt_alpha_histnat)

        par_hist = np.concatenate((hist_ext[-1], copula_hist[opt_alpha_hist].params))
        par_histnat = np.concatenate((histnat_ext[-1], copula_histnat[opt_alpha_histnat].params))

        # ── Joint fit (stage 2) ───────────────────────────────────────────────
        lm_joint_hist = legendre_matrix(data_hist[-2], max_deg_hist)
        joint_hist = FitMCGPD_joint(
            get_pairwise_timeseries_from_univariate(data_hist[-2][indices_hist]),
            lm_joint_hist[indices_hist][:-1],
            deg_scale_hist, deg_shape_hist, opt_alpha_hist,
            phi_hist_arr[indices_hist][:-1], thres_hist[indices_hist][:-1], par_hist
        ).fit()

        lm_joint_histnat = legendre_matrix(data_histnat[-1], max_deg_histnat)
        joint_histnat = FitMCGPD_joint(
            get_pairwise_timeseries_from_univariate(data_histnat[-1]),
            lm_joint_histnat[:-1],
            deg_scale_histnat, deg_shape_histnat, opt_alpha_histnat,
            phi_histnat_arr[:-1], thres_histnat[:-1], par_histnat
        ).fit()

        params_hist = joint_hist.params
        params_histnat = joint_histnat.params

        # ── Evaluate on ERA5 ──────────────────────────────────────────────────
        era5_ts = data_era5[-1]
        scale_h = np.exp(np.dot(legendre_matrix(era5_ts, deg_scale_hist),
                                params_hist[: deg_scale_hist + 1]))
        shape_h = np.dot(legendre_matrix(era5_ts, deg_shape_hist),
                         params_hist[deg_scale_hist + 1: deg_scale_hist + deg_shape_hist + 2])
        alpha_h = link_func(era5_ts, params_hist[deg_scale_hist + deg_shape_hist + 2:],
                            opt_alpha_hist)

        scale_hn = np.exp(np.dot(legendre_matrix(era5_ts, deg_scale_histnat),
                                 params_histnat[: deg_scale_histnat + 1]))
        shape_hn = np.dot(legendre_matrix(era5_ts, deg_shape_histnat),
                          params_histnat[deg_scale_histnat + 1: deg_scale_histnat + deg_shape_histnat + 2])
        alpha_hn = link_func(era5_ts, params_histnat[deg_scale_histnat + deg_shape_histnat + 2:],
                             opt_alpha_histnat)

        phi_h = np.repeat(1 - self.q, len(scale_h))
        phi_hn = np.repeat(1 - self.q, len(scale_hn))

        ds = self._build_result_dataset(
            scale_h, scale_hn, shape_h, shape_hn, alpha_h, alpha_hn,
            phi_h, phi_hn, data_era5[0].time, "varying threshold"
        )
        parameter_plot(phi_h, phi_hn, scale_h, scale_hn, shape_h, shape_hn, alpha_h, alpha_hn)

        tag = f"{self.model_name}_{self.q}"
        ds.to_netcdf(f"{self.output_dir}/results_model2_{tag}.nc", engine="netcdf4")
        joint_hist.save(f"{self.output_dir}/model2_hist_{tag}.pkl")
        joint_histnat.save(f"{self.output_dir}/model2_histnat_{tag}.pkl")
        print(f"[{self.model_name}] Model 2 analysis finished.")
