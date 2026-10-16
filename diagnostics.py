"""
diagnostics.py  |  Owner: SA
MCMC convergence diagnostics on scalar chains (Vehtari et al. 2021).

Used on label-invariant scalars only: K and the joint log-posterior.
"""

import numpy as np
from scipy.stats import norm, rankdata


def _split(chains):
    """(M, n) -> (2M, n // 2): each chain cut in half, dropping the middle draw if n is odd."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2 or chains.shape[1] < 4:
        raise ValueError("need a (n_chains, n_draws) array with n_draws >= 4")
    h = chains.shape[1] // 2
    return np.concatenate([chains[:, :h], chains[:, -h:]], axis=0)


def _rank_normalize(chains):
    """Pooled average ranks -> normal scores (Blom offset), same shape."""
    flat = np.asarray(chains, dtype=float).ravel()
    r = rankdata(flat, method="average")
    z = norm.ppf((r - 0.375) / (flat.size + 0.25))
    return z.reshape(np.shape(chains))


def _constant(chains):
    """None if the chains vary; otherwise the R-hat to report for constant input."""
    chains = np.asarray(chains, dtype=float)
    if np.ptp(chains) > 0:
        if np.all(np.ptp(chains, axis=1) == 0):
            return np.inf  # each chain stuck at a different value
        return None
    return 1.0  # every draw identical: the chains trivially agree


def _rhat_raw(chains):
    m, n = chains.shape
    W = chains.var(axis=1, ddof=1).mean()
    B_over_n = chains.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B_over_n
    return float(np.sqrt(var_plus / W))


def split_rhat(chains):
    """
    Rank-normalized split R-hat: the max of the bulk and folded versions
    (Vehtari et al. 2021, eqs. 4-5).

    Args:
        chains: (n_chains, n_draws) array of one scalar
    Returns:
        float; values below 1.05 suggest the chains agree. 1.0 if every draw
        is identical; inf if each chain is constant at a different value.
    """
    c = _constant(chains)
    if c is not None:
        return float(c)
    s = _split(chains)
    bulk = _rhat_raw(_rank_normalize(s))
    folded_in = np.abs(s - np.median(s))
    folded = _rhat_raw(_rank_normalize(folded_in)) if np.ptp(folded_in) > 0 else 1.0
    return max(bulk, folded)


def _autocov(x):
    """Biased autocovariance of each row via FFT. (M, n) -> (M, n)."""
    m, n = x.shape
    x = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=2 * n, axis=1)
    return np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n


def _ess_raw(chains):
    """Multi-chain ESS with Geyer's initial monotone sequence (Stan's estimator)."""
    m, n = chains.shape
    acov = _autocov(chains)
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n + chains.mean(axis=1).var(ddof=1)
    rho = np.zeros(n)
    rho[0] = 1.0
    even, odd = 1.0, 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = odd
    t = 1
    while t < n - 3 and even + odd > 0:
        even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if even + odd >= 0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:  # Stan's variance reduction for antithetic chains
        rho[max_t + 1] = even
    t = 1  # enforce monotone decrease of the paired sums
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2
        t += 2
    tau = -1.0 + 2.0 * rho[:max_t + 1].sum() + rho[max_t + 1]
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def ess_bulk(chains):
    """
    Bulk effective sample size: rank-normalized, split chains, Geyer's
    initial monotone sequence estimator.

    Args:
        chains: (n_chains, n_draws) array of one scalar
    Returns:
        float, ESS pooled over all chains; nan if every draw is identical
        or each chain is constant
    """
    if _constant(chains) is not None:
        return float("nan")
    return _ess_raw(_rank_normalize(_split(chains)))
