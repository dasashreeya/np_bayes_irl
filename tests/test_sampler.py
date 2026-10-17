"""
Exactness tests for sampler.run_chain on a model whose posterior over
partitions is known in closed form.

Toy model: 1-D weights, w_k ~ N(0, 1), and "trajectory" i is a scalar
x_i ~ N(w_k, 1). Integrating w out, a cluster S has marginal
x_S ~ N(0, I + 11^T), so for N = 4 the exact posterior over all 15
partitions is CRP(partition) * prod_S N(x_S; 0, I + 11^T). Tempering by tau
gives N(x_S; 0, I / tau + 11^T) up to a factor that is constant across
partitions.
"""
import json
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from sampler import SamplerConfig, Trace, log_crp, run_chain
from summarize import _relabel


class GaussianBackend:
    """Implements the LikelihoodBackend methods the sampler uses, for scalar data."""

    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)
        self.N, self.n_features = self.x.size, 1

    def loglik_matrix(self, weights):
        if len(weights) == 0:
            return np.zeros((self.N, 0))
        w = np.array([np.asarray(v, dtype=float)[0] for v in weights])
        return -0.5 * (self.x[:, None] - w[None, :]) ** 2 - 0.5 * np.log(2 * np.pi)

    def loglik_cluster(self, w, idx):
        return float(self.loglik_matrix([w])[np.asarray(idx, dtype=int), 0].sum())

    def invalidate(self, weight_id):
        return None


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        yield [[first]] + p
        for j in range(len(p)):
            yield p[:j] + [[first] + p[j]] + p[j + 1:]


def exact_posterior(x, alpha, tau):
    N = len(x)
    out = {}
    for p in set_partitions(list(range(N))):
        z = np.empty(N, dtype=int)
        for k, block in enumerate(p):
            z[block] = k
        lp = log_crp([len(b) for b in p], alpha)
        if tau > 0:
            for b in p:
                n = len(b)
                cov = np.eye(n) / tau + np.ones((n, n))
                lp += multivariate_normal(np.zeros(n), cov).logpdf(x[b])
        out[tuple(_relabel(z))] = lp
    keys = list(out)
    lps = np.array([out[k] for k in keys])
    probs = np.exp(lps - lps.max())
    return dict(zip(keys, probs / probs.sum()))


def empirical(assignments):
    keys, counts = np.unique(np.array([_relabel(z) for z in assignments]), axis=0,
                             return_counts=True)
    return {tuple(k): c / counts.sum() for k, c in zip(keys, counts)}


X = np.array([-1.8, -1.2, 1.5, 2.1])


@pytest.mark.parametrize("tau,alpha", [(1.0, 1.0), (0.5, 2.0), (0.0, 1.5)])
def test_partition_posterior_is_exact(tau, alpha):
    exact = exact_posterior(X, alpha, tau)
    assert len(exact) == 15
    cfg = SamplerConfig(alpha=alpha, n_sweeps=25000, burn_in=500, step_size=1.0,
                        n_mh_steps=3, tau=tau, adapt=False)
    tr = run_chain(GaussianBackend(X), cfg, rng_key=1)
    emp = empirical(tr.assignments)
    tv = 0.5 * sum(abs(emp.get(k, 0.0) - p) for k, p in exact.items())
    assert set(emp) <= set(exact)
    assert tv < 0.03, f"total variation to the exact posterior = {tv:.3f}"


def test_weights_follow_conditional_posterior():
    # With everything forced into one cluster (alpha tiny), w | x ~ N(sum x / (N+1), 1 / (N+1)).
    cfg = SamplerConfig(alpha=1e-8, n_sweeps=20000, burn_in=500, step_size=0.8,
                        init="one_cluster", adapt=False)
    tr = run_chain(GaussianBackend(X), cfg, rng_key=2)
    assert np.all(tr.K == 1)
    w = np.array([ws[0][0] for ws in tr.weights])
    assert abs(w.mean() - X.sum() / 5) < 0.05
    assert abs(w.var() - 1 / 5) < 0.03


def test_log_post_matches_model_terms():
    cfg = SamplerConfig(alpha=1.3, n_sweeps=30, burn_in=10, adapt=False)
    be = GaussianBackend(X)
    tr = run_chain(be, cfg, rng_key=3)
    for z, ws, lp in zip(tr.assignments, tr.weights, tr.log_post):
        counts = np.bincount(z)
        L = be.loglik_matrix(ws)
        manual = (log_crp(counts, 1.3)
                  + sum(-0.5 * float(w @ w) - 0.5 * np.log(2 * np.pi) for w in ws)
                  + L[np.arange(X.size), z].sum())
        assert np.isclose(lp, manual)
        assert counts.min() > 0 and len(ws) == counts.size


def test_trace_roundtrip_and_shapes():
    cfg = SamplerConfig(n_sweeps=20, burn_in=5, thin=3, adapt=False)
    tr = run_chain(GaussianBackend(X), cfg, rng_key=4)
    assert tr.assignments.shape == (5, X.size)  # sweeps 5, 8, 11, 14, 17
    assert len(tr.weights) == len(tr.log_post) == len(tr.K) == len(tr.accept_rate) == 5
    back = Trace.from_dict(json.loads(json.dumps(tr.to_dict())))
    assert np.array_equal(back.assignments, tr.assignments)
    assert np.allclose(back.log_post, tr.log_post)
    assert back.config == cfg.to_dict()


def test_same_seed_same_chain():
    cfg = SamplerConfig(n_sweeps=40, burn_in=10, adapt=False)
    a = run_chain(GaussianBackend(X), cfg, rng_key=7)
    b = run_chain(GaussianBackend(X), cfg.to_dict(), rng_key=7)
    assert np.array_equal(a.assignments, b.assignments)
    assert np.array_equal(a.log_post, b.log_post)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SamplerConfig.from_dict({"alpah": 1.0})


def test_adaptation_hits_target_and_keeps_posterior_exact():
    # Start far too small; burn-in adaptation should bring acceptance near 0.23,
    # and since the step is frozen afterwards the partition posterior stays exact.
    exact = exact_posterior(X, 1.0, 1.0)
    cfg = SamplerConfig(alpha=1.0, n_sweeps=25000, burn_in=2000, step_size=1e-3,
                        n_mh_steps=3, adapt=True, target_accept=0.23)
    tr = run_chain(GaussianBackend(X), cfg, rng_key=5)
    assert tr.step_size > 0.5
    assert abs(tr.accept_rate.mean() - 0.23) < 0.06
    emp = empirical(tr.assignments)
    assert 0.5 * sum(abs(emp.get(k, 0.0) - p) for k, p in exact.items()) < 0.03


def test_no_adaptation_after_burn_in():
    cfg = SamplerConfig(n_sweeps=30, burn_in=10, step_size=0.123, adapt=False)
    assert run_chain(GaussianBackend(X), cfg, rng_key=6).step_size == 0.123
