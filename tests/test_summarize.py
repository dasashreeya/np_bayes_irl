import numpy as np
import pytest

from diagnostics import ess_bulk, split_rhat
from sampler import Trace
from summarize import binder_loss, minimize_binder, posterior_similarity, summarize


def _trace(assignments, log_post, weights=None):
    A = np.asarray(assignments)
    K = np.array([np.unique(z).size for z in A])
    if weights is None:
        weights = [[np.full(2, float(k)) for k in range(Kk)] for Kk in K]
    return Trace(A, weights, np.asarray(log_post, dtype=float), K,
                 np.full(len(A), 0.3), 0.0)


def test_psm_is_label_invariant():
    A = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
    psm = posterior_similarity(A)
    assert np.allclose(np.diag(psm), 1)
    assert np.isclose(psm[0, 1], 2 / 3) and np.isclose(psm[1, 2], 1 / 3)
    assert np.allclose(psm, psm.T)


def test_binder_recovers_dominant_partition_and_improves_on_candidates():
    truth = np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(0)
    A = np.tile(truth, (200, 1))
    for z in A[:80]:  # noise: move one random item to a new cluster
        z[rng.integers(6)] = 2
    psm = posterior_similarity(A)
    z = minimize_binder(psm, A[:80])  # the truth is not among the candidates
    assert np.array_equal(z, truth)
    assert binder_loss(z, psm) <= min(binder_loss(c, psm) for c in A[:80])


def test_rhat_and_ess_behave():
    rng = np.random.default_rng(1)
    iid = rng.standard_normal((4, 400))
    assert split_rhat(iid) < 1.02
    assert 1200 < ess_bulk(iid) < 2400
    shifted = iid + np.array([0, 0, 0, 3])[:, None]
    assert split_rhat(shifted) > 1.3
    assert split_rhat(np.full((4, 10), 2.0)) == 1.0
    assert split_rhat(np.repeat([[2.0], [3.0]], 10, axis=1)) == np.inf
    assert np.isnan(ess_bulk(np.full((4, 10), 2.0)))


def test_summarize_pools_chains():
    a = _trace([[0, 0, 1, 1]] * 6 + [[0, 1, 2, 3]] * 2, log_post=np.arange(8))
    b = _trace([[1, 1, 0, 0]] * 8, log_post=np.arange(8) + 0.5)
    s = summarize([a, b])
    assert s.n_chains == 2 and s.n_kept == 8
    assert np.array_equal(s.k_values, [2, 4])
    assert np.allclose(s.k_probs, [14 / 16, 2 / 16])
    assert np.array_equal(s.point_clustering, [0, 0, 1, 1])
    assert np.array_equal(s.map_assignments, [1, 1, 0, 0]) and s.map_log_post == 7.5
    assert len(s.point_weights) == 2
    assert set(s.rhat) == set(s.ess) == {"K", "log_post"}


def test_summarize_rejects_unequal_chains():
    with pytest.raises(ValueError):
        summarize([_trace([[0, 0]] * 4, range(4)), _trace([[0, 0]] * 5, range(5))])
