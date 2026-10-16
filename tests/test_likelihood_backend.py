"""The reference LikelihoodBackend must agree with likelihood.log_likelihood."""
import numpy as np
import jax
import pytest


@pytest.fixture(scope="module")
def data():
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset
    env = ObjectWorld()
    phi, T = env.features(), env.transitions()
    trajs, true_w, _ = generate_dataset(env, phi, T, n_per_type=3)
    return phi, T, trajs, true_w


def test_backend_matches_log_likelihood(data):
    from likelihood import log_likelihood
    from likelihood_backend import LikelihoodBackend
    phi, T, trajs, true_w = data
    be = LikelihoodBackend(phi, T, 0.95, 5.0, trajs)
    ws = list(true_w) + [jax.random.normal(jax.random.PRNGKey(1), (phi.shape[1],))]
    L = be.loglik_matrix(ws)
    assert L.shape == (len(trajs), len(ws))
    for k, w in enumerate(ws):
        for i, t in enumerate(trajs):
            ref = float(log_likelihood(w, [t], phi, T, 0.95, 5.0))
            assert abs(L[i, k] - ref) < 1e-4 * max(1.0, abs(ref))
        idx = [0, 2, 4]
        assert np.isclose(be.loglik_cluster(w, idx), L[idx, k].sum())
    assert be.loglik_cluster(ws[0], []) == 0.0
    assert be.loglik_matrix([]).shape == (len(trajs), 0)


def test_backend_handles_unequal_lengths(data):
    from likelihood_backend import LikelihoodBackend
    phi, T, trajs, true_w = data
    short = [trajs[0][:5], trajs[1], trajs[2][:1]]
    be = LikelihoodBackend(phi, T, 0.95, 5.0, short)
    lp = be.log_pi(true_w[0])
    expect = [sum(lp[s, a] for s, a in t) for t in short]
    assert np.allclose(be.loglik_matrix([true_w[0]])[:, 0], expect)
