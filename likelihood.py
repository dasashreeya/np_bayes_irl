"""
likelihood.py  |  Owner: P2  |  Built: Day 3
Boltzmann likelihood: log P(trajectories | w).
This is the most important function in the project.
Gets called thousands of times inside the Gibbs loop.
"""
"""
likelihood.py  |  Owner: P2  |  Built: Day 3
Computes log P(trajectory | w) — the core of the Gibbs sampler.
Most-called function in the entire inference pipeline.
"""

import jax
import jax.numpy as jnp
from functools import partial

from mdp_utils import soft_value_iteration, log_boltzmann_policy
from reward_features import reward_sa


# ── JIT-compiled inner function ──────────────────────────────────────────────
# This is the hot path. Called thousands of times per experiment.
# JIT is applied here, not to the outer likelihood loop.
# phi and T are arrays (not static); gamma, beta are static scalars.

@partial(jax.jit, static_argnames=("gamma", "beta"))
def compute_log_pi(w, phi, T, gamma=0.95, beta=1.0):
    """
    JIT-compiled: reward → value iteration → log policy.
    Returns the full log policy matrix for a given weight vector.

    Args:
        w:     (n_features,)
        phi:   (n_states, n_features)
        T:     (n_states, n_actions, n_states)
        gamma: float  discount factor (static)
        beta:  float  temperature (static)

    Returns:
        log_pi: (n_states, n_actions)
    """
    n_actions = T.shape[1]
    R = reward_sa(w, phi, n_actions)                    # (n_states, n_actions)
    V, Q = soft_value_iteration(R, T, gamma, beta)      # V: (100,)  Q: (100,4)
    return log_boltzmann_policy(Q, beta)                # (n_states, n_actions)


# ── 1. log_likelihood ────────────────────────────────────────────────────────

def log_likelihood(w, trajectories, phi, T, gamma=0.95, beta=1.0):
    """
    Log P(trajectories | w) under the Boltzmann expert model.
    Sums log pi(a|s) over every (s,a) step in every trajectory.

    Args:
        w:            (n_features,)  reward weight vector
        trajectories: list of trajectories, each a list of (state, action) tuples
        phi:          (n_states, n_features)
        T:            (n_states, n_actions, n_states)
        gamma:        float  discount factor
        beta:         float  Boltzmann temperature

    Returns:
        total: float  scalar log likelihood (negative, finite)
    """
    log_pi = compute_log_pi(w, phi, T, gamma, beta)   # (n_states, n_actions)

    total = 0.0
    for traj in trajectories:
        for (s, a) in traj:
            total = total + log_pi[s, a]

    return total


# ── 2. log_likelihood_single ─────────────────────────────────────────────────

def log_likelihood_single(w, traj, phi, T, gamma=0.95, beta=1.0):
    """
    Convenience wrapper — log likelihood of a single trajectory.
    Called by sample_cluster_assignment in gibbs.py for each trajectory.

    Args:
        w:     (n_features,)
        traj:  list of (state, action) tuples
        phi:   (n_states, n_features)
        T:     (n_states, n_actions, n_states)
        gamma: float
        beta:  float

    Returns:
        float  scalar log likelihood
    """
    return log_likelihood(w, [traj], phi, T, gamma, beta)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import jax
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset

    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()

    trajs, true_weights, true_z = generate_dataset(env, phi, T)

    w1       = true_weights[0]   # red-lover
    w2       = true_weights[1]   # blue-lover
    trajs_t1 = trajs[:10]        # red-lover trajectories
    trajs_t2 = trajs[10:]        # blue-lover trajectories

    # ── Gate check 1: finite negative scalar ─────────────────────────────────
    ll = log_likelihood(w1, trajs_t1, phi, T)
    print(f"ll(w1, type1 trajs):  {ll:.2f}")
    assert jnp.isfinite(ll) and ll < 0, "FAIL: expected finite negative scalar"
    print("PASS: finite negative scalar")

    # ── Gate check 2: true weights score higher than random ──────────────────
    rng      = jax.random.PRNGKey(0)
    w_random = jax.random.normal(rng, shape=(16,))

    ll_true   = log_likelihood(w1, trajs_t1, phi, T)
    ll_random = log_likelihood(w_random, trajs_t1, phi, T)
    print(f"\nll_true:   {ll_true:.2f}")
    print(f"ll_random: {ll_random:.2f}")
    assert ll_true > ll_random, "FAIL: true weights should score higher than random"
    print("PASS: true weights score higher than random")

    # ── Gate check 3: matched type scores higher than mismatched ─────────────
    ll_matched   = log_likelihood(w1, trajs_t1, phi, T)
    ll_mismatched = log_likelihood(w2, trajs_t1, phi, T)
    print(f"\nll_matched:    {ll_matched:.2f}   (w1 on type-1 trajs)")
    print(f"ll_mismatched: {ll_mismatched:.2f}  (w2 on type-1 trajs)")
    assert ll_matched > ll_mismatched, "FAIL: matched type should score higher"
    print("PASS: matched reward type scores higher than mismatched")

    # ── Gate check 4: not -inf or NaN ────────────────────────────────────────
    assert not jnp.isinf(ll), "FAIL: -inf detected (zero-probability action)"
    assert not jnp.isnan(ll), "FAIL: NaN detected (reward normalization needed)"
    print("\nPASS: no -inf or NaN")

    # ── Gate check 5: single trajectory wrapper ───────────────────────────────
    ll_single = log_likelihood_single(w1, trajs_t1[0], phi, T)
    ll_batch  = log_likelihood(w1, [trajs_t1[0]], phi, T)
    assert abs(float(ll_single) - float(ll_batch)) < 1e-5, "FAIL: single != batch"
    print("PASS: log_likelihood_single matches log_likelihood([traj])")

    # ── Day 3 handoff test ────────────────────────────────────────────────────
    print(f"\n--- Day 3 Handoff ---")
    print(f"ll_true:   {ll_true:.2f}")
    print(f"ll_random: {ll_random:.2f}")
    print(f"True scores higher: {ll_true > ll_random}")

    print("\nAll gate checks passed.")