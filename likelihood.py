"""
likelihood.py  |  Owner: P2  |  Built: Day 3, Updated: Day 12
Boltzmann likelihood: log P(trajectories | w).
This is the most important function in the project.
Gets called thousands of times inside the Gibbs loop.
"""

import time
import jax
import jax.numpy as jnp
from functools import partial

from mdp_utils import soft_value_iteration, log_boltzmann_policy
from reward_features import reward_sa


# ── JIT-compiled inner function ──────────────────────────────────────────────

@partial(jax.jit, static_argnames=("gamma", "beta"))
def compute_log_pi(w, phi, T, gamma=0.95, beta=1.0):
    """
    JIT-compiled: reward → value iteration → log policy.

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
    R = reward_sa(w, phi, n_actions)
    V, Q = soft_value_iteration(R, T, gamma, beta)
    return log_boltzmann_policy(Q, beta)


# ── 1. log_likelihood ────────────────────────────────────────────────────────

def log_likelihood(w, trajectories, phi, T, gamma=0.95, beta=1.0):
    """
    Log P(trajectories | w) under the Boltzmann expert model.

    Args:
        w:            (n_features,)
        trajectories: list of trajectories, each a list of (state, action) tuples
        phi:          (n_states, n_features)
        T:            (n_states, n_actions, n_states)
        gamma:        float
        beta:         float

    Returns:
        total: float  scalar log likelihood (negative, finite)
    """
    log_pi = compute_log_pi(w, phi, T, gamma, beta)

    total = 0.0
    for traj in trajectories:
        for (s, a) in traj:
            total = total + log_pi[s, a]

    return total


# ── 2. log_likelihood_single ─────────────────────────────────────────────────

def log_likelihood_single(w, traj, phi, T, gamma=0.95, beta=1.0):
    """
    Convenience wrapper — log likelihood of a single trajectory.
    Called by sample_cluster_assignment in gibbs.py.
    """
    return log_likelihood(w, [traj], phi, T, gamma, beta)


# ── 3. jit_speed_gate ────────────────────────────────────────────────────────

def jit_speed_gate(w, phi, T, gamma=0.95, beta=1.0):
    """
    Gate: 2nd call must be faster than 1st, values must match.
    Call this once at startup to verify JIT is working.
    """
    # 1st call — triggers compilation
    t0 = time.perf_counter()
    out1 = compute_log_pi(w, phi, T, gamma, beta)
    out1.block_until_ready()
    t1 = time.perf_counter()

    # 2nd call — uses compiled version
    out2 = compute_log_pi(w, phi, T, gamma, beta)
    out2.block_until_ready()
    t2 = time.perf_counter()

    first  = t1 - t0
    second = t2 - t1

    print(f"JIT 1st call : {first:.4f}s")
    print(f"JIT 2nd call : {second:.4f}s")
    assert second < first,                        "FAIL: 2nd call not faster — JIT not working"
    assert jnp.allclose(out1, out2, atol=1e-5),  "FAIL: values differ between calls"
    print("JIT gate ✓ — 2nd call faster, values match")


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset

    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()

    trajs, true_weights, true_z = generate_dataset(env, phi, T)

    w1        = true_weights[0]
    w2        = true_weights[1]
    trajs_t1  = trajs[:10]
    trajs_t2  = trajs[10:]

    # ── Day 12 gate: JIT speed ────────────────────────────────────────────────
    print("--- Day 12 JIT Gate ---")
    jit_speed_gate(w1, phi, T)

    # ── Gate 1: finite negative scalar ───────────────────────────────────────
    ll = log_likelihood(w1, trajs_t1, phi, T)
    assert jnp.isfinite(ll) and ll < 0
    print(f"PASS: finite negative scalar  ({ll:.2f})")

    # ── Gate 2: true weights score higher than random ─────────────────────────
    w_random  = jax.random.normal(jax.random.PRNGKey(0), shape=(phi.shape[1],))
    ll_true   = log_likelihood(w1, trajs_t1, phi, T)
    ll_random = log_likelihood(w_random, trajs_t1, phi, T)
    assert ll_true > ll_random
    print(f"PASS: true > random  ({ll_true:.2f} > {ll_random:.2f})")

    # ── Gate 3: matched type scores higher than mismatched ────────────────────
    ll_matched    = log_likelihood(w1, trajs_t1, phi, T)
    ll_mismatched = log_likelihood(w2, trajs_t1, phi, T)
    assert ll_matched > ll_mismatched
    print(f"PASS: matched > mismatched  ({ll_matched:.2f} > {ll_mismatched:.2f})")

    # ── Gate 4: no -inf or NaN ────────────────────────────────────────────────
    assert not jnp.isinf(ll) and not jnp.isnan(ll)
    print("PASS: no -inf or NaN")

    # ── Gate 5: single == batch ───────────────────────────────────────────────
    ll_single = log_likelihood_single(w1, trajs_t1[0], phi, T)
    ll_batch  = log_likelihood(w1, [trajs_t1[0]], phi, T)
    assert abs(float(ll_single) - float(ll_batch)) < 1e-5
    print("PASS: log_likelihood_single matches log_likelihood([traj])")

    print("\nAll gates passed.")