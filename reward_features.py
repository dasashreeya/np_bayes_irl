"""
reward_features.py  |  Owner: P2  |  Built: Day 2
Utility functions for linear reward: R(s,a) = w . phi(s).
"""
"""
reward_features.py  |  Owner: P2  |  Built: Day 2
Utility functions for feature-based reward computation.
Used by likelihood.py (reward_sa) and eval.py (l2_weight_error, normalize_weights).
"""

import jax.numpy as jnp


# ── 1. compute_reward ────────────────────────────────────────────────────────

def compute_reward(w, phi):
    """
    State reward vector — dot product of weights with feature map.

    Args:
        w:   (n_features,)          reward weight vector
        phi: (n_states, n_features) feature matrix

    Returns:
        R: (n_states,)  one reward value per state
    """
    return phi @ w


# ── 2. reward_sa ─────────────────────────────────────────────────────────────

def reward_sa(w, phi, n_actions):
    """
    Expand state reward to (n_states, n_actions).
    Reward is the same for all actions — only the state matters.
    Required shape for soft_value_iteration in mdp_utils.py.

    Args:
        w:        (n_features,)
        phi:      (n_states, n_features)
        n_actions: int

    Returns:
        R: (n_states, n_actions)
    """
    R = compute_reward(w, phi)                    # (n_states,)
    return jnp.tile(R[:, None], (1, n_actions))   # (n_states, n_actions)


# ── 3. l2_weight_error ───────────────────────────────────────────────────────

def l2_weight_error(w_true, w_recovered):
    """
    Euclidean distance between true and recovered weight vectors.
    Raw (not normalized) — used for quick sanity checks.
    For fair comparison across scale, use normalize_weights() first.

    Args:
        w_true:      (n_features,)
        w_recovered: (n_features,)

    Returns:
        float  scalar L2 distance
    """
    return float(jnp.linalg.norm(w_true - w_recovered))


# ── 4. normalize_weights ─────────────────────────────────────────────────────

def normalize_weights(w):
    """
    Normalize weight vector to unit norm.
    The Gibbs sampler recovers rewards up to a scale factor —
    normalization is required before computing L2 error in eval.py.

    Args:
        w: (n_features,)

    Returns:
        w_normalized: (n_features,)  unit norm
    """
    norm = jnp.linalg.norm(w)
    return w / (norm + 1e-8)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import jax.numpy as jnp
    from objectworld import ObjectWorld

    env = ObjectWorld()
    phi = env.features()    # (100, 16)
    w   = jnp.zeros(16).at[0].set(1.0).at[1].set(-1.0)

    # Gate check 1: compute_reward → shape (100,)
    R = compute_reward(w, phi)
    assert R.shape == (100,), f"FAIL: expected (100,), got {R.shape}"
    print(f"PASS: compute_reward shape {R.shape}")

    # Gate check 2: reward_sa → shape (100, 4)
    R_sa = reward_sa(w, phi, 4)
    assert R_sa.shape == (100, 4), f"FAIL: expected (100,4), got {R_sa.shape}"
    print(f"PASS: reward_sa shape {R_sa.shape}")

    # Gate check 3: all rows of R_sa are identical (same reward for all actions)
    assert jnp.allclose(R_sa[:, 0], R_sa[:, 1]), "FAIL: rows not identical across actions"
    print("PASS: reward identical across all actions")

    # Gate check 4: l2_weight_error of identical vectors = 0
    err = l2_weight_error(w, w)
    assert err < 1e-6, f"FAIL: expected ~0, got {err}"
    print(f"PASS: l2_weight_error(w, w) = {err:.6f}")

    # Gate check 5: normalize_weights → unit norm
    w_norm = normalize_weights(w)
    norm = float(jnp.linalg.norm(w_norm))
    assert abs(norm - 1.0) < 1e-5, f"FAIL: expected norm 1.0, got {norm}"
    print(f"PASS: normalize_weights → norm {norm:.6f}")

    print("\nAll gate checks passed.")
