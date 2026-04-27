"""
dp_prior.py — Dirichlet Process prior for NP-Bayes IRL
=======================================================
Person A · Day 5

The base distribution G0 = N(0, I) over reward weight vectors.

Two functions
-------------
  sample_new_weights(rng, n_features)  →  w ~ N(0, I),  shape (n_features,)
  log_prior(w)                         →  log N(0, I) = -0.5 * ||w||²  (scalar)

Gates
-----
  ✓ sample_new_weights: shape (16,) · over 1000 samples: mean≈0, std≈1
  ✓ log_prior(zeros) > log_prior(5·ones)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Base distribution G0 = N(0, I)
# ---------------------------------------------------------------------------

# def sample_new_weights(
#     rng: np.random.Generator,
#     n_features: int = 16,
# ) -> np.ndarray:
#     """
#     Sample a new reward weight vector from the DP base distribution G0.

#     G0 = N(0, I)  — isotropic Gaussian over R^n_features

#     In the CRP interpretation this is called when a trajectory is assigned
#     to a *new* table: we draw a fresh reward type from the prior.

#     Parameters
#     ----------
#     rng        : np.random.Generator   (e.g. np.random.default_rng(42))
#     n_features : int                   number of reward features (default 16)

#     Returns
#     -------
#     w : ndarray, shape (n_features,)
#     """
#     return rng.standard_normal(n_features).astype(np.float64)

def sample_new_weights(rng_key, n_features):
    return jax.random.normal(rng_key, shape=(n_features,))


# ---------------------------------------------------------------------------
# Log-prior  log G0(w) = log N(w | 0, I)
# ---------------------------------------------------------------------------

def log_prior(w: np.ndarray) -> float:
    """
    Unnormalised log-prior under G0 = N(0, I).

    log N(w | 0, I)  =  -0.5 * ||w||²  +  const

    The normalisation constant  -0.5 * d * log(2π)  is dropped because it
    cancels in every MH acceptance ratio and CRP score comparison.

    Parameters
    ----------
    w : ndarray, shape (n_features,)

    Returns
    -------
    float  — log-prior value (always ≤ 0 for the unnormalised form)
    """
    return -0.5 * float(np.dot(w, w))


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Gate 1: shape (16,) and 1000-sample statistics ---
    n_features = 16
    samples = np.stack([sample_new_weights(rng, n_features) for _ in range(1_000)])

    assert samples.shape == (1_000, n_features), f"Bad shape: {samples.shape}"

    sample_mean = samples.mean()
    sample_std  = samples.std()
    assert abs(sample_mean) < 0.1,          f"Mean too far from 0: {sample_mean:.4f}"
    assert abs(sample_std  - 1.0) < 0.1,   f"Std too far from 1: {sample_std:.4f}"

    print(f"✓ sample_new_weights: shape {samples[0].shape}")
    print(f"  1000-sample mean = {sample_mean:.4f}  (expected ≈ 0)")
    print(f"  1000-sample std  = {sample_std:.4f}  (expected ≈ 1)")

    # --- Gate 2: log_prior(0) > log_prior(5·ones) ---
    w_zero   = np.zeros(n_features)
    w_far    = 5.0 * np.ones(n_features)

    lp_zero  = log_prior(w_zero)
    lp_far   = log_prior(w_far)

    assert lp_zero > lp_far, \
        f"log_prior(0)={lp_zero:.2f} should be > log_prior(5·ones)={lp_far:.2f}"

    print(f"\n✓ log_prior(zeros)   = {lp_zero:.2f}   (= 0, unnormalised peak)")
    print(f"  log_prior(5·ones)  = {lp_far:.2f}  (far from prior)")
    print(f"  log_prior(0) > log_prior(5·ones): {lp_zero > lp_far}")

    # Bonus: monotone sanity check across distances
    w_close  = 1.0 * np.ones(n_features)
    w_medium = 2.0 * np.ones(n_features)
    assert log_prior(w_zero)  > log_prior(w_close)
    assert log_prior(w_close) > log_prior(w_medium)
    assert log_prior(w_medium)> log_prior(w_far)
    print(f"\n✓ Monotone: log_prior decreases as ||w|| grows")

    print("\nAll gates passed ✓")