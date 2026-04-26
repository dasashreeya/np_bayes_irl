"""
reward_features.py  |  Owner: P2  |  Built: Day 2
Utility functions for linear reward: R(s,a) = w . phi(s).
"""
import jax.numpy as jnp

def compute_reward(w, phi):
    """R(s) = w . phi(s). Returns (n_states,)."""
    return phi @ w

def reward_sa(w, phi, n_actions):
    """Expand to (n_states, n_actions) -- same reward for all actions."""
    R = compute_reward(w, phi)          # (n_states,)
    return jnp.tile(R[:, None], (1, n_actions))  # (n_states, n_actions)

def normalize_weights(w):
    """Normalize to unit norm."""
    norm = jnp.linalg.norm(w)
    return w / jnp.where(norm > 0, norm, 1.0)

def l2_weight_error(w_true, w_recovered):
    """L2 distance between normalized weight vectors."""
    return float(jnp.linalg.norm(normalize_weights(w_true) - normalize_weights(w_recovered)))
