"""
reward_features.py  |  Owner: P2  |  Built: Day 2
Utility functions for linear reward: R(s,a) = w . phi(s).
"""
import jax.numpy as jnp

def compute_reward(w, phi):
    """R(s) = w . phi(s). Returns (n_states,)."""
    # TODO Step 6 Day 2 -- P2
    pass

def reward_sa(w, phi, n_actions):
    """Expand to (n_states, n_actions) -- same reward for all actions."""
    # TODO Step 6 Day 2 -- P2
    pass

def l2_weight_error(w_true, w_recovered):
    """L2 distance between normalized weight vectors."""
    # TODO Step 6 Day 2 -- P2
    pass

def normalize_weights(w):
    """Normalize to unit norm."""
    # TODO Step 6 Day 2 -- P2
    pass
