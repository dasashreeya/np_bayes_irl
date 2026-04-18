"""
dp_prior.py  |  Owner: P1  |  Built: Day 3
Dirichlet Process base measure G0.
New reward atoms are sampled from G0 = N(0, I) over weight vectors.
"""
import jax
import jax.numpy as jnp

def sample_new_weights(rng_key, n_features):
    """Sample a new reward weight vector from G0 = N(0, I). Returns (n_features,)."""
    # TODO Step 7 Day 3 -- P1
    pass

def log_prior(w):
    """Log P(w) under G0 = N(0, I). Used in MH acceptance ratio."""
    # TODO Step 7 Day 3 -- P1
    pass
