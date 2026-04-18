"""
eval.py  |  Owner: P2  |  Built: Day 5
All evaluation metrics.
  - L2 weight error (reward recovery)
  - Adjusted Rand Index (cluster recovery)
  - W&B logging
"""
import jax.numpy as jnp
import numpy as np

def l2_weight_error(w_true, w_recovered):
    """L2 distance between normalized weight vectors."""
    # TODO Step 12 Day 5 -- P2
    pass

def best_match_error(true_weights, recovered_weights):
    """Match each true weight to closest recovered. Returns mean L2."""
    # TODO Step 12 Day 5 -- P2
    pass

def adjusted_rand_index(true_z, pred_z):
    """Cluster assignment accuracy. 0=random, 1=perfect."""
    # TODO Step 12 Day 5 -- P2
    pass

def log_metrics(sweep, state, true_weights, true_z, wandb_run=None):
    """Print + log to W&B: n_clusters, weight_l2, ARI."""
    # TODO Step 12 Day 5 -- P2
    pass
