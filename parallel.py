"""
parallel.py  |  Owner: P2 (leads) + P1 (pure function requirement)  |  Built: Day 6
Data-parallel MCMC via Ray.
Splits trajectories across workers. Each worker runs one Gibbs sweep.
Merges sufficient statistics across workers.
THIS IS THE CONTRIBUTION -- the merge step is what you show your professor.
"""
import ray
import jax
import jax.numpy as jnp
import numpy as np
import time
from gibbs import gibbs_sweep
from dp_prior import sample_new_weights


@ray.remote
def worker_sweep(traj_chunk, state, phi, T, alpha, gamma, beta, step_size, seed):
    """Ray worker: runs one Gibbs sweep on trajectory chunk."""
    # TODO Step 15 Day 6 -- P2
    pass


def merge_states(states, n_features):
    """
    Merge cluster states from multiple workers.
    This is the academic contribution -- document what you merge and how.
    """
    # TODO Step 15 Day 6 -- P2
    pass


def run_parallel(trajectories, phi, T, n_workers=4, n_sweeps=500,
                 alpha=1.0, gamma=0.95, beta=1.0, step_size=0.1, seed=0):
    """Full parallel training run. Returns merged state and per-sweep times."""
    # TODO Step 15 Day 6 -- P2
    pass
