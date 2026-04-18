"""
gibbs.py  |  Owner: P1 (step 1) + P2 (step 2) + Both (loop)  |  Built: Day 4
Collapsed Gibbs sampler for NP-Bayes IRL.
Two steps per sweep:
  1. CRP cluster assignment (P1)  -- sample z_i for each trajectory
  2. MH reward update (P2)        -- sample w_k for each cluster
"""
import jax
import jax.numpy as jnp
import numpy as np
from dp_prior import sample_new_weights, log_prior
from likelihood import log_likelihood, log_likelihood_single


def sample_cluster_assignment(i, trajectories, assignments, weight_vectors,
                               alpha, phi, T, gamma, beta, rng_key):
    """
    CRP cluster assignment for trajectory i.
    P(z_i=k)   proportional to  n_{-i,k} * P(traj_i | w_k)
    P(z_i=new) proportional to  alpha    * P(traj_i | w_new)
    CRITICAL: exclude trajectory i when counting n_{-i,k}
    """
    # TODO Step 9 Day 4 -- P1
    pass


def update_weight_vector(k, trajectories, assignments, w_k,
                          phi, T, gamma, beta, rng_key, step_size=0.1):
    """
    MH update for reward weight vector of cluster k.
    Propose w_proposed = w_k + step_size * N(0,I)
    Accept/reject by log likelihood ratio.
    Target acceptance rate: 20-50%
    """
    # TODO Step 10 Day 4 -- P2
    pass


def gibbs_sweep(trajectories, state, phi, T,
                alpha=1.0, gamma=0.95, beta=1.0,
                step_size=0.1, rng_key=None):
    """
    One full Gibbs sweep.
    state = {'assignments': [...], 'weight_vectors': [...]}
    Returns new state. PURE FUNCTION -- no global state, no side effects.
    """
    # TODO Step 11 Day 4 -- Both
    pass


def run_gibbs(trajectories, phi, T, n_sweeps=500, alpha=1.0,
              gamma=0.95, beta=1.0, step_size=0.1, seed=0):
    """Full training run. Returns final state and history."""
    # TODO Step 11 Day 4 -- Both
    pass
