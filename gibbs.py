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
    # Sample cluster assignment for trajectory i using CRP formula
    # P(z_i = k)   proportional to  n_{-i,k} * P(traj_i | w_k)
    # P(z_i = new) proportional to  alpha    * P(traj_i | w_new)
    N = len(trajectories)
    K = len(weight_vectors)
    traj_i = trajectories[i]

    # Count assignments EXCLUDING trajectory i
    counts = np.zeros(K, dtype=float)
    for j in range(N):
        if j != i:  # CRITICAL: exclude i
            counts[assignments[j]] += 1

    log_scores = []
    cluster_ids = []

    # Score existing clusters
    for k in range(K):
        if counts[k] == 0:
            continue  # skip empty clusters
        log_prior_k = np.log(counts[k])
        log_lik_k   = float(log_likelihood_single(
                        weight_vectors[k], traj_i, phi, T, gamma, beta))
        log_scores.append(log_prior_k + log_lik_k)
        cluster_ids.append(k)

    # Score a new cluster
    rng_key, sk = jax.random.split(rng_key)
    w_new = sample_new_weights(sk, phi.shape[1])
    log_lik_new = float(log_likelihood_single(
                    w_new, traj_i, phi, T, gamma, beta))
    log_scores.append(np.log(alpha) + log_lik_new)
    cluster_ids.append('new')

    # Normalize in log space and sample
    log_scores = np.array(log_scores)
    log_scores -= log_scores.max()  # numerical stability
    probs = np.exp(log_scores)
    probs /= probs.sum()
    rng_key, sk = jax.random.split(rng_key)
    idx = int(jax.random.choice(sk, len(cluster_ids),
                                p=jnp.array(probs)))
    chosen = cluster_ids[idx]

    if chosen == 'new':
        weight_vectors.append(w_new)
        return len(weight_vectors) - 1, weight_vectors, rng_key
    return chosen, weight_vectors, rng_key


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
