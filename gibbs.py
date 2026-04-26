import jax
import jax.numpy as jnp
import numpy as np
from dp_prior import sample_new_weights, log_prior
from likelihood import log_likelihood_single

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

def update_weight_vector(k, trajectories, assignments, weight_vectors,
                         phi, T, gamma, beta, step_size, rng_key):
    """
    Metropolis-Hastings update for reward weight vector w_k.
    Gaussian random walk proposal.
    Accept rate target: 20-50%.
    """
    w_curr = weight_vectors[k]
    # Trajectories assigned to cluster k
    cluster_trajs = [trajectories[i] for i, z in enumerate(assignments) if z == k]

    if len(cluster_trajs) == 0:
        return weight_vectors, rng_key

    # Gaussian random walk proposal
    rng_key, sk = jax.random.split(rng_key)
    noise = jax.random.normal(sk, shape=w_curr.shape) * step_size
    w_prop = w_curr + noise

    # Log prior
    lp_curr = float(log_prior(w_curr))
    lp_prop = float(log_prior(w_prop))

    # Log likelihood — sum over all trajectories in cluster k
    ll_curr = sum(float(log_likelihood_single(w_curr, traj, phi, T, gamma, beta))
                  for traj in cluster_trajs)
    ll_prop = sum(float(log_likelihood_single(w_prop, traj, phi, T, gamma, beta))
                  for traj in cluster_trajs)

    # MH acceptance
    log_accept = (lp_prop + ll_prop) - (lp_curr + ll_curr)
    rng_key, sk = jax.random.split(rng_key)
    log_u = float(jnp.log(jax.random.uniform(sk)))

    if log_u < log_accept:
        weight_vectors[k] = w_prop  # accept

    return weight_vectors, rng_key

def gibbs_sweep(trajectories, assignments, weight_vectors,
                alpha, phi, T, gamma, beta, step_size, rng_key):
    """One full Gibbs sweep: cluster assignments + MH weight updates."""
    N = len(trajectories)

    # --- Step 1: resample all cluster assignments ---
    for i in range(N):
        rng_key, sk = jax.random.split(rng_key)
        assignments[i], weight_vectors, sk = sample_cluster_assignment(
            i, trajectories, assignments, weight_vectors,
            alpha, phi, T, gamma, beta, sk)
        rng_key = sk

    # Prune empty clusters and remap assignment indices
    assignments, weight_vectors = _prune_clusters(assignments, weight_vectors)

    # --- Step 2: MH update for each cluster's weight vector ---
    K = len(weight_vectors)
    for k in range(K):
        weight_vectors, rng_key = update_weight_vector(
            k, trajectories, assignments, weight_vectors,
            phi, T, gamma, beta, step_size, rng_key)

    return assignments, weight_vectors, rng_key


def remap_assignments(assignments, weight_vectors):
    """Remove empty clusters; remap assignment indices to be contiguous."""
    K = len(weight_vectors)
    occupied = sorted(set(assignments))
    remap = {old: new for new, old in enumerate(occupied)}
    assignments = [remap[z] for z in assignments]
    weight_vectors = [weight_vectors[k] for k in occupied]
    return assignments, weight_vectors

def run_gibbs(trajectories, phi, T, gamma, beta,
              alpha=1.0, step_size=0.1, n_sweeps=200, rng_key=None):
    """
    Full Gibbs sampler for NP-Bayes IRL.
    Returns final (assignments, weight_vectors, history).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    N = len(trajectories)

    # --- Init: every trajectory starts in its own cluster ---
    rng_key, *init_keys = jax.random.split(rng_key, N + 1)
    assignments    = list(range(N))
    weight_vectors = [sample_new_weights(k, phi.shape[1]) for k in init_keys]

    history = []  # track (sweep, n_clusters, assignments copy)

    for sweep in range(n_sweeps):
        assignments, weight_vectors, rng_key = gibbs_sweep(
            trajectories, assignments, weight_vectors,
            alpha, phi, T, gamma, beta, step_size, rng_key)

        if sweep % 10 == 0 or sweep == n_sweeps - 1:
            n_clusters = len(weight_vectors)
            history.append({
                'sweep':      sweep,
                'n_clusters': n_clusters,
                'assignments': list(assignments),
            })
            print(f"Sweep {sweep:>4d} | clusters: {n_clusters} | "
                  f"assignments: {assignments}")

    return assignments, weight_vectors, history