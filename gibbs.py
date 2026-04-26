# gibbs.py  -- pure-function refactor (Day 11, Person A)
# No global state, no class refs. pickle.dumps(gibbs_sweep) must succeed.

import jax
import jax.numpy as jnp
import numpy as np
from dp_prior import sample_new_weights, log_prior
from likelihood import log_likelihood_single


def sample_cluster_assignment(i, trajectories, assignments, weight_vectors,
                              alpha, phi, T, gamma, beta, rng_key):
    N = len(trajectories)
    K = len(weight_vectors)
    traj_i = trajectories[i]

    counts = np.zeros(K, dtype=float)
    for j in range(N):
        if j != i:
            counts[assignments[j]] += 1

    log_scores = []
    cluster_ids = []

    for k in range(K):
        if counts[k] == 0:
            continue
        log_scores.append(np.log(counts[k]) +
                          float(log_likelihood_single(
                              weight_vectors[k], traj_i, phi, T, gamma, beta)))
        cluster_ids.append(k)

    rng_key, sk = jax.random.split(rng_key)
    w_new = sample_new_weights(sk, phi.shape[1])
    log_scores.append(np.log(alpha) +
                      float(log_likelihood_single(
                          w_new, traj_i, phi, T, gamma, beta)))
    cluster_ids.append('new')

    log_scores = np.array(log_scores)
    log_scores -= log_scores.max()
    probs = np.exp(log_scores)
    probs /= probs.sum()

    rng_key, sk = jax.random.split(rng_key)
    idx = int(jax.random.choice(sk, len(cluster_ids), p=jnp.array(probs)))
    chosen = cluster_ids[idx]

    if chosen == 'new':
        weight_vectors = weight_vectors + [w_new]   # NEW list, no mutation
        return len(weight_vectors) - 1, weight_vectors, rng_key
    return chosen, weight_vectors, rng_key


def update_weight_vector(k, trajectories, assignments, weight_vectors,
                         phi, T, gamma, beta, step_size, rng_key):
    w_curr = weight_vectors[k]
    cluster_trajs = [trajectories[i]
                     for i, z in enumerate(assignments) if z == k]
    if not cluster_trajs:
        return weight_vectors, rng_key

    rng_key, sk = jax.random.split(rng_key)
    w_prop = w_curr + jax.random.normal(sk, shape=w_curr.shape) * step_size

    lp_curr = float(log_prior(w_curr))
    lp_prop = float(log_prior(w_prop))
    ll_curr = sum(float(log_likelihood_single(w_curr, t, phi, T, gamma, beta))
                  for t in cluster_trajs)
    ll_prop = sum(float(log_likelihood_single(w_prop, t, phi, T, gamma, beta))
                  for t in cluster_trajs)

    log_accept = (lp_prop + ll_prop) - (lp_curr + ll_curr)
    rng_key, sk = jax.random.split(rng_key)
    if float(jnp.log(jax.random.uniform(sk))) < log_accept:
        # return new list with updated entry — no mutation
        weight_vectors = (weight_vectors[:k] +
                          [w_prop] +
                          weight_vectors[k+1:])
    return weight_vectors, rng_key


def remap_assignments(assignments, weight_vectors):
    """Keep cluster labels contiguous after reassignment."""
    occupied = sorted(set(assignments))
    remap = {old: new for new, old in enumerate(occupied)}
    assignments = [remap[z] for z in assignments]
    weight_vectors = [weight_vectors[k] for k in occupied]
    return assignments, weight_vectors


def gibbs_sweep(trajectories, assignments, weight_vectors,
                alpha, phi, T, gamma, beta, step_size, rng_key):
    """
    Pure function — no global state, no class refs.
    All inputs are plain Python / numpy / jax arrays.
    pickle.dumps(gibbs_sweep) succeeds.
    Two calls with the same rng_key produce identical results.
    """
    N = len(trajectories)

    # Step 1: resample assignments
    for i in range(N):
        rng_key, sk = jax.random.split(rng_key)
        assignments[i], weight_vectors, sk = sample_cluster_assignment(
            i, trajectories, assignments, weight_vectors,
            alpha, phi, T, gamma, beta, sk)
        rng_key = sk

    # Step 2: remap to contiguous labels
    assignments, weight_vectors = remap_assignments(assignments, weight_vectors)

    # Step 3: MH weight update
    for k in range(len(weight_vectors)):
        weight_vectors, rng_key = update_weight_vector(
            k, trajectories, assignments, weight_vectors,
            phi, T, gamma, beta, step_size, rng_key)

    return assignments, weight_vectors, rng_key