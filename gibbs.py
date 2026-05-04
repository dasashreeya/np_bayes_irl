# # gibbs.py  -- pure-function refactor (Day 11, Person A)
# # No global state, no class refs. pickle.dumps(gibbs_sweep) must succeed.

# import jax
# import jax.numpy as jnp
# import numpy as np
# from dp_prior import sample_new_weights, log_prior
# from likelihood import log_likelihood_single


# def sample_cluster_assignment(i, trajectories, assignments, weight_vectors,
#                               alpha, phi, T, gamma, beta, rng_key):
#     N = len(trajectories)
#     K = len(weight_vectors)
#     traj_i = trajectories[i]

#     counts = np.zeros(K, dtype=float)
#     for j in range(N):
#         if j != i:
#             counts[assignments[j]] += 1

#     log_scores = []
#     cluster_ids = []

#     for k in range(K):
#         if counts[k] == 0:
#             continue
#         log_scores.append(np.log(counts[k]) +
#                           float(log_likelihood_single(
#                               weight_vectors[k], traj_i, phi, T, gamma, beta)))
#         cluster_ids.append(k)

#     rng_key, sk = jax.random.split(rng_key)
#     w_new = sample_new_weights(sk, phi.shape[1])
#     log_scores.append(np.log(alpha) +
#                       float(log_likelihood_single(
#                           w_new, traj_i, phi, T, gamma, beta)))
#     cluster_ids.append('new')

#     log_scores = np.array(log_scores)
#     log_scores -= log_scores.max()
#     probs = np.exp(log_scores)
#     probs /= probs.sum()

#     rng_key, sk = jax.random.split(rng_key)
#     idx = int(jax.random.choice(sk, len(cluster_ids), p=jnp.array(probs)))
#     chosen = cluster_ids[idx]

#     if chosen == 'new':
#         weight_vectors = weight_vectors + [w_new]   # NEW list, no mutation
#         return len(weight_vectors) - 1, weight_vectors, rng_key
#     return chosen, weight_vectors, rng_key


# def update_weight_vector(k, trajectories, assignments, weight_vectors,
#                          phi, T, gamma, beta, step_size, rng_key):
#     w_curr = weight_vectors[k]
#     cluster_trajs = [trajectories[i]
#                      for i, z in enumerate(assignments) if z == k]
#     if not cluster_trajs:
#         return weight_vectors, rng_key

#     rng_key, sk = jax.random.split(rng_key)
#     w_prop = w_curr + jax.random.normal(sk, shape=w_curr.shape) * step_size

#     lp_curr = float(log_prior(w_curr))
#     lp_prop = float(log_prior(w_prop))
#     ll_curr = sum(float(log_likelihood_single(w_curr, t, phi, T, gamma, beta))
#                   for t in cluster_trajs)
#     ll_prop = sum(float(log_likelihood_single(w_prop, t, phi, T, gamma, beta))
#                   for t in cluster_trajs)

#     log_accept = (lp_prop + ll_prop) - (lp_curr + ll_curr)
#     rng_key, sk = jax.random.split(rng_key)
#     if float(jnp.log(jax.random.uniform(sk))) < log_accept:
#         # return new list with updated entry — no mutation
#         weight_vectors = (weight_vectors[:k] +
#                           [w_prop] +
#                           weight_vectors[k+1:])
#     return weight_vectors, rng_key


# def remap_assignments(assignments, weight_vectors):
#     """Keep cluster labels contiguous after reassignment."""
#     occupied = sorted(set(assignments))
#     remap = {old: new for new, old in enumerate(occupied)}
#     assignments = [remap[z] for z in assignments]
#     weight_vectors = [weight_vectors[k] for k in occupied]
#     return assignments, weight_vectors


# def gibbs_sweep(trajectories, assignments, weight_vectors,
#                 alpha, phi, T, gamma, beta, step_size, rng_key):
#     """
#     Pure function — no global state, no class refs.
#     All inputs are plain Python / numpy / jax arrays.
#     pickle.dumps(gibbs_sweep) succeeds.
#     Two calls with the same rng_key produce identical results.
#     """
#     N = len(trajectories)

#     # Step 1: resample assignments
#     for i in range(N):
#         rng_key, sk = jax.random.split(rng_key)
#         assignments[i], weight_vectors, sk = sample_cluster_assignment(
#             i, trajectories, assignments, weight_vectors,
#             alpha, phi, T, gamma, beta, sk)
#         rng_key = sk

#     # Step 2: remap to contiguous labels
#     assignments, weight_vectors = remap_assignments(assignments, weight_vectors)

#     # Step 3: MH weight update
#     for k in range(len(weight_vectors)):
#         weight_vectors, rng_key = update_weight_vector(
#             k, trajectories, assignments, weight_vectors,
#             phi, T, gamma, beta, step_size, rng_key)

#     return assignments, weight_vectors, rng_key




# gibbs.py  -- pure-function refactor (Day 11, Person A + fixes)
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
        weight_vectors = weight_vectors + [w_new]
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


def _gibbs_sweep_inner(trajectories, assignments, weight_vectors,
                       alpha, phi, T, gamma, beta, step_size, rng_key):
    """
    Raw inner sweep — works on plain lists, no state dicts.
    Used internally by gibbs_sweep and parallel.py workers.
    pickle.dumps(_gibbs_sweep_inner) succeeds — no lambdas, no globals.
    """
    N = len(trajectories)

    # Step 1: resample all assignments
    for i in range(N):
        rng_key, sk = jax.random.split(rng_key)
        assignments[i], weight_vectors, sk = sample_cluster_assignment(
            i, trajectories, assignments, weight_vectors,
            alpha, phi, T, gamma, beta, sk)
        rng_key = sk

    # Step 2: remap to contiguous labels
    assignments, weight_vectors = remap_assignments(assignments, weight_vectors)

    # Step 3: MH weight update for each cluster
    for k in range(len(weight_vectors)):
        weight_vectors, rng_key = update_weight_vector(
            k, trajectories, assignments, weight_vectors,
            phi, T, gamma, beta, step_size, rng_key)

    return assignments, weight_vectors, rng_key


def gibbs_sweep(trajectories, state, phi, T,
                alpha=1.0, gamma=0.95, beta=1.0,
                step_size=0.1, rng_key=None):
    """
    Public API — accepts and returns state dicts.
    Called by tests, run_serial.py, and parallel.py.

    Args:
        trajectories: list of (state, action) trajectories
        state:        dict with keys 'assignments' and 'weight_vectors'
        phi:          (n_states, n_features)
        T:            (n_states, n_actions, n_states)
        alpha:        DP concentration parameter
        gamma:        discount factor
        beta:         Boltzmann temperature
        step_size:    MH proposal std
        rng_key:      JAX random key

    Returns:
        new_state: dict with 'assignments' and 'weight_vectors'
        metrics:   dict with 'n_clusters'
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    assignments    = list(state['assignments'])
    weight_vectors = list(state['weight_vectors'])

    assignments, weight_vectors, rng_key = _gibbs_sweep_inner(
        trajectories, assignments, weight_vectors,
        alpha, phi, T, gamma, beta, step_size, rng_key
    )

    new_state = {
        'assignments':    assignments,
        'weight_vectors': weight_vectors,
    }
    metrics = {
        'n_clusters': len(set(assignments)),
    }
    return new_state, metrics


def run_gibbs(trajectories, phi, T,
              n_sweeps=500, alpha=1.0, gamma=0.95, beta=1.0,
              step_size=0.1, burn_in=100, rng_key=None, log_every=10):
    """
    Full Gibbs training loop.

    Args:
        trajectories: list of expert trajectories
        phi:          (n_states, n_features)
        T:            (n_states, n_actions, n_states)
        n_sweeps:     total Gibbs sweeps
        alpha:        DP concentration
        gamma:        discount factor
        beta:         Boltzmann temperature
        step_size:    MH proposal std
        burn_in:      sweeps before collecting history
        rng_key:      JAX random key
        log_every:    print interval

    Returns:
        final_state: dict with 'assignments' and 'weight_vectors'
        history:     list of metric dicts (one per post-burn-in sweep)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Initialise: all trajectories in one cluster, random weight
    rng_key, sk = jax.random.split(rng_key)
    state = {
        'assignments':    [0] * len(trajectories),
        'weight_vectors': [sample_new_weights(sk, phi.shape[1])],
    }

    history = []

    for sweep in range(n_sweeps):
        rng_key, sk = jax.random.split(rng_key)
        state, metrics = gibbs_sweep(
            trajectories, state, phi, T,
            alpha=alpha, gamma=gamma, beta=beta,
            step_size=step_size, rng_key=sk
        )
        metrics['sweep'] = sweep

        if sweep >= burn_in:
            history.append(metrics)

        if sweep % log_every == 0:
            print(f"Sweep {sweep:4d} | n_clusters: {metrics['n_clusters']}")

    return state, history
