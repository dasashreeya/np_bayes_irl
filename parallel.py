# # parallel.py  |  Owner: Person B  |  Built: Day 11
# # Ray-parallel Gibbs: split trajectories across workers, merge states.

# import time
# import numpy as np
# import jax
# import ray

# from gibbs import gibbs_sweep, remap_assignments
# from dp_prior import sample_new_weights


# @ray.remote
# def worker_sweep(traj_chunk, assignments_chunk, weight_vectors,
#                  alpha, phi, T, gamma, beta, step_size, rng_seed):
#     """
#     One Gibbs sweep on a trajectory chunk.
#     Runs in a separate Ray worker process.
#     All args are plain numpy/python — no JAX device arrays crossed process boundary.
#     """
#     rng_key = jax.random.PRNGKey(rng_seed)
#     # Convert phi/T back to jnp inside worker
#     import jax.numpy as jnp
#     phi = jnp.array(phi)
#     T   = jnp.array(T)
#     weight_vectors = [jnp.array(w) for w in weight_vectors]

#     assignments_out, weights_out, _ = gibbs_sweep(
#         traj_chunk, assignments_chunk, weight_vectors,
#         alpha, phi, T, gamma, beta, step_size, rng_key)

#     # Return plain numpy so Ray can serialize cleanly
#     return assignments_out, [np.array(w) for w in weights_out]


# def merge_states(chunk_assignments, chunk_weights, chunk_sizes):
#     """
#     Collect assignments + weight vectors from all workers.
#     Re-indexes assignments globally and deduplicates weight vectors.
#     """
#     global_assignments = []
#     global_weights = []
#     offset = 0  # global trajectory index offset

#     for assigns, weights in zip(chunk_assignments, chunk_weights):
#         # Remap local cluster indices → global indices
#         local_to_global = {}
#         for local_k, w in enumerate(weights):
#             # Check if this weight vector is already in global_weights
#             matched = False
#             for global_k, gw in enumerate(global_weights):
#                 if np.allclose(w, gw, atol=1e-6):
#                     local_to_global[local_k] = global_k
#                     matched = True
#                     break
#             if not matched:
#                 local_to_global[local_k] = len(global_weights)
#                 global_weights.append(w)

#         for a in assigns:
#             global_assignments.append(local_to_global[a])

#     # Final remap to contiguous labels
#     import jax.numpy as jnp
#     weight_vectors_jnp = [jnp.array(w) for w in global_weights]
#     global_assignments, weight_vectors_jnp = remap_assignments(
#         global_assignments, weight_vectors_jnp)

#     return global_assignments, weight_vectors_jnp


# def run_parallel(trajectories, phi, T,
#                  alpha=1.0, step_size=0.1, gamma=0.95, beta=5.0,
#                  n_workers=2, n_sweeps=10, seed=0):
#     """
#     Split trajectories across n_workers Ray workers.
#     Each worker runs one full gibbs_sweep on its chunk per sweep.
#     Merges states after every sweep.
#     """
#     if not ray.is_initialized():
#         ray.init(ignore_reinit_error=True)

#     N = len(trajectories)
#     n_features = phi.shape[1]

#     # Init state
#     rng = np.random.default_rng(seed)
#     assignments    = list(range(N))
#     weight_vectors = [np.array(sample_new_weights(
#                         jax.random.PRNGKey(i), n_features))
#                       for i in range(N)]
#     assignments, weight_vectors_jnp = remap_assignments(
#         assignments, [__import__('jax').numpy.array(w) for w in weight_vectors])
#     weight_vectors = [np.array(w) for w in weight_vectors_jnp]

#     # Put shared read-only arrays in Ray object store once
#     phi_ref = ray.put(np.array(phi))
#     T_ref   = ray.put(np.array(T))

#     # Split trajectory indices into chunks
#     chunks = np.array_split(np.arange(N), n_workers)

#     t0 = time.time()

#     for sweep in range(n_sweeps):
#         futures = []
#         for chunk_idx in chunks:
#             chunk_idx = chunk_idx.tolist()
#             traj_chunk   = [trajectories[i] for i in chunk_idx]
#             assign_chunk = [assignments[i]   for i in chunk_idx]

#             seed_i = int(rng.integers(0, 2**31))
#             futures.append(worker_sweep.remote(
#                 traj_chunk, assign_chunk, weight_vectors,
#                 alpha, phi_ref, T_ref, gamma, beta, step_size, seed_i))

#         results = ray.get(futures)

#         chunk_assignments = [r[0] for r in results]
#         chunk_weights     = [r[1] for r in results]
#         chunk_sizes       = [len(c) for c in chunks]

#         assignments_jnp, weight_vectors_jnp = merge_states(
#             chunk_assignments, chunk_weights, chunk_sizes)

#         assignments    = assignments_jnp
#         weight_vectors = [np.array(w) for w in weight_vectors_jnp]

#         n_clusters = len(weight_vectors)
#         print(f"Sweep {sweep:>3d} | n_clusters: {n_clusters} | "
#               f"elapsed: {time.time()-t0:.1f}s")

#     print(f"\nTotal wall-time ({n_workers} workers, {n_sweeps} sweeps): "
#           f"{time.time()-t0:.2f}s")

#     return assignments, [__import__('jax').numpy.array(w) for w in weight_vectors]


# if __name__ == '__main__':
#     # Quick gate check
#     import pickle
#     from gibbs import gibbs_sweep
#     pickle.dumps(gibbs_sweep)
#     print("pickle.dumps(gibbs_sweep) ✓")


# parallel.py  |  Owner: Person B  |  Built: Day 11
# Ray-parallel Gibbs: split trajectories across workers, merge states.

import time
import numpy as np
import jax
import jax.numpy as jnp
import ray

from gibbs import _gibbs_sweep_inner, remap_assignments
from dp_prior import sample_new_weights


@ray.remote(max_retries=0)
def worker_sweep(traj_chunk, assignments_chunk, weight_vectors,
                 alpha, phi, T, gamma, beta, step_size, rng_seed):
    """
    One Gibbs sweep on a trajectory chunk.
    Runs in a separate Ray worker process.
    All args are plain numpy/python — no JAX device arrays crossed process boundary.
    Calls _gibbs_sweep_inner directly (picklable pure function, no state-dict wrap).
    """
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    rng_key = jax.random.PRNGKey(rng_seed)
    phi = jnp.array(phi)
    T   = jnp.array(T)
    weight_vectors = [jnp.array(w) for w in weight_vectors]

    assignments_out, weights_out, _ = _gibbs_sweep_inner(
        traj_chunk, list(assignments_chunk), weight_vectors,
        alpha, phi, T, gamma, beta, step_size, rng_key)

    return assignments_out, [np.array(w) for w in weights_out]


def merge_states(chunk_assignments, chunk_weights, chunk_sizes):
    """
    Collect assignments + weight vectors from all workers.
    Re-indexes assignments globally.

    KNOWN LIMITATION: dedupes weight vectors via np.allclose, which only works
    when workers happen to land on identical w (sweep 0 or post-init). After
    MH proposals are accepted independently per worker, "the same cluster"
    will have slightly different w in each worker and be treated as separate
    clusters here. Inflates K over time. Fix is a real consensus step.
    Acceptable for speedup-measurement runs; revisit before publishing
    cluster-recovery numbers from parallel mode.
    """
    global_assignments = []
    global_weights = []

    for assigns, weights in zip(chunk_assignments, chunk_weights):
        local_to_global = {}
        for local_k, w in enumerate(weights):
            matched = False
            for global_k, gw in enumerate(global_weights):
                if np.allclose(w, gw, atol=1e-6):
                    local_to_global[local_k] = global_k
                    matched = True
                    break
            if not matched:
                local_to_global[local_k] = len(global_weights)
                global_weights.append(w)

        for a in assigns:
            global_assignments.append(local_to_global[a])

    weight_vectors_jnp = [jnp.array(w) for w in global_weights]
    global_assignments, weight_vectors_jnp = remap_assignments(
        global_assignments, weight_vectors_jnp)

    return global_assignments, weight_vectors_jnp


def run_parallel(trajectories, phi, T,
                 alpha=1.0, step_size=0.1, gamma=0.95, beta=5.0,
                 n_workers=2, n_sweeps=10, seed=0):
    """
    Split trajectories across n_workers Ray workers.
    Each worker runs one full gibbs_sweep on its chunk per sweep.
    Merges states after every sweep.
    """
    # Warm up JAX on the main process so workers inherit a compiled XLA cache
    _dummy_w = jnp.zeros(phi.shape[1])
    _dummy_phi = jnp.array(phi)
    _dummy_T = jnp.array(T)
    from likelihood import compute_log_pi
    compute_log_pi(_dummy_w, _dummy_phi, _dummy_T, gamma, beta).block_until_ready()
    print("JAX warmed up on main process ✓")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    N = len(trajectories)
    n_features = phi.shape[1]

    rng = np.random.default_rng(seed)
    assignments    = list(range(N))
    weight_vectors = [np.array(sample_new_weights(
                        jax.random.PRNGKey(i), n_features))
                      for i in range(N)]
    assignments, weight_vectors_jnp = remap_assignments(
        assignments, [jnp.array(w) for w in weight_vectors])
    weight_vectors = [np.array(w) for w in weight_vectors_jnp]

    phi_ref = ray.put(np.array(phi))
    T_ref   = ray.put(np.array(T))

    chunks = np.array_split(np.arange(N), n_workers)

    t0 = time.time()

    for sweep in range(n_sweeps):
        futures = []
        for chunk_idx in chunks:
            chunk_idx = chunk_idx.tolist()
            traj_chunk   = [trajectories[i] for i in chunk_idx]
            assign_chunk = [assignments[i]   for i in chunk_idx]

            seed_i = int(rng.integers(0, 2**31))
            futures.append(worker_sweep.remote(
                traj_chunk, assign_chunk, weight_vectors,
                alpha, phi_ref, T_ref, gamma, beta, step_size, seed_i))

        results = ray.get(futures)

        chunk_assignments = [r[0] for r in results]
        chunk_weights     = [r[1] for r in results]
        chunk_sizes       = [len(c) for c in chunks]

        assignments_jnp, weight_vectors_jnp = merge_states(
            chunk_assignments, chunk_weights, chunk_sizes)

        assignments    = assignments_jnp
        weight_vectors = [np.array(w) for w in weight_vectors_jnp]

        n_clusters = len(weight_vectors)
        print(f"Sweep {sweep:>3d} | n_clusters: {n_clusters} | "
              f"elapsed: {time.time()-t0:.1f}s")

    print(f"\nTotal wall-time ({n_workers} workers, {n_sweeps} sweeps): "
          f"{time.time()-t0:.2f}s")

    return {
        'assignments':    assignments,
        'weight_vectors': [jnp.array(w) for w in weight_vectors],
    }


if __name__ == '__main__':
    # Quick gate check — confirm worker function is picklable
    import pickle
    from gibbs import _gibbs_sweep_inner
    pickle.dumps(_gibbs_sweep_inner)
    print("pickle.dumps(_gibbs_sweep_inner) ✓")