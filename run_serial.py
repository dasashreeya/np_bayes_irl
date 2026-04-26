"""
run_serial.py  |  Owner: Both  |  Built: Day 5
End-to-end serial NP-Bayes IRL experiment.
Logs all metrics to W&B.
"""
import jax
import numpy as np
import wandb
from objectworld import ObjectWorld
from expert_demos import generate_dataset
from gibbs import gibbs_sweep, remap_assignments
from dp_prior import sample_new_weights
from eval import log_metrics, best_match_error, adjusted_rand_index
from maxent_irl import maxent_irl


def run_experiment(n_sweeps=300, alpha=1.0, step_size=0.1, seed=0):

    # --- Env setup ---
    env = ObjectWorld()
    phi = env.features()   # (n_states, n_features)
    T   = env.transitions() # (n_states, n_actions, n_states)
    gamma, beta = 0.95, 5.0

    trajs, true_weights, true_assignments = generate_dataset(
        env, phi, T, n_per_type=10)

    N = len(trajs)

    # --- W&B init ---
    wandb.init(
        project='np-bayes-irl',
        name='serial',
        config={
            'n_sweeps':  n_sweeps,
            'alpha':     alpha,
            'step_size': step_size,
            'seed':      seed,
            'env':       'ObjectWorld',
            'mode':      'serial',
        }
    )

    # --- Gibbs init: every trajectory in its own cluster ---
    rng = jax.random.PRNGKey(seed)
    rng, *init_keys = jax.random.split(rng, N + 1)

    assignments    = list(range(N))
    weight_vectors = [sample_new_weights(k, phi.shape[1]) for k in init_keys]
    assignments, weight_vectors = remap_assignments(assignments, weight_vectors)

    # --- Gibbs loop ---
    for sweep in range(n_sweeps):
        rng, sk = jax.random.split(rng)
        assignments, weight_vectors, _ = gibbs_sweep(
            trajs, assignments, weight_vectors,
            alpha, phi, T, gamma, beta, step_size, sk)

        if sweep % 10 == 0 or sweep == n_sweeps - 1:
            metrics = log_metrics(sweep, assignments, weight_vectors,
                                  true_weights, true_assignments)
            wandb.log({
                'sweep':      sweep,
                'n_clusters': metrics['n_clusters'],
                'l2_error':   metrics['l2_error'],
                'ari':        metrics['ari'],
            })

    # --- MaxEnt baseline on same data ---
    print("\n--- MaxEnt IRL Baseline ---")
    w_maxent = maxent_irl(trajs, phi, T, gamma=gamma, beta=beta)
    maxent_l2 = min(
        best_match_error([w_t], [w_maxent])
        for w_t in true_weights
    )
    maxent_ari = adjusted_rand_index(
        true_assignments,
        _maxent_assignments(trajs, w_maxent, phi, T, gamma, beta)
    )
    print(f"MaxEnt | L2: {maxent_l2:.4f} | ARI: {maxent_ari:.4f}")
    wandb.log({
        'maxent_l2_error': maxent_l2,
        'maxent_ari':      maxent_ari,
    })

    wandb.finish()

    return assignments, weight_vectors


def _maxent_assignments(trajs, w, phi, T, gamma, beta):
    """
    Assign each trajectory to cluster 0 — MaxEnt recovers one reward,
    so all demos map to a single cluster for ARI comparison.
    """
    return [0] * len(trajs)


if __name__ == '__main__':
    assignments, weight_vectors = run_experiment(n_sweeps=300)
    print(f"\nFinal n_clusters : {len(weight_vectors)}")
    print(f"Final assignments: {assignments}")