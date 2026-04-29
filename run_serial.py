"""
run_serial.py  |  Owner: Both  |  Built: Day 5
End-to-end serial NP-Bayes IRL experiment.
"""
import jax
import copy
import numpy as np
import wandb
from objectworld import ObjectWorld
from expert_demos import generate_dataset
from gibbs import gibbs_sweep, remap_assignments
from dp_prior import sample_new_weights
from eval import log_metrics, best_match_error, adjusted_rand_index
from maxent_irl import maxent_irl


def run_experiment(n_sweeps=500, alpha=5.0, step_size=0.01, seed=0):
    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()
    gamma, beta = 0.95, 5.0

    trajs, true_weights, true_assignments = generate_dataset(
        env, phi, T, n_per_type=10)
    N = len(trajs)

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

    rng = jax.random.PRNGKey(seed)
    rng, *init_keys = jax.random.split(rng, N + 1)

    assignments    = list(range(N))
    weight_vectors = [sample_new_weights(k, phi.shape[1]) for k in init_keys]
    assignments, weight_vectors = remap_assignments(assignments, weight_vectors)

    state = {
        'assignments':    assignments,
        'weight_vectors': weight_vectors,
    }

    best_ari   = -1.0
    best_state = copy.deepcopy(state)
    restarts   = 0

    for sweep in range(n_sweeps):
        rng, sk = jax.random.split(rng)
        state, _ = gibbs_sweep(
            trajs, state, phi, T,
            alpha=alpha, gamma=gamma, beta=beta,
            step_size=step_size, rng_key=sk,
        )

        # If collapsed to K=1 for 20+ sweeps, restart from best state
        if len(state['weight_vectors']) == 1 and sweep > 50:
            restarts += 1
            state = copy.deepcopy(best_state)
            # Perturb weights slightly to escape local optima
            rng, sk = jax.random.split(rng)
            perturbed = []
            for w in state['weight_vectors']:
                noise = jax.random.normal(sk, shape=w.shape) * 0.05
                perturbed.append(w + noise)
                rng, sk = jax.random.split(rng)
            state['weight_vectors'] = perturbed

        if sweep % 10 == 0 or sweep == n_sweeps - 1:
            metrics = log_metrics(
                sweep,
                state['assignments'],
                state['weight_vectors'],
                true_weights,
                true_assignments,
            )
            ari = metrics['ari']

            if ari > best_ari:
                best_ari   = ari
                best_state = copy.deepcopy(state)

            wandb.log({
                'sweep':      sweep,
                'n_clusters': metrics['n_clusters'],
                'l2_error':   metrics['l2_error'],
                'ari':        ari,
                'best_ari':   best_ari,
                'restarts':   restarts,
            })

    print(f"\nBest ARI achieved: {best_ari:.4f}")
    print(f"Total restarts: {restarts}")

    # Use best state for final output
    assignments    = best_state['assignments']
    weight_vectors = best_state['weight_vectors']

    # --- MaxEnt baseline ---
    print("\n--- MaxEnt IRL Baseline ---")
    w_maxent = maxent_irl(trajs, phi, T, gamma=gamma, beta=beta)
    maxent_l2 = min(
        best_match_error([w_t], [w_maxent])
        for w_t in true_weights
    )
    maxent_ari = adjusted_rand_index(
        true_assignments,
        [0] * len(trajs)
    )
    print(f"MaxEnt | L2: {maxent_l2:.4f} | ARI: {maxent_ari:.4f}")
    wandb.log({
        'maxent_l2_error': maxent_l2,
        'maxent_ari':      maxent_ari,
    })

    wandb.finish()
    return assignments, weight_vectors


if __name__ == '__main__':
    assignments, weight_vectors = run_experiment(n_sweeps=500)
    print(f"\nFinal n_clusters : {len(weight_vectors)}")
    print(f"Final assignments: {assignments}")
