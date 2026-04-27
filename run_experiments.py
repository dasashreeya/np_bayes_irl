"""
run_experiments.py  |  Owner: Both  |  Built: Day 10
Master experiment script. Runs everything in sequence.
Clone repo + python run_experiments.py = all results reproduced.
"""


import os
import time
import jax
import jax.numpy as jnp
import wandb

from objectworld import ObjectWorld
from expert_demos import generate_dataset
from likelihood import log_likelihood
from dp_prior import sample_new_weights
from gibbs import run_gibbs
from parallel import run_parallel
from maxent_irl import maxent_irl
from eval import best_match_error, adjusted_rand_index, log_metrics
from reward_features import normalize_weights
from figures import (
    fig1_speedup_curve,
    fig2_weight_error,
    fig3_convergence,
    fig4_baseline_comparison,
    fig5_weight_recovery,
)

# ── Config ───────────────────────────────────────────────────────────────────

SEED       = 42
N_SWEEPS   = 500
BURN_IN    = 100
ALPHA      = 1.0
GAMMA      = 0.95
BETA       = 1.0
STEP_SIZE  = 0.1
N_WORKERS  = [1, 2, 4, 8, 16]
N_SWEEPS_SPEEDUP = 100    # shorter run for timing experiment

os.makedirs("results/figures", exist_ok=True)


# ── Shared setup (same seed for all experiments) ─────────────────────────────

def setup():
    env = ObjectWorld()
    phi = env.features()     # (100, 16)
    T   = env.transitions()  # (100, 4, 100)

    trajs, true_weights, true_assignments = generate_dataset(
        env, phi, T, n_per_type=10, seed=SEED
    )
    return env, phi, T, trajs, true_weights, true_assignments


# ── Experiment 1: Serial NP-Bayes IRL ────────────────────────────────────────

def run_serial_experiment(phi, T, trajs, true_weights, true_assignments):
    print("\n" + "="*60)
    print("EXPERIMENT 1: Serial NP-Bayes IRL")
    print("="*60)

    wandb.init(project="np-bayes-irl", name="serial", reinit=True,
               config={"n_sweeps": N_SWEEPS, "alpha": ALPHA, "gamma": GAMMA,
                       "beta": BETA, "step_size": STEP_SIZE, "seed": SEED})

    rng = jax.random.PRNGKey(SEED)

    t0 = time.time()
    final_state, history = run_gibbs(
        trajs, phi, T,
        n_sweeps=N_SWEEPS,
        alpha=ALPHA,
        gamma=GAMMA,
        beta=BETA,
        step_size=STEP_SIZE,
        burn_in=BURN_IN,
        rng_key=rng,
        log_every=10,
    )
    wall = time.time() - t0

    # ── Metrics ──────────────────────────────────────────────────────────────
    recovered_assignments = final_state['assignments']
    recovered_weights     = final_state['weight_vectors']

    ari    = adjusted_rand_index(true_assignments, recovered_assignments)
    l2_err = best_match_error(true_weights, recovered_weights)

    print(f"\nSerial NP-Bayes Results:")
    print(f"  Wall time:       {wall:.2f}s")
    print(f"  n_clusters:      {len(set(recovered_assignments))}")
    print(f"  ARI:             {ari:.4f}")
    print(f"  L2 weight error: {l2_err:.4f}")

    wandb.log({"wall_time": wall, "ari": ari, "l2_error": l2_err,
               "n_clusters": len(set(recovered_assignments))})
    wandb.finish()

    return {
        "wall":       wall,
        "ari":        ari,
        "l2":         l2_err,
        "history":    history,
        "weights":    recovered_weights,
        "assignments": recovered_assignments,
    }


# ── Experiment 2: Parallel NP-Bayes — speedup curve ─────────────────────────

def run_parallel_experiment(phi, T, trajs, true_weights, true_assignments):
    print("\n" + "="*60)
    print("EXPERIMENT 2: Parallel NP-Bayes IRL — Speedup Curve")
    print("="*60)

    wandb.init(project="np-bayes-irl", name="parallel-speedup", reinit=True,
               config={"n_sweeps": N_SWEEPS_SPEEDUP, "seed": SEED})

    speedup_results  = {}   # worker → wall time
    worker_errors    = {}   # worker → l2 error
    parallel_weights = None

    for n_workers in N_WORKERS:
        print(f"\n  Running {n_workers} worker(s)...")
        rng = jax.random.PRNGKey(SEED)   # same seed every run

        t0 = time.time()
        final_state = run_parallel(
            trajs, phi, T,
            n_workers=n_workers,
            n_sweeps=N_SWEEPS_SPEEDUP,
            alpha=ALPHA,
            gamma=GAMMA,
            beta=BETA,
            step_size=STEP_SIZE,
            seed=SEED,
        )
        wall = time.time() - t0

        recovered_assignments = final_state['assignments']
        recovered_weights     = final_state['weight_vectors']

        ari    = adjusted_rand_index(true_assignments, recovered_assignments)
        l2_err = best_match_error(true_weights, recovered_weights)

        speedup_results[n_workers] = wall
        worker_errors[n_workers]   = l2_err

        if n_workers == 1:
            parallel_weights = recovered_weights

        speedup = speedup_results[1] / wall if n_workers > 1 else 1.0

        print(f"  n_workers={n_workers} | wall={wall:.2f}s | "
              f"speedup={speedup:.2f}x | ARI={ari:.3f} | L2={l2_err:.3f}")

        wandb.log({
            "n_workers":  n_workers,
            "wall_clock": wall,
            "speedup":    speedup_results[1] / wall if n_workers > 1 else 1.0,
            "ari":        ari,
            "l2_error":   l2_err,
        })

    wandb.finish()

    return {
        "speedup_results": speedup_results,
        "worker_errors":   worker_errors,
        "weights":         parallel_weights,
    }


# ── Experiment 3: MaxEnt IRL baseline ────────────────────────────────────────

def run_maxent_experiment(env, phi, T, trajs, true_weights, true_assignments):
    print("\n" + "="*60)
    print("EXPERIMENT 3: MaxEnt IRL Baseline")
    print("="*60)

    wandb.init(project="np-bayes-irl", name="maxent-baseline", reinit=True,
               config={"seed": SEED})

    t0 = time.time()
    w_maxent = maxent_irl(trajs, phi, T, gamma=GAMMA, n_iters=200, lr=0.01)
    wall = time.time() - t0

    # MaxEnt returns one weight vector — compare against both true types
    l2_err = min(
        best_match_error([true_weights[0]], [w_maxent]),
        best_match_error([true_weights[1]], [w_maxent]),
    )

    # ARI: MaxEnt assigns everyone to one cluster — always 0
    maxent_assignments = [0] * len(trajs)
    ari = adjusted_rand_index(true_assignments, maxent_assignments)

    print(f"\nMaxEnt IRL Results:")
    print(f"  Wall time:       {wall:.2f}s")
    print(f"  L2 weight error: {l2_err:.4f}")
    print(f"  ARI:             {ari:.4f}  (expected ~0 — single cluster)")

    wandb.log({"wall_time": wall, "l2_error": l2_err, "ari": ari})
    wandb.finish()

    return {"wall": wall, "l2": l2_err, "ari": ari, "weights": w_maxent}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NP-Bayes IRL — Full Experiment Suite")
    print(f"Seed: {SEED} | Sweeps: {N_SWEEPS} | Workers: {N_WORKERS}")

    # ── Setup ────────────────────────────────────────────────────────────────
    env, phi, T, trajs, true_weights, true_assignments = setup()
    print(f"Dataset ready: {len(trajs)} trajectories, "
          f"true K={len(set(true_assignments))}")

    # ── Run all three experiments ─────────────────────────────────────────────
    serial_results   = run_serial_experiment(phi, T, trajs, true_weights, true_assignments)
    parallel_results = run_parallel_experiment(phi, T, trajs, true_weights, true_assignments)
    maxent_results   = run_maxent_experiment(env, phi, T, trajs, true_weights, true_assignments)

    # ── Generate all 5 figures ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generating figures...")
    print("="*60)

    fig1_speedup_curve(parallel_results['speedup_results'])

    fig2_weight_error(parallel_results['worker_errors'])

    fig3_convergence(serial_results['history'])

    fig4_baseline_comparison({
        'NP-Bayes (parallel)': {
            'ari': adjusted_rand_index(
                true_assignments,
                run_parallel.__module__ and parallel_results.get('assignments',
                serial_results['assignments'])
            ),
            'l2': parallel_results['worker_errors'].get(1, serial_results['l2']),
        },
        'NP-Bayes (serial)': {
            'ari': serial_results['ari'],
            'l2':  serial_results['l2'],
        },
        'MaxEnt IRL': {
            'ari': maxent_results['ari'],
            'l2':  maxent_results['l2'],
        },
    })

    fig5_weight_recovery(
        true_weights,
        serial_results['weights'][:2]
        if len(serial_results['weights']) >= 2
        else serial_results['weights']
    )

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Serial NP-Bayes  | ARI: {serial_results['ari']:.3f} | "
          f"L2: {serial_results['l2']:.3f} | Time: {serial_results['wall']:.1f}s")
    print(f"Parallel (1w)    | ARI: N/A | "
          f"L2: {parallel_results['worker_errors'].get(1, 0):.3f} | "
          f"Time: {parallel_results['speedup_results'].get(1, 0):.1f}s")
    print(f"MaxEnt baseline  | ARI: {maxent_results['ari']:.3f} | "
          f"L2: {maxent_results['l2']:.3f} | Time: {maxent_results['wall']:.1f}s")
    print(f"\nBest speedup: {max(parallel_results['speedup_results'].get(1,1) / v for v in parallel_results['speedup_results'].values()):.2f}x")
    print(f"\nAll figures saved to results/figures/")
    print("Done.")