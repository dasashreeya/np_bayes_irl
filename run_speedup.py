# run_speedup.py  |  Owner: Person A  |  Built: Day 11
# Time 1/2/4/8/16 workers at n_sweeps=100. Log speedup curve to W&B.

import time
import wandb
import ray
from expert_demos import generate_dataset
from objectworld import ObjectWorld
from parallel import run_parallel
import jax.numpy as jnp


def time_run(trajectories, phi, T, n_workers, n_sweeps=100, seed=42):
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    t0 = time.time()
    run_parallel(trajectories, phi, T,
                 n_workers=n_workers, n_sweeps=n_sweeps, seed=seed)
    return time.time() - t0


def run_speedup_experiment(n_sweeps=100, seed=0):
    env  = ObjectWorld()
    phi  = env.features()
    T    = env.transitions()
    trajs, _, _ = generate_dataset(env, phi, T, n_per_type=10)

    worker_counts = [1, 2, 4, 8, 16]

    wandb.init(project='np-bayes-irl', name='speedup-experiment',
               config={'n_sweeps': n_sweeps, 'seed': seed})

    baseline_time = None
    results = []

    for n_workers in worker_counts:
        print(f"\n--- {n_workers} worker(s) ---")
        wall = time_run(trajectories=trajs, phi=phi, T=T,
                        n_workers=n_workers, n_sweeps=n_sweeps, seed=seed)

        if baseline_time is None:
            baseline_time = wall          # 1-worker is the baseline

        speedup = baseline_time / wall
        results.append((n_workers, wall, speedup))

        print(f"n_workers={n_workers} | wall={wall:.2f}s | speedup={speedup:.2f}x")
        wandb.log({'n_workers': n_workers, 'wall_clock': wall, 'speedup': speedup})

    wandb.finish()

    print("\n=== Speedup Summary ===")
    print(f"{'workers':>8}  {'wall(s)':>10}  {'speedup':>8}")
    for n_workers, wall, speedup in results:
        print(f"{n_workers:>8}  {wall:>10.2f}  {speedup:>8.2f}x")

    return results


if __name__ == '__main__':
    # Gate: JIT warmup check
    import jax
    import jax.numpy as jnp
    import time
    from objectworld import ObjectWorld
    from likelihood import compute_log_pi

    env = ObjectWorld()
    phi = jnp.array(env.features())
    T   = jnp.array(env.transitions())
    w   = jnp.zeros(phi.shape[1])

    t1 = time.time(); compute_log_pi(w, phi, T, 0.95, 5.0).block_until_ready()
    t2 = time.time(); compute_log_pi(w, phi, T, 0.95, 5.0).block_until_ready()
    t3 = time.time()

    print(f"1st call: {t2-t1:.4f}s  |  2nd call: {t3-t2:.4f}s")
    assert t3-t2 < t2-t1, "JIT gate failed — 2nd call should be faster"
    print("JIT gate ✓")

    # Run speedup experiment
    run_speedup_experiment(n_sweeps=100)