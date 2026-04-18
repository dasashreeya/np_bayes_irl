"""
run_speedup.py  |  Owner: Both  |  Built: Day 8
Speedup experiment: 1, 2, 4, 8, 16 workers on Zartan.
Generates Figure 1 (speedup curve) data.
This is the headline result.
"""
import wandb
import time
from objectworld import ObjectWorld
from expert_demos import generate_dataset
from parallel import run_parallel


def speedup_experiment(n_sweeps=100, seed=0):
    env  = ObjectWorld()
    phi  = env.features()
    T    = env.transitions()
    trajs, true_w, true_z = generate_dataset(env, phi, T, n_per_type=10)

    run = wandb.init(project='np-bayes-irl', name='speedup-zartan')
    results = {}

    for n_workers in [1, 2, 4, 8, 16]:
        print(f'Running {n_workers} workers...')
        t0 = time.time()
        state, times = run_parallel(trajs, phi, T,
                                    n_workers=n_workers, n_sweeps=n_sweeps, seed=seed)
        wall_clock = time.time() - t0
        results[n_workers] = wall_clock
        speedup = results[1] / wall_clock if 1 in results else 1.0
        run.log({'n_workers': n_workers, 'wall_clock': wall_clock, 'speedup': speedup})
        print(f'  {n_workers} workers: {wall_clock:.2f}s  ({speedup:.2f}x)')

    run.finish()
    return results


if __name__ == '__main__':
    results = speedup_experiment()
    baseline = results[1]
    print('\n--- Speedup Summary ---')
    for w, t in results.items():
        print(f'{w:2d} workers: {t:.2f}s  ({baseline/t:.2f}x speedup)')
