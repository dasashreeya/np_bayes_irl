"""
run_experiments.py  |  Owner: Both  |  Built: Day 10
Master experiment script. Runs everything in sequence.
Clone repo + python run_experiments.py = all results reproduced.
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='all',
                        choices=['all','serial','parallel','baselines','speedup'])
    parser.add_argument('--n_sweeps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.exp in ('all', 'serial'):
        print('=== Running serial NP-Bayes IRL ===')
        from run_serial import run_experiment
        run_experiment(n_sweeps=args.n_sweeps, seed=args.seed)

    if args.exp in ('all', 'speedup'):
        print('=== Running speedup experiment ===')
        from run_speedup import speedup_experiment
        speedup_experiment(n_sweeps=100, seed=args.seed)

    if args.exp in ('all', 'baselines'):
        print('=== Running baselines ===')
        from experiments.exp_baselines import run_baselines
        run_baselines(seed=args.seed)

    print('All experiments complete. Check W&B for results.')


if __name__ == '__main__':
    main()
