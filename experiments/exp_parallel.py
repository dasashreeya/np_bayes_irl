"""Parallel NP-Bayes IRL -- all worker counts."""
from run_speedup import run_speedup_experiment
if __name__ == '__main__':
    run_speedup_experiment(n_sweeps=100)
