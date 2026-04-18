"""Parallel NP-Bayes IRL -- all worker counts."""
from run_speedup import speedup_experiment
if __name__ == '__main__':
    speedup_experiment(n_sweeps=100)
