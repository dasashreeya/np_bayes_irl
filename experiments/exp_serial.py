"""Serial NP-Bayes IRL -- ObjectWorld, 2 reward types."""
from run_serial import run_experiment
if __name__ == '__main__':
    run_experiment(n_sweeps=500, alpha=1.0, seed=0)
