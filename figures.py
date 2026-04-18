"""
figures.py  |  Owner: Both  |  Built: Day 11
Generates all 5 publication figures from W&B data.
Saves PDF + PNG at 300dpi.
"""
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = 'results/figures/'

def fig1_speedup_curve(results):
    """Figure 1: wall-clock time vs number of workers."""
    workers = sorted(results.keys())
    times   = [results[w] for w in workers]
    speedups = [results[1] / t for t in times]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, speedups, 'o-', color='#1A7A6E', linewidth=2, markersize=8)
    ax.plot(workers, workers, '--', color='gray', label='Linear ideal')
    ax.set_xlabel('Number of workers')
    ax.set_ylabel('Speedup over serial')
    ax.set_title('Parallel Gibbs Speedup on Zartan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIGDIR + 'fig1_speedup.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR + 'fig1_speedup.png', dpi=300, bbox_inches='tight')
    print('Saved fig1_speedup')

def fig2_weight_error(worker_errors):
    """Figure 2: reward weight L2 error vs number of workers."""
    # TODO Day 11
    pass

def fig3_convergence(history):
    """Figure 3: n_clusters over Gibbs sweeps."""
    # TODO Day 11
    pass

def fig4_baseline_comparison(results_dict):
    """Figure 4: bar chart comparing NP-Bayes vs MaxEnt vs serial BayesIRL."""
    # TODO Day 11
    pass

def fig5_weight_recovery(true_w, recovered_w):
    """Figure 5: recovered vs true weight vectors (ObjectWorld specific)."""
    # TODO Day 11
    pass
