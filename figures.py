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
    workers = sorted(worker_errors.keys())
    errors  = [worker_errors[w] for w in workers]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(workers, errors, 's-', color='#C0392B', linewidth=2, markersize=8)
    ax.set_xlabel('Number of workers')
    ax.set_ylabel('L2 weight recovery error')
    ax.set_title('Weight Recovery Error vs Workers')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.savefig(FIGDIR + 'fig2_weight_error.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR + 'fig2_weight_error.png', dpi=300, bbox_inches='tight')
    print('Saved fig2_weight_error')


def fig3_convergence(history):
    """Figure 3: n_clusters over Gibbs sweeps."""
    # history: list of dicts with keys 'sweep' and 'n_clusters'
    sweeps     = [h['sweep']      for h in history]
    n_clusters = [h['n_clusters'] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweeps, n_clusters, color='#2C3E7A', linewidth=2)
    ax.axhline(y=2, color='gray', linestyle='--', label='True K=2')
    ax.axvline(x=100, color='orange', linestyle=':', alpha=0.7, label='Burn-in (100)')
    ax.set_xlabel('Gibbs sweep')
    ax.set_ylabel('Number of clusters K')
    ax.set_title('Cluster Count Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.savefig(FIGDIR + 'fig3_convergence.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR + 'fig3_convergence.png', dpi=300, bbox_inches='tight')
    print('Saved fig3_convergence')


def fig4_baseline_comparison(results_dict):
    """
    Figure 4: bar chart comparing NP-Bayes vs MaxEnt vs serial BayesIRL.

    results_dict format:
    {
        'NP-Bayes (parallel)': {'ari': 0.91, 'l2': 0.23},
        'NP-Bayes (serial)':   {'ari': 0.89, 'l2': 0.25},
        'MaxEnt IRL':          {'ari': 0.41, 'l2': 0.71},
    }
    """
    methods = list(results_dict.keys())
    ari     = [results_dict[m]['ari'] for m in methods]
    l2      = [results_dict[m]['l2']  for m in methods]

    x      = np.arange(len(methods))
    width  = 0.35
    colors = ['#1A7A6E', '#2C3E7A', '#C0392B']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ARI bars
    bars1 = ax1.bar(x, ari, width=0.5, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax1.set_ylabel('Adjusted Rand Index')
    ax1.set_title('Cluster Recovery (ARI)')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars1, ari):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # L2 bars
    bars2 = ax2.bar(x, l2, width=0.5, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel('L2 Weight Error (normalized)')
    ax2.set_title('Reward Recovery (L2 Error)')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars2, l2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('NP-Bayes IRL vs Baselines', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR + 'fig4_baseline_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR + 'fig4_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved fig4_baseline_comparison')


def fig5_weight_recovery(true_w, recovered_w):
    """
    Figure 5: recovered vs true weight vectors side by side.

    true_w:      list of 2 arrays, shape (16,) each  [w1, w2]
    recovered_w: list of 2 arrays, shape (16,) each  [w1_hat, w2_hat]
    """
    feature_labels = (
        [f'C{c}' for c in range(8)] +
        [f'C{c}(2nd)' for c in range(8)]
    )
    x = np.arange(16)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    titles = ['Expert Type 1 (red-lover)', 'Expert Type 2 (blue-lover)']
    colors_true = '#2C3E7A'
    colors_rec  = '#1A7A6E'

    for idx, (ax, title) in enumerate(zip(axes, titles)):
        tw = np.array(true_w[idx])
        rw = np.array(recovered_w[idx])

        # Align sign — recovered weights may be sign-flipped
        if np.dot(tw, rw) < 0:
            rw = -rw

        # Normalize both for fair comparison
        tw = tw / (np.linalg.norm(tw) + 1e-8)
        rw = rw / (np.linalg.norm(rw) + 1e-8)

        ax.bar(x - width/2, tw, width, label='True',      color=colors_true, alpha=0.85, edgecolor='white')
        ax.bar(x + width/2, rw, width, label='Recovered', color=colors_rec,  alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Normalized weight' if idx == 0 else '')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle('Recovered vs True Reward Weights', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGDIR + 'fig5_weight_recovery.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR + 'fig5_weight_recovery.png', dpi=300, bbox_inches='tight')
    print('Saved fig5_weight_recovery')


# ── Main: generate all figures from a completed run ──────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs(FIGDIR, exist_ok=True)

    # ── Paste your actual run_speedup results here ──
    speedup_results = {1: 37.26, 2: 21.13, 4: 21.46, 8: 31.65, 16: 137.84}

    worker_errors = {1: 0.25, 2: 0.26, 4: 0.27, 8: 0.29, 16: 0.34}  # replace with real values

    # ── Replace with real Gibbs history from run_serial ──
    history = [{'sweep': i*10, 'n_clusters': max(2, 6 - i//3)}
               for i in range(50)]

    baseline_results = {
        'NP-Bayes (parallel)': {'ari': 0.91, 'l2': 0.23},
        'NP-Bayes (serial)':   {'ari': 0.89, 'l2': 0.25},
        'MaxEnt IRL':          {'ari': 0.41, 'l2': 0.71},
    }

    # ── Replace with real weights from run_serial output ──
    import numpy as np
    true_w = [
        np.array([1,-1,0,0,0,0,0,0, 1,-1,0,0,0,0,0,0], dtype=float),
        np.array([-1,1,0,0,0,0,0,0,-1, 1,0,0,0,0,0,0], dtype=float),
    ]
    recovered_w = [
        np.array([0.9,-0.8,0.1,-0.1,0,0,0,0, 0.85,-0.75,0,0,0,0,0,0], dtype=float),
        np.array([-0.85,0.9,0,0,0,0,0,0,-0.8, 0.88,0,0,0,0,0,0], dtype=float),
    ]

    fig1_speedup_curve(speedup_results)
    fig2_weight_error(worker_errors)
    fig3_convergence(history)
    fig4_baseline_comparison(baseline_results)
    fig5_weight_recovery(true_w, recovered_w)

    print(f"\nAll 5 figures saved to {FIGDIR}")
