"""
evaluate.py  |  Owner: SA  |  Interface: PROJECT_SCOPE.md §5
THE ONLY MODULE THAT SEES GROUND TRUTH (decision D4).

Scores a PosteriorSummary against the true labels and weights. Nothing
computed here may flow back into the sampler or into choosing which
state to report.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

# Largest possible distance between two unit vectors; charged to each true
# type that no recovered cluster is matched to.
UNMATCHED_L2 = 2.0


def _unit(w):
    w = np.asarray(w, dtype=float)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def matched_weight_error(true_weights, true_sizes, rec_weights):
    """
    Hungarian matching between true and recovered reward weights (fixes P4).

    Cost of matching true type t to recovered cluster r is the L2 distance
    between the unit-normalized weights. Errors are averaged with weights
    proportional to the size of each true type.

    Args:
        true_weights: list of K_true (F,) arrays
        true_sizes:   (K_true,) number of trajectories of each true type
        rec_weights:  list of K_hat (F,) arrays
    Returns:
        dict with
          l2_matched:   size-weighted mean L2 over matched true types
          l2_penalized: same, but unmatched true types count as UNMATCHED_L2
          n_unmatched_true: true types left unmatched (under-clustering)
          n_extra:          recovered clusters left unmatched (over-clustering)
          match:            list of (true index, recovered index) pairs
    """
    T = np.stack([_unit(w) for w in true_weights])
    R = np.stack([_unit(w) for w in rec_weights])
    cost = np.linalg.norm(T[:, None, :] - R[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    sizes = np.asarray(true_sizes, dtype=float)
    err = np.full(len(T), UNMATCHED_L2)
    err[rows] = cost[rows, cols]
    return {
        "l2_matched": float(np.average(err[rows], weights=sizes[rows])),
        "l2_penalized": float(np.average(err, weights=sizes)),
        "n_unmatched_true": int(len(T) - len(rows)),
        "n_extra": int(len(R) - len(rows)),
        "match": [(int(r), int(c)) for r, c in zip(rows, cols)],
    }


def evaluate(summary, true_assignments, true_weights):
    """
    Args:
        summary:          PosteriorSummary
        true_assignments: (N,) true type of each trajectory, labels 0..K_true-1
        true_weights:     list of K_true (F,) true weight vectors
    Returns:
        dict with
          K_true, K_point (clusters in the Binder point estimate), K_map
          p_K_true:  posterior probability that K = K_true
          K_mean:    posterior mean of K
          K_mode:    posterior mode of K
          ari_point: ARI of the Binder point estimate
          ari_map:   ARI of the MAP sample
          plus matched_weight_error() of the point estimate's weights,
          and rhat_* / ess_* / accept_rate copied from the summary
    """
    z_true = np.asarray(true_assignments)
    K_true = len(true_weights)
    k_values = np.asarray(summary.k_values)
    k_probs = np.asarray(summary.k_probs)

    out = {
        "K_true": K_true,
        "K_point": int(np.unique(summary.point_clustering).size),
        "K_map": int(np.unique(summary.map_assignments).size),
        "p_K_true": float(k_probs[k_values == K_true].sum()),
        "K_mean": float(k_values @ k_probs),
        "K_mode": int(k_values[np.argmax(k_probs)]),
        "ari_point": float(adjusted_rand_score(z_true, summary.point_clustering)),
        "ari_map": float(adjusted_rand_score(z_true, summary.map_assignments)),
    }
    sizes = np.bincount(z_true, minlength=K_true)
    m = matched_weight_error(true_weights, sizes, summary.point_weights)
    out.update({k: v for k, v in m.items() if k != "match"})
    out["match"] = m["match"]
    out.update({f"rhat_{k}": float(v) for k, v in summary.rhat.items()})
    out.update({f"ess_{k}": float(v) for k, v in summary.ess.items()})
    out["accept_rate"] = float(summary.accept_rate)
    return out
