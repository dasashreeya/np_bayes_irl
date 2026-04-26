"""
eval.py  |  Owner: P1  |  Built: Day 2
Evaluation metrics for NP-Bayes IRL.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from reward_features import normalize_weights

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


def l2_weight_error(w_true, w_recovered):
    """Normalized L2 between true and recovered weight vectors."""
    w_t = normalize_weights(np.array(w_true))
    w_r = normalize_weights(np.array(w_recovered))
    return float(np.linalg.norm(w_t - w_r))


def best_match_error(true_weights, recovered_weights):
    """
    Match each true weight vector to its closest recovered vector.
    Returns mean L2 error over best matches.
    true_weights:      list of (n_features,) arrays
    recovered_weights: list of (n_features,) arrays
    """
    errors = []
    for w_true in true_weights:
        best = min(recovered_weights,
                   key=lambda w_rec: l2_weight_error(w_true, w_rec))
        errors.append(l2_weight_error(w_true, best))
    return float(np.mean(errors))


def adjusted_rand_index(true_assignments, pred_assignments):
    """
    Cluster assignment accuracy via ARI (sklearn).
    ARI=1.0 for perfect match (label-permutation invariant).
    """
    return float(adjusted_rand_score(true_assignments, pred_assignments))


def log_metrics(sweep, assignments, weight_vectors, true_weights, true_assignments):
    """
    Print K, L2 error, ARI per sweep. Log to W&B if available.
    """
    K   = len(weight_vectors)
    l2  = best_match_error(true_weights, weight_vectors)
    ari = adjusted_rand_index(true_assignments, assignments)

    print(f"Sweep {sweep:>4d} | K: {K} | L2: {l2:.4f} | ARI: {ari:.4f}")

    if WANDB:
        wandb.log({"sweep": sweep, "n_clusters": K, "l2_error": l2, "ari": ari})

    return {"sweep": sweep, "n_clusters": K, "l2_error": l2, "ari": ari}
