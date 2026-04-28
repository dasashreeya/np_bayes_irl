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

try:
    from scipy.stats import pearsonr
    SCIPY = True
except ImportError:
    SCIPY = False


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


def policy_similarity(w_true, w_recovered, test_obs_list, env,
                      beta=1.0, n_action_samples=20):
    """
    KL divergence between true and recovered policy action distributions,
    averaged over test_obs_list.

    Approximates both policies as Boltzmann distributions over sampled actions.
    Lower is better (KL >= 0).

    Args:
        w_true:          flat neural reward parameter vector (true)
        w_recovered:     flat neural reward parameter vector (recovered)
        test_obs_list:   list of (obs_dim,) observations
        env:             MuJoCoEnv instance for sampling actions
        beta:            inverse temperature
        n_action_samples: number of random actions to approximate distribution

    Returns:
        mean KL divergence (float)
    """
    import jax.numpy as jnp
    from neural_reward import compute_reward

    kls = []
    for obs in test_obs_list:
        obs_arr = np.array(obs, dtype=np.float32)
        actions = [env.sample_action() for _ in range(n_action_samples)]

        r_true = np.array([float(compute_reward(w_true, jnp.asarray(obs_arr)))
                           for _ in actions], dtype=np.float64)
        r_rec  = np.array([float(compute_reward(w_recovered, jnp.asarray(obs_arr)))
                           for _ in actions], dtype=np.float64)

        def softmax(x):
            x = x - x.max()
            e = np.exp(beta * x)
            return e / e.sum()

        p = softmax(r_true)
        q = softmax(r_rec)
        kl = float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))
        kls.append(kl)

    return float(np.mean(kls))


def reward_correlation(w_true, w_recovered, test_trajs):
    """
    Pearson correlation between true and recovered reward on held-out trajectories.

    Args:
        w_true:       flat neural reward parameter vector (true)
        w_recovered:  flat neural reward parameter vector (recovered)
        test_trajs:   list of (obs, action) trajectory lists

    Returns:
        Pearson r in [-1, 1]; higher is better
    """
    import jax.numpy as jnp
    from neural_reward import compute_reward

    r_true_all, r_rec_all = [], []
    for traj in test_trajs:
        for obs, _action in traj:
            obs_arr = jnp.asarray(obs, dtype=jnp.float32)
            r_true_all.append(float(compute_reward(w_true, obs_arr)))
            r_rec_all.append(float(compute_reward(w_recovered, obs_arr)))

    r_true_all = np.array(r_true_all)
    r_rec_all  = np.array(r_rec_all)

    if SCIPY:
        corr, _ = pearsonr(r_true_all, r_rec_all)
    else:
        corr = float(np.corrcoef(r_true_all, r_rec_all)[0, 1])

    return float(corr) if np.isfinite(corr) else 0.0


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
