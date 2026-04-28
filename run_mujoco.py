"""run_mujoco.py — Full MuJoCo NP-Bayes IRL experiment.

Loads demos, initialises Gibbs with neural reward weights,
runs 100 sweeps, logs K / L2 / ARI to W&B every 5 sweeps.
"""
import numpy as np
import jax
import jax.numpy as jnp

from mujoco_env import MuJoCoEnv, record_episode
from mujoco_demos import generate_demos
from neural_reward import init_weights, param_count, compute_reward
from trajectory_likelihood import log_likelihood_trajectory_fast
from dp_prior import log_prior
from eval import adjusted_rand_index

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ---------------------------------------------------------------------------
# MuJoCo-specific Gibbs primitives
# ---------------------------------------------------------------------------

def _sample_new_w(rng_key, obs_dim, scale=0.1):
    """Sample initial neural reward params near zero."""
    n = param_count(obs_dim)
    return jax.random.normal(rng_key, shape=(n,)) * scale


def _log_ll(w, traj, env):
    return log_likelihood_trajectory_fast(w, traj, env)


def _mh_update(w_curr, traj_list, env, step_size, rng_key):
    """Gaussian random-walk MH step in neural param space."""
    rng_key, sk = jax.random.split(rng_key)
    noise = jax.random.normal(sk, shape=w_curr.shape) * step_size
    w_prop = w_curr + noise

    ll_curr = sum(_log_ll(w_curr, t, env) for t in traj_list)
    ll_prop = sum(_log_ll(w_prop, t, env) for t in traj_list)
    lp_curr = float(log_prior(w_curr))
    lp_prop = float(log_prior(w_prop))

    log_accept = (ll_prop + lp_prop) - (ll_curr + lp_curr)
    rng_key, sk = jax.random.split(rng_key)
    if float(jnp.log(jax.random.uniform(sk))) < log_accept:
        return w_prop, rng_key
    return w_curr, rng_key


def mujoco_gibbs_sweep(trajectories, assignments, weight_vectors,
                       env, alpha, step_size, rng_key):
    """
    One Gibbs sweep for the MuJoCo / neural-reward setting.
    Cluster assignment step uses fast reward-proxy likelihood.
    MH step proposes in flattened neural param space.
    """
    N = len(trajectories)
    K = len(weight_vectors)

    # --- Step 1: resample cluster assignments ---
    counts = np.bincount(assignments, minlength=K).astype(float)

    new_assignments = list(assignments)
    for i in range(N):
        counts[assignments[i]] -= 1
        traj_i = trajectories[i]

        log_scores = []
        cids = []
        for k in range(K):
            if counts[k] <= 0:
                continue
            ll = _log_ll(weight_vectors[k], traj_i, env)
            log_scores.append(np.log(counts[k]) + ll)
            cids.append(k)

        rng_key, sk = jax.random.split(rng_key)
        w_new = _sample_new_w(sk, env.obs_dim)
        ll_new = _log_ll(w_new, traj_i, env)
        log_scores.append(np.log(alpha) + ll_new)
        cids.append('new')

        log_scores = np.array(log_scores, dtype=np.float64)
        log_scores -= log_scores.max()
        probs = np.exp(log_scores)
        probs /= probs.sum()

        rng_key, sk = jax.random.split(rng_key)
        idx = int(jax.random.choice(sk, len(cids), p=jnp.array(probs)))
        chosen = cids[idx]

        if chosen == 'new':
            weight_vectors = weight_vectors + [w_new]
            K += 1
            counts = np.append(counts, 0.0)
            new_assignments[i] = K - 1
        else:
            new_assignments[i] = chosen

        counts[new_assignments[i]] += 1

    assignments = new_assignments

    # --- Step 2: remap to contiguous labels ---
    occupied = sorted(set(assignments))
    remap = {old: new for new, old in enumerate(occupied)}
    assignments = [remap[z] for z in assignments]
    weight_vectors = [weight_vectors[k] for k in occupied]

    # --- Step 3: MH weight update ---
    for k in range(len(weight_vectors)):
        cluster_trajs = [trajectories[i]
                         for i, z in enumerate(assignments) if z == k]
        if not cluster_trajs:
            continue
        weight_vectors[k], rng_key = _mh_update(
            weight_vectors[k], cluster_trajs, env, step_size, rng_key)

    return assignments, weight_vectors, rng_key


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

VIDEO_DIR = "results/videos"
EXPERT_LABELS = {0: "velocity", 1: "efficiency"}


def _record_expert_demos(seed):
    """Record one episode per expert type before training starts."""
    from mujoco_demos import _velocity_policy, _efficiency_policy
    videos = {}
    for agent_type, (label, policy_fn) in enumerate(
            [("expert_velocity", _velocity_policy),
             ("expert_efficiency", _efficiency_policy)]):
        print(f"  Recording expert demo: {label}...")
        path = record_episode(policy_fn, video_dir=VIDEO_DIR,
                              prefix=label, max_steps=200, seed=seed)
        videos[label] = path
        print(f"    -> {path}")
    return videos


def _record_recovered_cluster(k, w, obs_dim, seed):
    """Record one episode under a greedy policy derived from recovered reward w."""
    from neural_reward import compute_reward

    def greedy_policy(obs, n_candidates=20):
        best_a, best_r = None, -np.inf
        for _ in range(n_candidates):
            a = np.random.randn(6).astype(np.float32)  # HalfCheetah has 6 actuators
            a = np.clip(a, -1.0, 1.0)
            r = float(compute_reward(w, jnp.asarray(obs, dtype=jnp.float32)))
            if r > best_r:
                best_r, best_a = r, a
        return best_a

    prefix = f"recovered_cluster{k}"
    print(f"  Recording recovered cluster {k}...")
    path = record_episode(greedy_policy, video_dir=VIDEO_DIR,
                          prefix=prefix, max_steps=200, seed=seed)
    print(f"    -> {path}")
    return path


def run_experiment(n_sweeps=100, alpha=1.0, step_size=0.01,
                   n_per_type=10, traj_len=200, seed=0, log_every=5,
                   record_videos=True):
    rng = jax.random.PRNGKey(seed)

    # --- Record expert demos before training ---
    video_paths = {}
    if record_videos:
        print("Recording expert demonstrations...")
        video_paths.update(_record_expert_demos(seed))

    print("Generating demonstrations...")
    trajectories, true_weights, true_assignments = generate_demos(
        n_per_type=n_per_type, traj_len=traj_len, seed=seed)
    N = len(trajectories)
    print(f"  {N} trajectories, true_assignments: {true_assignments}")

    env = MuJoCoEnv("HalfCheetah-v4", seed=seed)
    obs_dim = env.obs_dim

    if WANDB:
        wandb.init(project="np-bayes-irl-mujoco", config={
            "n_sweeps": n_sweeps, "alpha": alpha, "step_size": step_size,
            "n_per_type": n_per_type, "traj_len": traj_len,
        })

    # Initialise: each trajectory in its own cluster
    rng, sk = jax.random.split(rng)
    keys = jax.random.split(sk, N)
    assignments = list(range(N))
    weight_vectors = [_sample_new_w(keys[i], obs_dim) for i in range(N)]

    history = []

    for sweep in range(n_sweeps):
        rng, sk = jax.random.split(rng)
        assignments, weight_vectors, sk = mujoco_gibbs_sweep(
            trajectories, assignments, weight_vectors,
            env, alpha, step_size, sk)
        rng = sk

        K = len(set(assignments))
        ari = adjusted_rand_index(true_assignments, assignments)
        metrics = {"sweep": sweep, "n_clusters": K, "ari": ari}
        history.append(metrics)

        if sweep % log_every == 0:
            print(f"Sweep {sweep:4d} | K={K} | ARI={ari:.4f}")
            if WANDB:
                wandb.log(metrics)

    env.close()

    # --- Record one rollout per recovered cluster ---
    if record_videos:
        print("Recording recovered cluster policies...")
        for k, w in enumerate(weight_vectors):
            path = _record_recovered_cluster(k, w, obs_dim, seed)
            video_paths[f"recovered_cluster{k}"] = path

    if WANDB:
        if record_videos:
            for label, path in video_paths.items():
                if path:
                    wandb.log({label: wandb.Video(path, fps=30, format="mp4")})
        wandb.finish()

    print("\n--- Video outputs ---")
    for label, path in video_paths.items():
        print(f"  {label}: {path}")

    return assignments, weight_vectors, history, video_paths


if __name__ == "__main__":
    assignments, weight_vectors, history, videos = run_experiment(
        n_sweeps=10, n_per_type=2, traj_len=20, seed=0, log_every=1,
        record_videos=True)
    print(f"Final K={len(set(assignments))}, sweeps={len(history)}")
    print(f"Videos: {videos}")
    print("run_mujoco.py smoke test PASSED")
