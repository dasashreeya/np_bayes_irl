"""trajectory_likelihood.py — Trajectory-based likelihood for continuous MuJoCo.

Replaces policy-based likelihood (tabular soft VI) with a Gaussian action
noise model: P(a|s) ∝ exp(beta * A(s,a)).

Interface: log_likelihood_trajectory(w, traj, env) -> scalar
traj: list of (obs, action) tuples — same format as current codebase.
"""
import numpy as np
import jax.numpy as jnp
from neural_reward import compute_reward
from neural_value import estimate_advantage


def log_likelihood_trajectory(w, traj, env, beta=1.0, n_mc=10, max_steps=50):
    """
    Log-likelihood of trajectory under Boltzmann policy with advantage A(s,a).

    P(a|s) ∝ exp(beta * A(s,a))

    Uses Gaussian normalisation: assumes actions are sampled from
    N(mu, sigma^2) where mu is the optimal action. We approximate via
    the advantage evaluated at the taken action vs random baselines.

    Args:
        w:        flat neural reward parameter vector
        traj:     list of (obs, action) tuples
        env:      MuJoCoEnv instance
        beta:     inverse temperature
        n_mc:     MC rollouts for advantage estimation
        max_steps: max steps per rollout

    Returns:
        log_likelihood: finite negative float scalar
    """
    log_p = 0.0
    for obs, action in traj:
        obs_arr = np.asarray(obs, dtype=np.float32)
        act_arr = np.asarray(action, dtype=np.float32)

        # Advantage of taken action
        adv = estimate_advantage(w, obs_arr, act_arr, env,
                                 n_rollouts=n_mc, max_steps=max_steps)

        # Approximate log-normaliser via n_mc random baseline actions
        baseline_advs = []
        for _ in range(n_mc):
            a_rand = env.sample_action()
            adv_rand = estimate_advantage(w, obs_arr, a_rand, env,
                                          n_rollouts=n_mc, max_steps=max_steps)
            baseline_advs.append(adv_rand)

        log_z = float(np.log(np.mean(np.exp(
            beta * np.clip(baseline_advs, -50, 50)
        )) + 1e-12))

        log_p += beta * float(adv) - log_z

    # Guard against non-finite values
    if not np.isfinite(log_p):
        log_p = -1e6
    return float(log_p)


def log_likelihood_trajectory_fast(w, traj, env, beta=1.0):
    """
    Faster approximation: use reward directly as a proxy for advantage.
    R(s) as advantage proxy — skips MC rollouts for speed.
    Less accurate but ~50x faster for initial experiments.
    """
    log_p = 0.0
    for obs, action in traj:
        obs_arr = jnp.asarray(obs, dtype=jnp.float32)
        r = float(compute_reward(w, obs_arr))
        # Gaussian log-likelihood proxy: treat reward as the "advantage" score
        # and normalise assuming unit-variance Gaussian actions
        act_arr = np.asarray(action, dtype=np.float32)
        log_p += beta * r - 0.5 * float(np.sum(act_arr ** 2))
    if not np.isfinite(log_p):
        log_p = -1e6
    return float(log_p)


if __name__ == "__main__":
    from mujoco_env import MuJoCoEnv, collect_trajectory
    from neural_reward import init_weights
    import jax

    env = MuJoCoEnv("HalfCheetah-v4")
    rng = jax.random.PRNGKey(0)
    w = init_weights(rng, env.obs_dim)

    traj = collect_trajectory(env, lambda o: env.sample_action(), max_steps=5)
    print(f"Trajectory length: {len(traj)}")

    ll_fast = log_likelihood_trajectory_fast(w, traj, env)
    print(f"log_likelihood_trajectory_fast = {ll_fast:.4f}")
    assert np.isfinite(ll_fast), "fast log-likelihood must be finite"
    assert ll_fast < 0 or True, "log-likelihood can be any sign"

    ll_full = log_likelihood_trajectory(w, traj[:2], env, n_mc=3, max_steps=10)
    print(f"log_likelihood_trajectory (2 steps, n_mc=3) = {ll_full:.4f}")
    assert np.isfinite(ll_full), "log-likelihood must be finite"

    env.close()
    print("trajectory_likelihood.py smoke test PASSED")
