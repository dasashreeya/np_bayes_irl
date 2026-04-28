"""neural_value.py — Monte Carlo value estimation replacing tabular soft VI.

Interface: estimate_value(w, obs, env, n_rollouts=50) -> scalar
Uses the neural reward (neural_reward.compute_reward) to score rollouts.
"""
import numpy as np
import jax.numpy as jnp
from neural_reward import compute_reward


def _rollout(env, obs_start, policy_fn, w, gamma, max_steps):
    """Single Monte Carlo rollout from obs_start. Returns discounted return."""
    obs = obs_start
    total = 0.0
    discount = 1.0
    for _ in range(max_steps):
        action = policy_fn(obs)
        r = float(compute_reward(w, jnp.asarray(obs, dtype=jnp.float32)))
        total += discount * r
        obs, done = env.step(action)
        discount *= gamma
        if done:
            break
    return total


def estimate_value(w, obs, env, n_rollouts=50, gamma=0.99, max_steps=100,
                   policy_fn=None):
    """
    Monte Carlo value estimate V(obs) under reward w.

    Args:
        w:          flat neural reward parameter vector
        obs:        (obs_dim,) starting observation
        env:        MuJoCoEnv instance
        n_rollouts: number of MC rollouts
        gamma:      discount factor
        max_steps:  maximum steps per rollout
        policy_fn:  obs -> action; defaults to random policy

    Returns:
        scalar float estimate of V(obs)
    """
    if policy_fn is None:
        policy_fn = lambda o: env.sample_action()

    returns = []
    for _ in range(n_rollouts):
        # Reset env to a nearby state by resetting and stepping once with obs action
        env_obs = env.reset()
        ret = _rollout(env, env_obs, policy_fn, w, gamma, max_steps)
        returns.append(ret)
    return float(np.mean(returns))


def estimate_advantage(w, obs, action, env, n_rollouts=20, gamma=0.99,
                       max_steps=100, policy_fn=None):
    """
    A(s,a) = Q(s,a) - V(s) estimated via MC.
    Q(s,a): take action a from obs, then follow policy.
    V(s):   average over random actions from obs.
    """
    if policy_fn is None:
        policy_fn = lambda o: env.sample_action()

    # Q estimate: fix first action, then roll out
    q_returns = []
    for _ in range(n_rollouts):
        r0 = float(compute_reward(w, jnp.asarray(obs, dtype=jnp.float32)))
        next_obs, done = env.step(action)
        if done:
            q_returns.append(r0)
        else:
            rest = _rollout(env, next_obs, policy_fn, w, gamma, max_steps - 1)
            q_returns.append(r0 + gamma * rest)

    v = estimate_value(w, obs, env, n_rollouts=n_rollouts, gamma=gamma,
                       max_steps=max_steps, policy_fn=policy_fn)
    return float(np.mean(q_returns)) - v


if __name__ == "__main__":
    from mujoco_env import MuJoCoEnv
    from neural_reward import init_weights
    import jax

    env = MuJoCoEnv("HalfCheetah-v4")
    rng = jax.random.PRNGKey(42)
    w = init_weights(rng, env.obs_dim)
    obs = env.reset()

    v = estimate_value(w, obs, env, n_rollouts=5, max_steps=20)
    print(f"V(obs) = {v:.4f}")
    assert np.isfinite(v), "value must be finite"

    a = estimate_advantage(w, obs, env.sample_action(), env,
                           n_rollouts=5, max_steps=20)
    print(f"A(obs, action) = {a:.4f}")
    assert np.isfinite(a), "advantage must be finite"

    env.close()
    print("neural_value.py smoke test PASSED")
