"""mujoco_demos.py — Expert demonstration generation for MuJoCo.

Two agent types with different reward functions:
  Type 0: maximize forward velocity (standard HalfCheetah reward)
  Type 1: maximize energy efficiency (penalize control cost heavily)

Uses stable-baselines3 SAC if available; falls back to scripted heuristic
policies so the rest of the pipeline can run without a trained checkpoint.
"""
import os
import numpy as np
from mujoco_env import MuJoCoEnv, collect_trajectory

try:
    from stable_baselines3 import SAC
    SB3 = True
except ImportError:
    SB3 = False


# ---------------------------------------------------------------------------
# Heuristic policies (fallback when SB3 is not available / no checkpoint)
# ---------------------------------------------------------------------------

def _velocity_policy(obs):
    """Type 0: push forward — always apply max forward torque."""
    action_dim = 6  # HalfCheetah has 6 actuators
    return np.ones(action_dim, dtype=np.float32)


def _efficiency_policy(obs):
    """Type 1: energy efficient — small, smooth actions."""
    action_dim = 6
    # Gentle sinusoidal gait scaled down heavily
    t = float(obs[0]) if len(obs) > 0 else 0.0
    action = 0.1 * np.sin(np.linspace(0, np.pi, action_dim) + t).astype(np.float32)
    return action


# ---------------------------------------------------------------------------
# SB3-based policies (preferred)
# ---------------------------------------------------------------------------

def _load_sb3_policy(env_name, model_path=None):
    """Load a SAC policy. If model_path is None, return a freshly-init SAC
    (not trained) — good enough for interface verification."""
    import gymnasium as gym
    raw_env = gym.make(env_name)
    if model_path and os.path.exists(model_path):
        model = SAC.load(model_path, env=raw_env)
    else:
        model = SAC("MlpPolicy", raw_env, verbose=0)
    raw_env.close()
    return model


def _sb3_policy_fn(model):
    def _policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action.astype(np.float32)
    return _policy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_demos(env_name="HalfCheetah-v4", n_per_type=10, traj_len=200,
                   seed=0, model_paths=None):
    """
    Generate expert demonstrations for two agent types.

    Args:
        env_name:    gymnasium env id
        n_per_type:  trajectories per agent type
        traj_len:    max steps per trajectory
        seed:        RNG seed
        model_paths: dict {0: path_type0, 1: path_type1} or None for heuristics

    Returns:
        trajectories:    list of (obs, action) trajectory lists (len = 2*n_per_type)
        true_weights:    list of 2 placeholder weight arrays (None — use neural)
        true_assignments: list of int cluster labels
    """
    np.random.seed(seed)
    model_paths = model_paths or {}

    if SB3 and model_paths:
        m0 = _load_sb3_policy(env_name, model_paths.get(0))
        m1 = _load_sb3_policy(env_name, model_paths.get(1))
        p0 = _sb3_policy_fn(m0)
        p1 = _sb3_policy_fn(m1)
    else:
        p0 = _velocity_policy
        p1 = _efficiency_policy

    trajectories = []
    true_assignments = []

    for agent_type, policy_fn in enumerate([p0, p1]):
        for i in range(n_per_type):
            env = MuJoCoEnv(env_name, seed=seed + agent_type * n_per_type + i)
            traj = collect_trajectory(env, policy_fn, max_steps=traj_len)
            env.close()
            trajectories.append(traj)
            true_assignments.append(agent_type)

    true_weights = [None, None]  # neural reward — no single weight vector
    return trajectories, true_weights, true_assignments


if __name__ == "__main__":
    trajs, tw, ta = generate_demos(n_per_type=2, traj_len=20, seed=0)
    print(f"Total trajectories: {len(trajs)}")
    print(f"True assignments:   {ta}")
    print(f"Traj[0] length: {len(trajs[0])}, obs shape: {trajs[0][0][0].shape}")
    assert len(trajs) == 4
    assert ta == [0, 0, 1, 1]
    print("mujoco_demos.py smoke test PASSED")
