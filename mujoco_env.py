"""mujoco_env.py — MuJoCo wrapper with same interface as ObjectWorld."""
import os
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    GYMNASIUM = True
except ImportError:
    GYMNASIUM = False


class MuJoCoEnv:
    """
    Wraps a MuJoCo gymnasium environment and exposes the same interface
    used by ObjectWorld: features(), step(), reset().
    State space is continuous — no transition matrix T.
    """

    def __init__(self, env_name="HalfCheetah-v4", seed=0):
        if not GYMNASIUM:
            raise ImportError("gymnasium[mujoco] required: pip install gymnasium[mujoco]")
        self.env_name = env_name
        self.seed = seed
        self._env = gym.make(env_name)
        obs, _ = self._env.reset(seed=seed)
        self.obs_dim = obs.shape[0]
        self.action_dim = self._env.action_space.shape[0]
        self.n_features = self.obs_dim  # phi(s) = raw obs

    def reset(self):
        obs, _ = self._env.reset()
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, self._env.action_space.low,
                         self._env.action_space.high)
        obs, _reward, terminated, truncated, _info = self._env.step(action)
        done = terminated or truncated
        return np.array(obs, dtype=np.float32), done

    def features(self, obs):
        """phi(s) — raw observation as feature vector."""
        return np.array(obs, dtype=np.float32)

    def sample_action(self):
        return self._env.action_space.sample()

    def close(self):
        self._env.close()


def collect_trajectory(env, policy_fn, max_steps=200):
    """Roll out policy_fn(obs) -> action. Returns list of (obs, action) tuples."""
    traj = []
    obs = env.reset()
    for _ in range(max_steps):
        action = policy_fn(obs)
        traj.append((obs, action))
        obs, done = env.step(action)
        if done:
            break
    return traj


def record_episode(policy_fn, video_dir="results/videos", prefix="episode",
                   env_name="HalfCheetah-v4", max_steps=200, seed=0):
    """
    Record a single episode to an MP4 file and return the file path.

    Args:
        policy_fn:  obs (np.ndarray) -> action (np.ndarray)
        video_dir:  directory to save the video
        prefix:     filename prefix (e.g. "expert_type0", "recovered_cluster1")
        env_name:   gymnasium env id
        max_steps:  episode length cap
        seed:       env reset seed

    Returns:
        path to the saved MP4, or None if recording failed
    """
    if not GYMNASIUM:
        raise ImportError("gymnasium[mujoco] required")

    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, name_prefix=prefix,
                      episode_trigger=lambda _ep: True, disable_logger=True)

    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        action = policy_fn(np.array(obs, dtype=np.float32))
        obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    env.close()

    # Return the most recently written mp4 in video_dir
    mp4s = sorted(
        [f for f in os.listdir(video_dir) if f.endswith(".mp4")],
        key=lambda f: os.path.getmtime(os.path.join(video_dir, f))
    )
    return os.path.join(video_dir, mp4s[-1]) if mp4s else None


if __name__ == "__main__":
    env = MuJoCoEnv("HalfCheetah-v4")
    print(f"obs_dim={env.obs_dim}, action_dim={env.action_dim}")
    traj = collect_trajectory(env, lambda obs: env.sample_action(), max_steps=10)
    print(f"Collected {len(traj)} steps, obs shape: {traj[0][0].shape}")
    phi0 = env.features(traj[0][0])
    print(f"phi(s) shape: {phi0.shape}, sample: {phi0[:4]}")
    env.close()

    path = record_episode(lambda obs: np.zeros(6), prefix="smoke_test",
                          video_dir="results/videos", max_steps=10)
    print(f"Recorded video: {path}")
    assert path is not None and path.endswith(".mp4")
    print("mujoco_env.py smoke test PASSED")
