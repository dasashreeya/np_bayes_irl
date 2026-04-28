"""neural_reward.py — Small MLP reward function using JAX.

Replace linear R(s,a) = w·phi(s) with a 2-layer MLP.
Weight vector w is the flattened network parameters.
Interface: compute_reward(w, obs) -> scalar
"""
import jax
import jax.numpy as jnp
import numpy as np

HIDDEN = 64


def _net_shapes(obs_dim):
    """Return list of (in, out) layer shapes for a [obs_dim -> 64 -> 64 -> 1] MLP."""
    return [(obs_dim, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, 1)]


def param_count(obs_dim):
    """Total number of scalar parameters in the network."""
    total = 0
    for (i, o) in _net_shapes(obs_dim):
        total += i * o + o  # weights + bias
    return total


def init_weights(rng_key, obs_dim):
    """Return a flat JAX array of initialised network parameters (Glorot uniform)."""
    params = []
    for (fan_in, fan_out) in _net_shapes(obs_dim):
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        rng_key, sk = jax.random.split(rng_key)
        W = jax.random.uniform(sk, shape=(fan_in, fan_out),
                               minval=-limit, maxval=limit)
        b = jnp.zeros(fan_out)
        params.extend([W.ravel(), b])
    return jnp.concatenate(params)


def _unpack(w, obs_dim):
    """Split flat weight vector back into layer (W, b) pairs."""
    shapes = _net_shapes(obs_dim)
    layers = []
    idx = 0
    for (fan_in, fan_out) in shapes:
        n_w = fan_in * fan_out
        W = w[idx: idx + n_w].reshape(fan_in, fan_out)
        idx += n_w
        b = w[idx: idx + fan_out]
        idx += fan_out
        layers.append((W, b))
    return layers


def compute_reward(w, obs, obs_dim=None):
    """
    Forward pass: returns scalar reward R(obs).

    Args:
        w:       flat JAX array of network parameters (length param_count(obs_dim))
        obs:     (obs_dim,) observation vector
        obs_dim: inferred from obs if not supplied
    """
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs_dim is None:
        obs_dim = obs.shape[0]
    layers = _unpack(w, obs_dim)
    x = obs
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = jax.nn.tanh(x)
    return x[0]  # scalar


def compute_reward_batch(w, obs_batch, obs_dim=None):
    """Vectorised version over a batch of observations."""
    obs_batch = jnp.asarray(obs_batch, dtype=jnp.float32)
    if obs_dim is None:
        obs_dim = obs_batch.shape[-1]
    return jax.vmap(lambda o: compute_reward(w, o, obs_dim))(obs_batch)


if __name__ == "__main__":
    obs_dim = 17  # HalfCheetah-v4
    rng = jax.random.PRNGKey(0)
    w = init_weights(rng, obs_dim)
    print(f"param_count={param_count(obs_dim)}, w.shape={w.shape}")
    obs = jnp.zeros(obs_dim)
    r = compute_reward(w, obs)
    print(f"compute_reward(w, zeros) = {r:.4f}")
    assert jnp.isfinite(r), "reward must be finite"
    obs_batch = jnp.ones((5, obs_dim))
    rs = compute_reward_batch(w, obs_batch)
    print(f"batch rewards: {rs}")
    assert rs.shape == (5,)
    print("neural_reward.py smoke test PASSED")
