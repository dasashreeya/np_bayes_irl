"""
expert_demos.py  |  Owner: P2  |  Built: Day 1
Generates expert trajectories from known reward weight vectors.
Ground truth: 2 reward types (likes red vs likes blue).
"""

import jax
import jax.numpy as jnp
import numpy as np

W1 = jnp.zeros(16).at[0].set(1.0).at[1].set(-1.0)   # red-lover
W2 = jnp.zeros(16).at[0].set(-1.0).at[1].set(1.0)   # blue-lover


def value_iteration_simple(R, T, gamma=0.95, n_iter=200):
    n_states = R.shape[0]
    V = jnp.zeros(n_states)
    for _ in range(n_iter):
        Q = R + gamma * jnp.einsum("san,n->sa", T, V)
        V = jnp.max(Q, axis=1)
    Q = R + gamma * jnp.einsum("san,n->sa", T, V)
    return Q


def generate_trajectory(rng_key, Q, T, length=20, beta=1.0):
    n_states = Q.shape[0]
    log_pi = jax.nn.log_softmax(beta * Q, axis=1)
    pi     = jnp.exp(log_pi)

    key, subkey = jax.random.split(rng_key)
    state = int(jax.random.randint(subkey, (), 0, n_states))

    trajectory = []
    for _ in range(length):
        key, subkey = jax.random.split(key)
        action_probs = np.array(pi[state])
        action = int(jax.random.choice(subkey, Q.shape[1], p=action_probs))
        trajectory.append((state, action))
        next_state = int(jnp.argmax(T[state, action]))
        state = next_state
    return trajectory


# ── signature matches build guide exactly ────────────────────────────────────
def generate_dataset(env, phi, T, n_per_type=10, seed=0):
    true_weights     = [W1, W2]
    trajectories     = []
    true_assignments = []

    rng = jax.random.PRNGKey(seed)

    for type_idx, w in enumerate(true_weights):
        R = env.reward_from_weights(w)
        Q = value_iteration_simple(R, T)

        for _ in range(n_per_type):
            rng, subkey = jax.random.split(rng)
            traj = generate_trajectory(subkey, Q, T)
            trajectories.append(traj)
            true_assignments.append(type_idx)

    print(f"Integration OK: {len(trajectories)} trajectories")
    return trajectories, true_weights, true_assignments


if __name__ == "__main__":
    from objectworld import ObjectWorld

    env  = ObjectWorld()
    phi  = env.features()
    T    = env.transitions()
    trajs, true_w, true_z = generate_dataset(env, phi, T)

    print(f"Labels:               {true_z}")
    print(f"First traj (5 steps): {trajs[0][:5]}")
    print(f"Traj length check:    {set(len(t) for t in trajs)}")
    print(f"Label counts:         {true_z.count(0)} red, {true_z.count(1)} blue")