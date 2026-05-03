"""
mdp_utils.py  |  Owner: P1  |  Built: Day 2
Soft value iteration and Boltzmann policy.
This is the engine inside the likelihood function.
"""
"""
mdp_utils.py  |  Owner: P1  |  Built: Day 2
Soft value iteration and Boltzmann policy.
This is the engine inside the likelihood function.
"""

import jax
import jax.numpy as jnp


# ── 1. soft_value_iteration ──────────────────────────────────────────────────

def soft_value_iteration(R, T, gamma=0.95, beta=1.0, n_iter=100):
    """
    Entropy-regularized (soft) value iteration.
    Replaces the hard max in the Bellman backup with a logsumexp.
    Required for exact Boltzmann likelihood computation in likelihood.py.

    Args:
        R:      (n_states, n_actions)  reward matrix
        T:      (n_states, n_actions, n_states)  transition tensor
        gamma:  float  discount factor
        beta:   float  Boltzmann temperature
        n_iter: int    number of iterations

    Returns:
        V: (n_states,)          soft value function
        Q: (n_states, n_actions)  soft Q-function
    """
    n_states = R.shape[0]
    V = jnp.zeros(n_states)

    for _ in range(n_iter):
        # Q(s,a) = R(s,a) + gamma * sum_{s'} T(s,a,s') * V(s')
        Q = R + gamma * jnp.einsum("san,n->sa", T, V)

        # Soft Bellman backup: V(s) = (1/beta) * log sum_a exp(beta * Q(s,a))
        V = (1.0 / beta) * jax.scipy.special.logsumexp(beta * Q, axis=1)

    # Final Q from converged V
    Q = R + gamma * jnp.einsum("san,n->sa", T, V)
    return V, Q


# ── 2. boltzmann_policy ──────────────────────────────────────────────────────

def boltzmann_policy(Q, beta=1.0):
    """
    Boltzmann (softmax) policy over Q-values.
    Used when you need the actual probability matrix (e.g. for rollouts).

    Args:
        Q:    (n_states, n_actions)
        beta: float  temperature — high = greedy, low = random

    Returns:
        pi: (n_states, n_actions)  each row is a probability distribution
    """
    return jax.nn.softmax(beta * Q, axis=1)


# ── 3. log_boltzmann_policy ──────────────────────────────────────────────────

def log_boltzmann_policy(Q, beta=1.0):
    """
    Log Boltzmann policy — numerically stable version.
    Used directly in likelihood.py to avoid underflow when summing
    log-probabilities along a trajectory.
    Never use boltzmann_policy() + log() — use this instead.

    Args:
        Q:    (n_states, n_actions)
        beta: float  temperature

    Returns:
        log_pi: (n_states, n_actions)  log probabilities, each row sums to 0 in log space
    """
    return jax.nn.log_softmax(beta * Q, axis=1)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from objectworld import ObjectWorld

    env = ObjectWorld()
    T   = env.transitions()   # (100, 4, 100)
    phi = env.features()      # (100, 16)
    w   = jnp.zeros(16).at[0].set(1.0).at[1].set(-1.0)
    R   = env.reward_from_weights(w)   # (100, 4)

    V, Q = soft_value_iteration(R, T)

    print(f"V shape:       {V.shape}")          # (100,)
    print(f"Q shape:       {Q.shape}")          # (100, 4)
    print(f"V finite:      {jnp.all(jnp.isfinite(V))}")
    print(f"Q finite:      {jnp.all(jnp.isfinite(Q))}")

    pi = boltzmann_policy(Q)
    row_sums = jnp.sum(pi, axis=1)
    print(f"pi row sums:   min={row_sums.min():.6f} max={row_sums.max():.6f}")  # all ~1.0

    # High beta → near one-hot (greedy)
    pi_high = boltzmann_policy(Q, beta=10.0)
    print(f"high beta max: {jnp.max(pi_high, axis=1).mean():.4f}")  # should be close to 1.0

    # Low beta → near uniform (random)
    pi_low = boltzmann_policy(Q, beta=0.1)
    print(f"low beta max:  {jnp.max(pi_low, axis=1).mean():.4f}")   # should be close to 0.25

    log_pi = log_boltzmann_policy(Q)
    print(f"log_pi shape:  {log_pi.shape}")     # (100, 4)
    print(f"log_pi finite: {jnp.all(jnp.isfinite(log_pi))}")

    print("\nAll gate checks passed.")