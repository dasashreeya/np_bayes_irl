"""
maxent_irl.py  |  Owner: P1  |  Built: Day 2
MaxEnt IRL baseline. Finds w such that policy feature expectations match expert.
This is the comparison baseline -- your method should beat this.
"""
import jax
import jax.numpy as jnp
import numpy as np
from mdp_utils import soft_value_iteration, boltzmann_policy


# def feature_expectations(trajectories, phi):
#     """
#     Empirical feature counts from expert trajectories.
#     trajectories: list of (states, actions) tuples, states is list of int indices
#     phi: (n_states, n_features)
#     Returns: (n_features,) mean feature vector
#     """
#     n_features = phi.shape[1]
#     mu = np.zeros(n_features)
#     total = 0
#     for (states, actions) in trajectories:
#         for s in states:
#             mu += np.array(phi[s])
#             total += 1
#     return mu / total if total > 0 else mu


def feature_expectations(trajectories, phi):
    """
    Empirical feature counts from expert trajectories.
    trajectories: list of trajectories, each a list of (state, action) tuples
    phi: (n_states, n_features)
    Returns: (n_features,) mean feature vector
    """
    n_features = phi.shape[1]
    mu = np.zeros(n_features)
    total = 0
    for traj in trajectories:
        for (s, a) in traj:
            mu += np.array(phi[s])
            total += 1
    return mu / total if total > 0 else mu


def policy_feature_expectations(pi, phi, T, gamma=0.95, n_iter=50):
    """
    Expected feature counts under policy pi via discounted state visitation.
    pi:  (n_states, n_actions)
    phi: (n_states, n_features)
    T:   (n_states, n_actions, n_states) transition tensor
    Returns: (n_features,)
    """
    n_states, n_features = phi.shape
    n_actions = pi.shape[1]

    # Uniform initial state distribution
    d = np.ones(n_states) / n_states

    mu = np.zeros(n_features)
    d_t = d.copy()

    for _ in range(n_iter):
        # Feature counts at this timestep
        mu += np.array(phi.T @ d_t)          # (n_features,)

        # Transition: d_{t+1}(s') = sum_{s,a} pi(a|s) * T(s,a,s') * d_t(s)
        d_next = np.zeros(n_states)
        for a in range(n_actions):
            T_a = np.array(T[:, a, :])       # (n_states, n_states)
            pi_a = np.array(pi[:, a])        # (n_states,)
            d_next += T_a.T @ (pi_a * d_t)

        d_t = gamma * d_next

    return mu


def maxent_irl(trajectories, phi, T, gamma=0.95, beta=5.0, lr=0.05, n_iters=200):
    """
    MaxEnt IRL via gradient ascent to match feature expectations.
    Gradient = mu_expert - mu_policy
    Returns recovered weight vector w (n_features,).
    """
    n_features = phi.shape[1]
    w = np.zeros(n_features)

    mu_expert = feature_expectations(trajectories, phi)

    for it in range(n_iters):
        w_jnp = jnp.array(w)

        # Compute reward and policy under current w
        R = np.array(phi @ w)                          # (n_states,)
        R_sa = np.tile(R[:, None], (1, T.shape[1]))    # (n_states, n_actions)
        _, Q = soft_value_iteration(R_sa, T, gamma)
        pi = np.array(boltzmann_policy(Q, beta))       # (n_states, n_actions)

        # Expected feature counts under current policy
        mu_policy = policy_feature_expectations(pi, phi, T, gamma)

        # Gradient ascent
        grad = mu_expert - mu_policy
        w = w + lr * grad

        grad_norm = np.linalg.norm(grad)
        if it % 20 == 0:
            print(f"Iter {it:>4d} | grad_norm: {grad_norm:.4f} | w[0]: {w[0]:.4f}")

        if grad_norm < 1e-4:
            print(f"Converged at iter {it}")
            break

    return w