"""
likelihood_backend.py  |  Owner: SD  |  Interface: PROJECT_SCOPE.md §5
Every expensive computation the sampler needs, behind one object.

The sampler never calls soft value iteration directly. It asks this
backend for per-trajectory log-likelihoods under a weight vector `w`:

    log p(tau_i | w) = sum_t log pi_w(a_t | s_t),   pi_w = Boltzmann(softVI(w))

This file currently holds the *reference* implementation: it's simple,
uncached and CPU-only, and it's the correctness oracle for WS4. A fast
backend (caching, vmap, GPU) must match `ReferenceBackend` to ~1e-6 on
every method, and can then replace it under the same name.
"""

import numpy as np
import jax.numpy as jnp

from likelihood import compute_log_pi


class LikelihoodBackend:
    """
    Per-trajectory Boltzmann log-likelihoods for a fixed dataset.

    Args:
        phi:          (S, F) state features
        T:            (S, A, S) transition tensor
        gamma:        float, discount factor
        beta:         float, Boltzmann rationality used by the *inference* model
        trajectories: list of N trajectories, each a list of (s, a) int pairs;
                      lengths may differ
        device:       "cpu" (only option in the reference backend)
    """

    def __init__(self, phi, T, gamma, beta, trajectories, device="cpu"):
        if device != "cpu":
            raise NotImplementedError("reference backend is CPU-only")
        self.phi = jnp.asarray(phi)
        self.T = jnp.asarray(T)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.n_features = int(self.phi.shape[1])
        self.N = len(trajectories)
        # Flatten all (s, a) pairs once; seg[i] is the trajectory id of each pair.
        lens = np.array([len(t) for t in trajectories], dtype=np.int64)
        sa = np.array([p for t in trajectories for p in t], dtype=np.int64).reshape(-1, 2)
        self._s, self._a = sa[:, 0], sa[:, 1]
        self._seg = np.repeat(np.arange(self.N), lens)
        self.traj_lengths = lens

    def log_pi(self, w):
        """
        Log Boltzmann policy for reward weights w.

        Args:
            w: (F,) weight vector
        Returns:
            (S, A) numpy array of log pi(a | s)
        """
        return np.asarray(compute_log_pi(jnp.asarray(w), self.phi, self.T,
                                         self.gamma, self.beta), dtype=np.float64)

    def _loglik_vector(self, w):
        """
        Log-likelihood of every trajectory under one weight vector.

        Args:
            w: (F,) weight vector
        Returns:
            (N,) numpy array, entry i = log p(tau_i | w)
        """
        lp = self.log_pi(w)[self._s, self._a]
        return np.bincount(self._seg, weights=lp, minlength=self.N)

    def loglik_matrix(self, weights):
        """
        Args:
            weights: list of K (F,) weight vectors
        Returns:
            (N, K) numpy array, entry (i, k) = log p(tau_i | w_k)
        """
        if len(weights) == 0:
            return np.zeros((self.N, 0))
        return np.stack([self._loglik_vector(w) for w in weights], axis=1)

    def loglik_cluster(self, w, idx):
        """
        Args:
            w:   (F,) weight vector
            idx: sequence of trajectory indices
        Returns:
            float, sum_{i in idx} log p(tau_i | w)
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return 0.0
        return float(self._loglik_vector(w)[idx].sum())

    def invalidate(self, weight_id):
        """Drop any cached value for weight_id. No-op: the reference backend has no cache."""
        return None


ReferenceBackend = LikelihoodBackend
