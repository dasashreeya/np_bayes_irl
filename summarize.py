"""
summarize.py  |  Owner: SA  |  Interface: PROJECT_SCOPE.md §5
Turn the Traces of several chains into one PosteriorSummary.

Everything here is label-invariant (posterior over K, the posterior
similarity matrix, joint log-posterior), so label switching across
samples or chains doesn't matter. No ground truth is used.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PosteriorSummary:
    """
    k_values:        (M,) distinct K values seen, ascending
    k_probs:         (M,) posterior probability of each, pooled over chains
    psm:             (N, N) posterior similarity matrix, P(z_i = z_j | D)
    point_clustering:(N,) partition minimizing expected Binder loss under the PSM
    point_weights:   list of (F,) arrays, one per point cluster: the posterior mean
                     of the weight attached to that cluster's members
    map_assignments: (N,) assignments of the highest joint log-posterior sample
    map_weights:     list of (F,) arrays for the MAP sample
    map_log_post:    float
    rhat:            dict, {"K": float, "log_post": float}, rank-normalized split R-hat
    ess:             dict, {"K": float, "log_post": float}, bulk ESS pooled over chains
    accept_rate:     float, mean post-burn-in MH acceptance over chains
    n_chains:        int
    n_kept:          int, samples per chain
    """
    k_values: np.ndarray
    k_probs: np.ndarray
    psm: np.ndarray
    point_clustering: np.ndarray
    point_weights: list
    map_assignments: np.ndarray
    map_weights: list
    map_log_post: float
    rhat: dict
    ess: dict
    accept_rate: float
    n_chains: int
    n_kept: int


def summarize(traces):
    """
    Args:
        traces: list of Trace, one per chain, all on the same dataset
    Returns:
        PosteriorSummary
    """
    raise NotImplementedError
