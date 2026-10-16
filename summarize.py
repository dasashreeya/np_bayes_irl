"""
summarize.py  |  Owner: SA  |  Interface: PROJECT_SCOPE.md §5
Turn the Traces of several chains into one PosteriorSummary.

Everything here is label-invariant (posterior over K, the posterior
similarity matrix, joint log-posterior), so label switching across
samples or chains doesn't matter. No ground truth is used.
"""

from dataclasses import dataclass

import numpy as np

from diagnostics import ess_bulk, split_rhat


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


def posterior_similarity(assignments):
    """
    Args:
        assignments: (S, N) int, one partition per sample
    Returns:
        (N, N) float, fraction of samples in which i and j share a cluster
    """
    A = np.asarray(assignments)
    S, N = A.shape
    psm = np.zeros((N, N))
    for z in A:
        onehot = np.zeros((N, int(z.max()) + 1))
        onehot[np.arange(N), z] = 1.0
        psm += onehot @ onehot.T
    return psm / S


def binder_loss(z, psm):
    """
    Expected Binder loss (equal costs) of partition z, up to a constant:
    sum over pairs i < j in the same cluster of (1/2 - psm_ij).
    """
    z = np.asarray(z)
    same = z[:, None] == z[None, :]
    return float(np.triu(np.where(same, 0.5 - psm, 0.0), k=1).sum())


def _relabel(z):
    """Labels in order of first appearance, so equal partitions compare equal."""
    _, first, inv = np.unique(z, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inv]


def minimize_binder(psm, candidates):
    """
    Partition minimizing expected Binder loss: best of the candidate partitions,
    then greedy single-item moves (to another cluster or a new one) until no
    move lowers the loss (Dahl et al. 2022 describe the same local search).

    Args:
        psm:        (N, N) posterior similarity matrix
        candidates: (S, N) int partitions to start from, e.g. the sampled ones
    Returns:
        (N,) int partition, labels in order of first appearance
    """
    uniq = np.unique(np.array([_relabel(c) for c in candidates]), axis=0)
    z = uniq[int(np.argmin([binder_loss(c, psm) for c in uniq]))].copy()
    gain = 0.5 - psm  # cost of putting i and j together
    np.fill_diagonal(gain, 0.0)
    improved = True
    while improved:
        improved = False
        for i in range(z.size):
            labels = np.unique(z)
            cost = np.array([gain[i, z == k].sum() for k in labels] + [0.0])
            best = int(np.argmin(cost))
            if cost[best] < cost[np.searchsorted(labels, z[i])] - 1e-12:
                z[i] = labels[best] if best < labels.size else labels.max() + 1
                improved = True
    return _relabel(z)


def _point_weights(point, assignments, weights):
    """
    Posterior mean weight for each point cluster. In each sample, a point
    cluster takes the weight of the sampled cluster holding most of its
    members. Label switching doesn't matter because the mapping is redone
    per sample.
    """
    out = []
    for c in range(int(point.max()) + 1):
        members = point == c
        acc = []
        for z, ws in zip(assignments, weights):
            k = np.bincount(z[members]).argmax()
            acc.append(np.asarray(ws[k], dtype=float))
        out.append(np.mean(acc, axis=0))
    return out


def summarize(traces):
    """
    Args:
        traces: list of Trace, one per chain, all on the same dataset and
                with the same number of kept samples
    Returns:
        PosteriorSummary
    """
    n_kept = {len(t.K) for t in traces}
    if len(n_kept) != 1:
        raise ValueError(f"chains kept different numbers of samples: {sorted(n_kept)}")
    n_kept = n_kept.pop()

    A = np.concatenate([np.asarray(t.assignments) for t in traces], axis=0)
    W = [ws for t in traces for ws in t.weights]
    K = np.stack([np.asarray(t.K) for t in traces])
    LP = np.stack([np.asarray(t.log_post) for t in traces])

    k_values, k_counts = np.unique(K, return_counts=True)
    psm = posterior_similarity(A)
    point = minimize_binder(psm, A)
    s_map = int(np.argmax(LP.ravel()))

    return PosteriorSummary(
        k_values=k_values,
        k_probs=k_counts / k_counts.sum(),
        psm=psm,
        point_clustering=point,
        point_weights=_point_weights(point, A, W),
        map_assignments=A[s_map].copy(),
        map_weights=[np.asarray(w, dtype=float) for w in W[s_map]],
        map_log_post=float(LP.ravel()[s_map]),
        rhat={"K": split_rhat(K), "log_post": split_rhat(LP)},
        ess={"K": ess_bulk(K), "log_post": ess_bulk(LP)},
        accept_rate=float(np.mean([np.mean(t.accept_rate) for t in traces])),
        n_chains=len(traces),
        n_kept=n_kept,
    )
