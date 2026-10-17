"""
diagnostics.py  |  Owner: SA
MCMC convergence diagnostics on scalar chains (Vehtari et al. 2021).

Used on label-invariant scalars only: K and the joint log-posterior.
"""

import numpy as np


def split_rhat(chains):
    """
    Rank-normalized split R-hat.

    Args:
        chains: (n_chains, n_draws) array of one scalar
    Returns:
        float; values below 1.05 suggest the chains agree
    """
    raise NotImplementedError


def ess_bulk(chains):
    """
    Bulk effective sample size (rank-normalized, split chains, Geyer's
    initial monotone sequence estimator).

    Args:
        chains: (n_chains, n_draws) array of one scalar
    Returns:
        float, ESS pooled over all chains
    """
    raise NotImplementedError
