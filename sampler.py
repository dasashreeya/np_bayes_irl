"""
sampler.py  |  Owner: SA  |  Interface: PROJECT_SCOPE.md §5
MCMC for the DP mixture of Boltzmann experts (DPM-BIRL).

Model:
    z ~ CRP(alpha),   w_k ~ G0 = N(0, I),   tau_i | z_i, w ~ Boltzmann(w_{z_i}) ^ tau

Sampler: Neal (2000) Algorithm 8 with m = 1 auxiliary component for the
assignments, then random-walk Metropolis-Hastings on each cluster's `w`.
The weights are instantiated, so this is *not* a collapsed sampler (P7).

The sampler only sees a LikelihoodBackend, never ground truth. It returns
post-burn-in samples as a Trace; summarizing them is summarize.py's job.
"""

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class SamplerConfig:
    """
    Everything that controls one chain. Every field is recorded in the result JSON.

    alpha:        DP concentration
    n_sweeps:     total sweeps, including burn-in
    burn_in:      sweeps discarded before samples are kept
    thin:         keep every `thin`-th post-burn-in sweep
    step_size:    initial std of the random-walk MH proposal on w
    n_mh_steps:   MH proposals per cluster per sweep
    adapt:        tune step_size during burn-in toward target_accept, then freeze it (WS2)
    target_accept: target MH acceptance rate for adaptation
    tau:          likelihood tempering exponent (Experiment A); 1.0 = untempered
    init:         "prior" (partition from CRP(alpha), weights from G0) or "one_cluster"
    """
    alpha: float = 1.0
    n_sweeps: int = 500
    burn_in: int = 100
    thin: int = 1
    step_size: float = 0.05
    n_mh_steps: int = 1
    adapt: bool = True
    target_accept: float = 0.23
    tau: float = 1.0
    init: str = "prior"

    @classmethod
    def from_dict(cls, d):
        """Build from a config dict, rejecting unknown keys so typos fail loudly."""
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown SamplerConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self):
        return asdict(self)


@dataclass
class Trace:
    """
    Post-burn-in samples from one chain.

    assignments: (n_kept, N) int, labels contiguous 0..K-1 within each sample
    weights:     list of n_kept lists; sample s holds K_s arrays of shape (F,)
    log_post:    (n_kept,) joint log p(z, w, D), with the likelihood raised to tau
    K:           (n_kept,) number of occupied clusters
    accept_rate: (n_kept,) MH acceptance rate in that sweep
    wall_time_s: total wall time of the chain, burn-in included
    step_size:   MH step size used after burn-in (frozen if adapt=True)
    config:      SamplerConfig.to_dict() of the run
    """
    assignments: np.ndarray
    weights: list
    log_post: np.ndarray
    K: np.ndarray
    accept_rate: np.ndarray
    wall_time_s: float
    step_size: float = float("nan")
    config: dict = field(default_factory=dict)

    def to_dict(self):
        """JSON-serializable dict (lists of plain floats and ints)."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError


def run_chain(backend, cfg, rng_key):
    """
    Run one chain.

    Args:
        backend: LikelihoodBackend for the dataset
        cfg:     SamplerConfig (or a dict accepted by SamplerConfig.from_dict)
        rng_key: int seed, np.random.Generator, or JAX PRNG key
    Returns:
        Trace of post-burn-in samples
    """
    raise NotImplementedError
