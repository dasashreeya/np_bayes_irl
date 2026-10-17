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

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.special import gammaln


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
        return {
            "assignments": np.asarray(self.assignments).tolist(),
            "weights": [[np.asarray(w, dtype=float).tolist() for w in ws] for ws in self.weights],
            "log_post": np.asarray(self.log_post, dtype=float).tolist(),
            "K": np.asarray(self.K).tolist(),
            "accept_rate": np.asarray(self.accept_rate, dtype=float).tolist(),
            "wall_time_s": float(self.wall_time_s),
            "step_size": float(self.step_size),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            assignments=np.asarray(d["assignments"], dtype=np.int64),
            weights=[[np.asarray(w, dtype=float) for w in ws] for ws in d["weights"]],
            log_post=np.asarray(d["log_post"], dtype=float),
            K=np.asarray(d["K"], dtype=np.int64),
            accept_rate=np.asarray(d["accept_rate"], dtype=float),
            wall_time_s=float(d["wall_time_s"]),
            step_size=float(d.get("step_size", float("nan"))),
            config=dict(d.get("config", {})),
        )


# ── Model terms ──────────────────────────────────────────────────────────────

def log_g0(w):
    """Normalized log N(w | 0, I). The constant matters: K varies across samples."""
    w = np.asarray(w, dtype=float)
    return -0.5 * float(w @ w) - 0.5 * w.size * np.log(2.0 * np.pi)


def log_crp(counts, alpha):
    """
    log P(partition) under CRP(alpha) for cluster sizes `counts`.

    Args:
        counts: (K,) positive ints, sizes of the occupied clusters
        alpha:  DP concentration
    Returns:
        float
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    return float(len(counts) * np.log(alpha) + gammaln(counts).sum()
                 + gammaln(alpha) - gammaln(alpha + N))


def _as_generator(rng_key):
    if isinstance(rng_key, np.random.Generator):
        return rng_key
    if isinstance(rng_key, (int, np.integer)):
        return np.random.default_rng(int(rng_key))
    import jax  # a JAX key: typed (jax.random.key) or legacy uint32 array
    try:
        data = jax.random.key_data(rng_key)
    except TypeError:
        data = rng_key
    return np.random.default_rng(np.asarray(data, dtype=np.uint32).ravel().tolist())


# ── Chain state ──────────────────────────────────────────────────────────────

class _State:
    """
    Mutable chain state. Labels stay contiguous: z in 0..K-1, every cluster occupied.

    z:      (N,) int labels
    W:      list of K (F,) weights
    n:      list of K cluster sizes
    L:      (N, K) untempered log p(tau_i | w_k), kept in sync with W
    """

    def __init__(self, z, W, L):
        self.z = np.asarray(z, dtype=np.int64)
        self.W = list(W)
        self.n = list(np.bincount(self.z, minlength=len(W)))
        self.L = L

    @property
    def K(self):
        return len(self.W)

    def remove_cluster(self, k):
        """Drop empty cluster k and shift higher labels down. Returns its (w, column)."""
        w, col = self.W.pop(k), self.L[:, k].copy()
        self.n.pop(k)
        self.L = np.delete(self.L, k, axis=1)
        self.z[self.z > k] -= 1
        return w, col

    def add_cluster(self, w, col):
        self.W.append(w)
        self.n.append(0)
        self.L = np.concatenate([self.L, col[:, None]], axis=1)
        return self.K - 1

    def log_post(self, alpha, tau):
        """Joint log p(z, w, D) with the likelihood raised to tau."""
        N = self.z.size
        return (log_crp(self.n, alpha)
                + sum(log_g0(w) for w in self.W)
                + tau * float(self.L[np.arange(N), self.z].sum()))


def _init_state(backend, cfg, rng):
    N, F = backend.N, backend.n_features
    if cfg.init == "one_cluster":
        z = np.zeros(N, dtype=np.int64)
    elif cfg.init == "prior":
        z = np.zeros(N, dtype=np.int64)
        counts = [1]
        for i in range(1, N):
            p = np.append(np.asarray(counts, dtype=float), cfg.alpha)
            k = rng.choice(len(p), p=p / p.sum())
            if k == len(counts):
                counts.append(0)
            counts[k] += 1
            z[i] = k
    else:
        raise ValueError(f"unknown init {cfg.init!r}")
    W = [rng.standard_normal(F) for _ in range(int(z.max()) + 1)]
    return _State(z, W, backend.loglik_matrix(W))


# ── Sweep ────────────────────────────────────────────────────────────────────

def _sample_assignments(st, backend, cfg, rng):
    """
    One pass of Neal (2000) Algorithm 8, m = 1, over all trajectories.

    For trajectory i: remove it from its cluster. If that leaves the cluster
    empty, the auxiliary component takes the old cluster's weight (Neal's rule
    for singletons); otherwise the auxiliary weight is a fresh draw from G0.
    Then choose among the occupied clusters (weight n_{-i,k}) and the
    auxiliary (weight alpha), each times p(tau_i | w)^tau.
    """
    F = backend.n_features
    for i in range(st.z.size):
        k_old = st.z[i]
        st.n[k_old] -= 1
        if st.n[k_old] == 0:
            aux_w, aux_col = st.remove_cluster(k_old)
            aux_ll = aux_col[i]
        else:
            aux_w, aux_col = rng.standard_normal(F), None
            aux_ll = backend.loglik_cluster(aux_w, [i])

        logits = np.append(np.log(np.asarray(st.n, dtype=float)) + cfg.tau * st.L[i],
                           np.log(cfg.alpha) + cfg.tau * aux_ll)
        p = np.exp(logits - logits.max())
        k = int(rng.choice(p.size, p=p / p.sum()))
        if k == st.K:  # the auxiliary component becomes a new cluster
            if aux_col is None:
                aux_col = backend.loglik_matrix([aux_w])[:, 0]
            k = st.add_cluster(aux_w, aux_col)
        st.z[i] = k
        st.n[k] += 1


def _update_weights(st, backend, cfg, step_size, rng):
    """
    Random-walk MH on each cluster's weight given the assignments.
    Target: G0(w) * prod_{i in k} p(tau_i | w)^tau.

    Returns:
        (n_accepted, n_proposed)
    """
    accepted = proposed = 0
    for k in range(st.K):
        idx = np.flatnonzero(st.z == k)
        cur = log_g0(st.W[k]) + cfg.tau * st.L[idx, k].sum()
        for _ in range(cfg.n_mh_steps):
            w_prop = st.W[k] + step_size * rng.standard_normal(st.W[k].shape)
            col = backend.loglik_matrix([w_prop])[:, 0]
            new = log_g0(w_prop) + cfg.tau * col[idx].sum()
            proposed += 1
            if np.log(rng.uniform()) < new - cur:
                st.W[k], st.L[:, k], cur = w_prop, col, new
                accepted += 1
    return accepted, proposed


def _adapt_step(step_size, accept_rate, t, target):
    """
    Robbins-Monro update of log(step_size) toward the target acceptance rate,
    with gain (t + 1)^-0.6. Called only during burn-in; the step size is then
    frozen, so the kept samples come from a fixed, valid MH kernel.
    """
    return float(step_size * np.exp((accept_rate - target) / (t + 1) ** 0.6))


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
    if isinstance(cfg, dict):
        cfg = SamplerConfig.from_dict(cfg)
    if not 0 <= cfg.burn_in < cfg.n_sweeps:
        raise ValueError("need 0 <= burn_in < n_sweeps")
    rng = _as_generator(rng_key)
    t0 = time.perf_counter()

    st = _init_state(backend, cfg, rng)
    step_size = cfg.step_size
    Z, Ws, LP, Ks, AR = [], [], [], [], []

    for sweep in range(cfg.n_sweeps):
        _sample_assignments(st, backend, cfg, rng)
        acc, prop = _update_weights(st, backend, cfg, step_size, rng)
        if cfg.adapt and sweep < cfg.burn_in:
            step_size = _adapt_step(step_size, acc / max(prop, 1), sweep, cfg.target_accept)
        if sweep >= cfg.burn_in and (sweep - cfg.burn_in) % cfg.thin == 0:
            Z.append(st.z.copy())
            Ws.append([w.copy() for w in st.W])
            LP.append(st.log_post(cfg.alpha, cfg.tau))
            Ks.append(st.K)
            AR.append(acc / max(prop, 1))

    return Trace(
        assignments=np.asarray(Z, dtype=np.int64),
        weights=Ws,
        log_post=np.asarray(LP),
        K=np.asarray(Ks, dtype=np.int64),
        accept_rate=np.asarray(AR),
        wall_time_s=time.perf_counter() - t0,
        step_size=float(step_size),
        config=cfg.to_dict(),
    )
