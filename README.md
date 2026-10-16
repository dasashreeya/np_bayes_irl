# NP-Bayes IRL with Parallelization

Nonparametric Bayesian Inverse Reinforcement Learning using a Dirichlet Process prior,
with data-parallel Gibbs sampling via Ray.

## Project status (Oct 2026): read this first

This repo is being reworked for a **TMLR submission** (target **Dec 18, 2026**). The full plan is in
**[`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md)**: the audit, the work split between the two authors, the interfaces, the experiment matrix, the compute plan, the timeline and the decision log.

> **Warning: don't cite the v1 numbers.** The current experiment scripts select the reported sampler state by ARI against the
> ground-truth labels (`best_ari` / `best_state`), and parallel mode (`parallel.py`) does not sample the true posterior.
> All results in `results/` and in arXiv v1 are being re-validated. See §2 of the scope doc.

| Area | Owner |
|---|---|
| Inference, estimator, diagnostics, metrics, baselines; Experiments E0, A, C; writing lead | Sai Anirudh Katupilla (SA) |
| Fast exact sampler, sweep runner, CI, compute (Zaratan, CPU and GPU); Experiments B, D, E | Shreeya Dasa Lakshminath (SD) |

## Open decisions for Shreeya (from the `sa/*` branches, Oct 16)

These came out of the WS0 skeleton and the WS1/WS2 work. Each one touches §5 of the scope doc, so it needs SD's sign-off; none blocks review of the code.

1. **Interface additions proposed in the skeleton** (commit "WS0: skeleton for the shared interfaces in §5"). §5 lists `run_chain(backend, cfg, rng_key) -> Trace` and the `Trace` fields; the skeleton adds:
   - `SamplerConfig`, a dataclass holding everything that controls one chain (`alpha`, `n_sweeps`, `burn_in`, `thin`, `step_size`, `n_mh_steps`, `adapt`, `target_accept`, `tau`, `init`), with `from_dict` rejecting unknown keys and `to_dict` for the result JSON;
   - `Trace.step_size` (the MH step size in force after burn-in) and `Trace.config` (the `SamplerConfig` as a dict), next to the §5 fields;
   - `Trace.to_dict()` / `Trace.from_dict()`, so `sweep.py` can write and read one JSON per chain;
   - `run_chain` accepting an `int` seed or a `numpy.random.Generator` as well as a JAX PRNG key.
   If SD agrees, §5 gets updated to match; if not, say which to drop.

2. **`LikelihoodBackend.invalidate(weight_id)` exists but is never called.** §5 says `log_pi` is "cached by weight id" without saying what a weight id is: the Python `id()` of the array, a hash of its bytes, or an integer the sampler assigns per cluster. The sampler stores each cluster's weight as a numpy array and swaps in a new array on every accepted MH move, so it can pass whichever id SD defines. SD decides the id and the cache contract (who creates entries, when they expire); SA then wires the call into `sampler._update_weights`.

3. **The O(N·K) → O(K) cost claim (§2 and WS4) needs revising.** The new sampler already keeps an `(N, K)` log-likelihood matrix in sync with the cluster weights, so the occupied clusters cost `K · n_mh_steps` soft-VI solves per sweep, not `N · K`. But Algorithm 8 draws a fresh auxiliary weight from G0 for every trajectory whose cluster isn't a singleton, and each one needs its own solve, so the per-sweep count is `N + K · n_mh_steps` solves. A `log_pi` cache can't remove the `N` term. Two ways to attack it, both WS4: (a) draw all N auxiliary weights at the start of the sweep (they're independent G0 draws that don't depend on the state, so this is still exact) and solve them in one `vmap` batch; (b) accept the `N` term and report the speedup honestly as "`N·K` → `N + K`". The scope text should say which.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce all results

```bash
python run_experiments.py --exp all
```

## Run individual experiments

```bash
# Serial NP-Bayes IRL
python run_experiments.py --exp serial

# Speedup experiment (requires Zartan with Ray)
python run_experiments.py --exp speedup

# Baselines only
python run_experiments.py --exp baselines
```

## Run tests

```bash
pytest tests/
```

## Generate figures

```bash
python figures.py
# Saves to results/figures/
```

## Project structure

```
np_bayes_irl/
  objectworld.py       # Environment (P1, Day 1)
  expert_demos.py      # Expert trajectory generation (P2, Day 1)
  mdp_utils.py         # Soft value iteration, Boltzmann policy (P1, Day 2)
  maxent_irl.py        # MaxEnt IRL baseline (P1, Day 2)
  reward_features.py   # Linear reward utilities (P2, Day 2)
  dp_prior.py          # DP base measure G0 (P1, Day 3)
  likelihood.py        # Boltzmann likelihood (P2, Day 3)
  gibbs.py             # v1 Gibbs sweep (Neal Alg. 8 + MH, not collapsed); legacy, used by parallel.py
  sampler.py           # DPM-BIRL MCMC (Neal Alg. 8, m=1, + MH), returns a Trace (SA)
  likelihood_backend.py # Per-trajectory log-likelihoods behind one interface (SD)
  summarize.py         # Posterior over K, PSM + Binder point estimate, MAP, R-hat/ESS (SA)
  diagnostics.py       # Rank-normalized split R-hat and bulk ESS (SA)
  evaluate.py          # The only module that sees ground truth (SA)
  eval.py              # v1 metrics; best_match_error is one-sided (P4), use evaluate.py
  parallel.py          # Ray parallel workers (P2, Day 6)
  run_serial.py        # Serial experiment: chains -> summarize -> evaluate (SA)
  run_speedup.py       # Speedup experiment (Both, Day 8)
  run_experiments.py   # Master script (Both, Day 10)
  figures.py           # All publication figures (Both, Day 11)
  experiments/         # Individual experiment scripts
  configs/             # YAML configs
  tests/               # Unit tests
  results/             # Figures and run outputs
  docs/                # Project scope and plan (PROJECT_SCOPE.md)
```

## Zaratan cluster (UMD HPC)

The cluster is UMD's **Zaratan**. The config file is still named `configs/zartan.yaml`. Sweeps run as CPU array jobs, and the GPU is used only for the Experiment D benchmark. See §7 of [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md) for who runs what, job conventions and the SU budget. The commands below are the legacy v1 workflow.

```bash
# On Zartan head node:
ray start --head --num-cpus=16

# Then run:
python run_experiments.py --exp speedup
```
