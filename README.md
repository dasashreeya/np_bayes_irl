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
  gibbs.py             # Collapsed Gibbs sampler (Both, Day 4)
  eval.py              # Evaluation metrics (P2, Day 5)
  parallel.py          # Ray parallel workers (P2, Day 6)
  run_serial.py        # Serial experiment (Both, Day 5)
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
