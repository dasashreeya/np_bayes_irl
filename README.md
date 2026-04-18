# NP-Bayes IRL with Parallelization

Nonparametric Bayesian Inverse Reinforcement Learning using a Dirichlet Process prior,
with data-parallel Gibbs sampling via Ray.

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
```

## Zartan cluster

```bash
# On Zartan head node:
ray start --head --num-cpus=16

# Then run:
python run_experiments.py --exp speedup
```
