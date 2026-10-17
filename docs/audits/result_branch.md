# Audit: the `Result` branch (WS0, owner SA)

| | |
|---|---|
| **Scope item** | WS0 · "SA: audit the `Result` branch for P1. Don't cite its numbers either way." |
| **Answers** | Open question 4 in `PROJECT_SCOPE.md` §12 |
| **Audited commit** | `129459c` "Working batch with ARI 0.74 results - from Zartan" (2026-04-28), the only commit on `Result` past `main`'s ancestor `5bfcf18` |
| **Auditor** | SA, 2026-10-16 |

## Verdict

**Yes, `Result` uses ARI-based selection (P1).** The ARI 0.74 number is computed on a state chosen by its ARI against the true labels, and restarts reset the chain to that same ARI-chosen state. Following §12 Q4, **its numbers are dropped for good.**

A second finding goes beyond P1: **the figures committed under `results/figures/` are placeholders, not experiment output** (see below). They are on `main` as well as on `Result`.

## Finding 1: the reported state is chosen by ARI (P1)

`run_serial.py` at `129459c`:

```python
best_ari   = -1.0
best_state = copy.deepcopy(state)
...
if sweep % 10 == 0 or sweep == n_sweeps - 1:
    metrics = log_metrics(sweep, state['assignments'], state['weight_vectors'],
                          true_weights, true_assignments)        # ground truth
    ari = metrics['ari']
    if ari > best_ari:
        best_ari   = ari
        best_state = copy.deepcopy(state)                         # selected by ARI
...
assignments    = best_state['assignments']                        # reported output
weight_vectors = best_state['weight_vectors']
```

Three things are wrong here, and each one alone is enough to drop the number:

1. **Selection.** The output is the max-ARI state over about 50 checkpoints (every 10th sweep of 500). So "ARI 0.74" is a maximum over the chain, scored against the labels, not a posterior summary.
2. **Restarts feed ground truth back into the chain.** When K collapses to 1 after sweep 50, the chain is reset to `best_state` (the max-ARI state so far) plus N(0, 0.05²) noise. After the first restart, every later state depends on the true labels, so even the *last* state isn't a valid sample.
3. **The reported L2 has the same problem.** It's computed on `best_state` and also uses the one-sided `best_match_error` (P4).

On the `Result` line, `129459c` is the commit that *introduces* this logic; its parent `5bfcf18` returns the final state. `main` has the same logic from `d9b9a96` ("Initial commit: NP-Bayes IRL full pipeline", a separate root commit), which merge `6564303` combined with `5bfcf18`. So `run_serial.py` on `main` has P1 today; WS1 removes it.

## Finding 2: the committed figures are placeholders

All ten files in `results/figures/` were added in `5bfcf18` (2026-04-26). That's **two days before** the Zaratan run in `129459c`, and the blobs are byte-identical on `main` and `Result`. So they can't show Zaratan results. `README.md` says to make them with `python figures.py`, whose `__main__` block draws from hardcoded values:

| Figure | Data source in `figures.py __main__` | Evidence it's a placeholder |
|---|---|---|
| fig1 speedup | `{1: 37.26, 2: 21.13, 4: 21.46, 8: 31.65, 16: 137.84}` ("Paste your actual run_speedup results here") | Possibly real timings, but there's no JSON or git SHA to trace them, and they come from the P2 parallel runner, which isn't a posterior sampler. |
| fig2 weight error | `{1: 0.25, ..., 16: 0.34}  # replace with real values` | The comment says so. |
| fig3 convergence | `[{'sweep': i*10, 'n_clusters': max(2, 6 - i//3)} ...]` | **The committed PNG matches this formula exactly:** K = 6, 5, 4, 3 in steps of 30 sweeps, then flat at 2 from sweep 120. |
| fig4 baselines | `{'NP-Bayes (parallel)': {'ari': 0.91, 'l2': 0.23}, 'NP-Bayes (serial)': {'ari': 0.89, ...}, 'MaxEnt IRL': {'ari': 0.41, ...}}` | **The committed PNG shows exactly 0.91 / 0.89 / 0.41 and 0.23 / 0.25 / 0.71.** Also, `run_serial.py` scores MaxEnt as a single cluster, which gives ARI = 0 by construction, not 0.41. |
| fig5 weight recovery | `recovered_w = [np.array([0.9, -0.8, 0.1, ...]), ...]  # Replace with real weights` | Hand-typed vectors. |

`run_experiments.py` has a real pipeline that calls the same figure functions with computed values, but nothing shows it produced these files. Even if it had, its numbers would be subject to P2 (parallel) and P4 (L2).

## What this means for us

- **Don't cite anything** from `Result`, from W&B runs of `run_serial.py`, or from `results/figures/`. This includes the ARI 0.74 and the 4.79× speedup.
- **arXiv v1 (checked 2026-10-16): it does not appear to use these placeholder files.** v1 reports MaxEnt ARI = 0.000 (fig4 placeholder: 0.41), NP-Bayes ARI = 1.000 (placeholder: 0.89/0.91), and K converging after about 250 sweeps (fig3 placeholder: sweep 120). v1's numbers come from separate runs. The paper says ARI = 1.000 "at the end of a 500-sweep run", but `run_serial.py` with 500 sweeps (its default on both branches) returns the max-ARI state, not the last one. The repo can't show which script made the v1 run, so treat v1's numbers as affected by P1, and the v2 changelog retracts them.
- **Recommendation (needs SD's approval, since SD owns `results/` sync):** in SD's `legacy/` consolidation PR, move `results/figures/` to `legacy/results/figures_placeholder_2026-04/` with a one-line README pointing here. Then nothing under `results/` predates the `sweep.py` provenance rules (§5).
- **Branch fate.** `Result` holds no unique code worth keeping. Its only non-P1 changes are the Ray `max_retries=0`, the JAX-on-CPU env vars, a warm-up in `parallel.py` (all superseded by WS4), and an import-name fix in `experiments/exp_parallel.py`. Proposal: tag it `archive/result-2026-04` and delete the branch.

## How to reproduce this audit

```bash
git fetch upstream
git diff 5bfcf18 129459c -- run_serial.py            # adds best_ari / best_state / restarts
git show 5bfcf18 --stat -- results/figures            # figures added before the Zaratan run
git ls-tree -r upstream/main results/figures          # same blob ids as on Result
git show 129459c:figures.py | sed -n '/__main__/,$p'  # the hardcoded values
```
