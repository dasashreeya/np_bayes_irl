# NP-Bayes IRL: Project Scope for TMLR Submission

| | |
|---|---|
| **Status** | Proposed. Both authors must sign off on §4 (the work split) and §12 (open questions) before Phase 1 starts. |
| **Authors** | Sai Anirudh Katupilla (**SA**, first author) · Shreeya Dasa Lakshminath (**SD**) |
| **Target venue** | [TMLR](https://jmlr.org/tmlr/) (rolling submission, no deadline) |
| **Submit by** | Fri **Dec 18, 2026** · hard fallback Fri **Jan 15, 2027** |
| **Preprint** | [arXiv:2607.09886](https://arxiv.org/abs/2607.09886) (v1). A v2 is part of this scope. |
| **Doc owner** | SD maintains this file; either author may propose changes by PR (§0) |
| **Last updated** | Oct 9, 2026 |

---

## Contents

0. [How to use this document](#0-how-to-use-this-document)
1. [What we're trying to publish](#1-what-were-trying-to-publish)
2. [Current state (audit)](#2-current-state-audit-oct-2026)
3. [Out of scope](#3-out-of-scope-and-why)
4. [Workstreams and owners](#4-workstreams-and-owners)
5. [Shared interfaces and conventions](#5-shared-interfaces-and-conventions)
6. [Experiment matrix](#6-experiment-matrix)
7. [Compute plan (Zaratan, CPU vs GPU)](#7-compute-plan-zaratan-cpu-vs-gpu)
8. [Timeline and decision gates](#8-timeline-and-decision-gates)
9. [Risks and mitigations](#9-risks-and-mitigations)
10. [Pre-submission checklist](#10-pre-submission-checklist-mock-review-dec-11)
11. [Decision log](#11-decision-log)
12. [Open questions](#12-open-questions)
13. [Glossary](#13-glossary)
14. [References](#14-references)

---

## 0. How to use this document

- **This is the single source of truth** for scope, ownership and dates. If it isn't in this file, it hasn't been agreed.
- **Checkboxes are the task list.** When you finish an item, tick it in the same PR that does the work. Add the PR number next to it, for example `- [x] Cache log_pi (#14)`.
- **Changing scope.** Open a PR that edits this file and add a row to the [decision log](#11-decision-log). The other author approves it. Don't change scope in chat or commit messages alone.
- **Owners.** Every task has exactly one owner, shown as **SA** or **SD**. The owner decides how the work gets done. The other author reviews the PR.
- **Statuses.** Use `[ ]` for not started, `[~]` for in progress (add a short note), and `[x]` for done. Use ~~strikethrough~~ for items we cut, with the reason in the decision log.
- **Weekly sync.** Spend 15 minutes each week walking §8 together. If a gate is at risk, decide then — don't wait until the gate date.

---

## 1. What we're trying to publish

### Why TMLR
TMLR's acceptance bar is *"are the claims supported by accurate, convincing evidence"*, not novelty or projected impact. The model we use (a Dirichlet-process prior over reward functions, sampled with Gibbs) was introduced by Choi & Kim (2012). A venue that judges novelty first is therefore a bad fit, and one that judges rigor is a good fit. TMLR also has no deadline and no travel cost.

### The reframed contribution
v1 claims *"our method recovers expert clusters (ARI = 1.000)."* That claim does not survive an audit of the code (§2). The paper we can defend makes two contributions:

1. **An exact, substantially faster sampler for DPM-BIRL.**
   - The speedup comes from removing redundant value iteration: per-sweep cost drops from O(N·K) to O(K) soft-VI solves.
   - The sampler is then parallelized in a way that provably leaves the target posterior unchanged.
2. **An account of when DP-based reward clustering works, and why it fails when it does.** This covers four questions:
   - how the prior scales against the likelihood;
   - how much of the failure is model misspecification;
   - whether the model can tell different rewards apart from different levels of rationality;
   - in which regime a DP beats a finite mixture with K chosen by BIC.

In one sentence:
> *We make DPM-BIRL fast enough to study properly, and then study it properly.*

---

## 2. Current state (audit, Oct 2026)

### What exists and works
- Tabular ObjectWorld environment, Boltzmann expert model and soft value iteration (`objectworld.py`, `mdp_utils.py`, `likelihood.py`), with unit tests in `tests/`.
- A DP prior, plus a Gibbs sampler with Metropolis–Hastings (MH) weight updates (`dp_prior.py`, `gibbs.py`).
- A Ray-based distributed runner (`parallel.py`), a MaxEnt IRL baseline (`maxent_irl.py`) and figure scripts.

### Blocking problems
These must be fixed before *any* number in the paper can be trusted.

| # | Problem | Where | Why it matters |
|---|---|---|---|
| P1 | **Ground-truth leakage.** The reported state is the one with the highest ARI against the true labels, and restarts reset the chain to that ARI-best state. | `run_serial.py`, `run_ablation.py`, `run_data_efficiency.py`, `run_k3_seed_validation.py` (the `best_ari` / `best_state` logic) | Every reported ARI and L2, including v1's ARI = 1.000, is selected using ground truth, which isn't available at inference time. |
| P2 | **Parallel mode doesn't sample the posterior.** Each worker computes CRP counts from its own chunk only, so the prior is wrong. `merge_states` matches clusters across workers with `np.allclose(atol=1e-6)`, so "the same" cluster gets split once workers' MH moves diverge. | `parallel.py` (an in-code comment acknowledges this) | The 4.79× speedup was measured on a procedure whose output is not a posterior sample. |
| P3 | **The posterior is thrown away.** `run_gibbs` keeps only `n_clusters` per sweep. | `gibbs.py` | No posterior summaries are possible, which is why the runners fell back on peeking at ARI. |
| P4 | **The L2 metric is one-sided.** `best_match_error` matches each true `w` to its *nearest* recovered `w`, so extra spurious clusters cost nothing. | `eval.py` | L2 sits at 1.05–1.33 while K is badly over-recovered. The metric hides the failure. |
| P5 | **Weight updates barely mix.** A random-walk MH step (`step_size=0.01`) on a 16-dimensional `w`, one proposal per cluster per sweep, and the acceptance rate is never logged. | `gibbs.py::update_weight_vector` | The weights likely never converge. The collapse-to-K=1 restart hack is a symptom of this. |
| P6 | **The data generator and the inference model don't match.** Demos are generated with *hard-max* VI at `beta=1.0`, while inference uses *soft* VI at `beta=5.0`. | `expert_demos.py` vs `likelihood.py` | Some unknown share of the recovery failure is model misspecification, not sampler failure. |
| P7 | **Terminology.** v1 calls the sampler "collapsed Gibbs", but the weights are instantiated and MH-updated. That makes it Neal (2000) Algorithm 8 with m = 1. | paper | A Bayesian reviewer will catch this in the method section. |
| P8 | **Seed selection.** `run_k3_seed_validation.py` scans seeds 1–20 and keeps the one grid layout that works. | results | It has to be replaced by a filter that is stated *before* any results are seen (§6). |

### What the current results actually show
These come from local result files that haven't been committed yet.
- K = 3 runs recover 4 or 9 clusters (`k3_results.json`, `k3_seed_validation.json`).
- K = 2 ablations recover 3 or 4 clusters.
- ARI is non-monotonic in data: 1.0 → 0.46 → 0.94 → 0.26 as demos per type go 5 → 10 → 15 → 20 (`data_efficiency_results.json`).
- Changing α across 1, 3 and 5 gives **identical results to six decimal places** (`ablation_results.json`). This is the first piece of evidence for Experiment A.

### Repo hygiene
- **Branches to resolve.**
  - `Result` ("ARI 0.74 from Zartan") very likely shares problem P1.
  - `mujoco-neural-scale` is out of scope (§3).
  - The local ref `main-split-1` no longer exists on the remote.
- **Uncommitted work.** `run_ablation.py`, `run_data_efficiency.py`, `run_k3.py`, `run_k3_seed_validation.py`, `Dockerfile` and a modified `requirements.txt` exist only on SD's machine.
- **Stub.** `experiments/exp_baselines.py` is still a `pass`.
- **Naming.** The cluster is UMD's **Zaratan**. The README and `configs/zartan.yaml` spell it "Zartan". Keep the file name, but use the correct spelling in the paper.

---

## 3. Out of scope (and why)

- **The `mujoco-neural-scale` branch (HalfCheetah, MLP reward, Monte Carlo value).** Shelve it for this paper; don't merge it. There are three reasons:
  - The 17→64→64→1 MLP reward has about 5.4k parameters, and random-walk MH won't mix in 5.4k dimensions.
  - The likelihood runs Monte Carlo rollouts on every call, and the sampler makes N·K such calls per sweep.
  - The action likelihood ("advantage vs random baselines") isn't a normalized density.

  It inherits P1–P5 and adds new problems on top. It's a different paper (neural nonparametric IRL): keep the branch and revisit it after TMLR. It's also the only GPU-heavy workload in the repo, and §7 explains why we don't need GPUs without it.
- **Any conference deadline before submission.** AAMAS, AISTATS and ICLR 2027 have all closed. IJCAI and ICML (January) would force a rushed submission.
- **Novelty claims about the model itself.** We cite Choi & Kim (2012), Babes-Vroman et al. (2011) and Michini & How (2012) as the model's origin.

---

## 4. Workstreams and owners

The split follows a natural seam:
- **SA** owns *inference and science*: what the sampler computes, and what the experiments mean.
- **SD** owns *systems and scale*: how fast it runs, where and how runs are executed, and the large-N regime.

Each workstream has one owner, and the other author reviews its PRs. **SD is also the compute owner** (§7), but each experiment's owner launches their own runs.

### At a glance

| Workstream | Owner | Phase | Main output |
|---|---|---|---|
| WS0 Consolidation and infrastructure | SD leads, both contribute | 0 | One clean `main`, a sweep runner, a CI leakage test, Zaratan job templates |
| WS1 Valid estimator and diagnostics | SA | 1 | `Trace`, `summarize()`, R-hat/ESS, no ground-truth leakage |
| WS2 Sampler mixing | SA | 1 | Adaptive MH (HMC as a stretch), acceptance logging |
| WS3 Metrics and baselines | SA | 2 | Hungarian-matched metrics, four baselines |
| WS4 Fast exact sampler | SD | 1–2 | Cached and vectorized likelihood, exact parallelism, exactness test |
| WS5 Experiments | A and C: SA · B, D and E: SD | 2–3 | All results in `results/` |
| WS6 Writing | SA leads; SD writes the systems sections | 4 | Manuscript, arXiv v2, TMLR submission |

### WS0: Consolidation and infrastructure (Phase 0)
- [ ] **SD:** commit the local-only scripts and results into `legacy/`, so the audit trail is kept.
- [ ] **SA:** audit the `Result` branch for P1. Don't cite its numbers either way.
- [ ] **Both:** agree on the interfaces and the file-ownership table in §5. Merge a skeleton PR with stub functions before splitting up.
- [ ] **SD:** build a config-driven sweep runner (`sweep.py`). It takes a YAML grid as input and writes one JSON per (config, seed, chain) under `results/<exp>/<config_hash>/`. Each record carries the git SHA, the full config, timings and the hostname/node type.
- [ ] **SD:** add `tests/test_no_ground_truth_leak.py`. It's a CI check that fails if any sampler module imports `evaluate`, or receives true labels.
- [ ] **SD:** build the Zaratan environment and job templates (§7): a pinned `requirements.txt` or container, and SLURM array-job scripts in `jobs/`.

**Done when:**
- `main` holds one coherent codebase;
- CI runs the existing tests plus the leakage test;
- one command runs a 2-config × 2-seed toy sweep end to end, locally and on Zaratan.

### WS1: Valid estimator and diagnostics (SA)
- [x] Store post-burn-in samples: assignments `(n_kept, N)`, weights, joint log-posterior, K and the MH acceptance rate. (`sampler.Trace`)
- [x] Implement `summarize()`, with three outputs:
  - the posterior distribution over K;
  - a point clustering from the posterior similarity matrix, chosen to minimize Binder loss;
  - the MAP state by joint log-posterior, reported as a secondary summary.
- [~] Delete all ARI-based selection and all ARI-triggered restarts. This fixes P1. *(`run_serial.py` on `main` rewritten; `run_ablation.py`, `run_data_efficiency.py`, `run_k3_seed_validation.py` are only on SD's machine and go to `legacy/` as they are.)*
- [~] Implement R-hat and ESS on the label-invariant scalars (K and the joint log-posterior), plus trace plots across 4 chains. *(R-hat/ESS done, matching arviz to 1e-15; trace plots wait for the `sweep.py` result format.)*
- [~] Fix the "collapsed" terminology everywhere (P7), or actually collapse the sampler if a conjugate approximation is tractable. The default is to fix the terminology. *(code and README done; the paper text is done in WS6.)*

**Done when:** a K=2 run and a K=3 run each produce a `PosteriorSummary` with R-hat < 1.05 on K and on log-posterior across 4 chains, and ground truth appears only in `evaluate()`.

### WS2: Sampler mixing (SA)
- [ ] Adapt the MH step size during burn-in toward a target acceptance rate of about 0.23, then freeze it so the post-burn-in chain stays valid.
- [ ] Make the number of MH steps per cluster per sweep configurable. As a stretch, add HMC on `w` given the assignments (JAX makes the gradient cheap).
- [ ] Remove the collapse-to-K=1 restart hack. If it's still needed after adaptation, report it as a known limitation.
- [ ] Log acceptance rates in every run (P5).

**Done when:** post-burn-in acceptance is between 0.15 and 0.40, and the K and log-posterior traces mix across chains without any restarts.

### WS3: Metrics and baselines (SA)
- [x] Replace `best_match_error` with Hungarian matching between true and recovered clusters. Report unmatched clusters as a separate over-clustering count, and weight the errors by cluster size. This fixes P4.
- [x] Always report the full posterior over K next to ARI and L2.
- [ ] Implement the baselines, replacing the `exp_baselines.py` stub:
  - **sequential sampler** (same model; the reference for every speedup claim);
  - **finite mixture with K selected by BIC** over K ∈ {1…8};
  - **EM multiple-intentions IRL** (Babes-Vroman et al. 2011);
  - **single-reward MaxEnt IRL** (already exists; used as a floor only).

**Done when:** all four baselines run through the sweep runner on the same data, seeds and metrics as our method.

### WS4: Fast exact sampler (SD)
- [ ] **Cache `log_pi` per weight vector.** Invalidate an entry on an accepted MH move, and add one on new-cluster creation. Because `log_pi` depends only on `w` and not on the trajectory, this cuts per-sweep soft-VI solves from O(N·K) to O(K).
- [ ] **Vectorize the likelihood.** Store each trajectory as `(s_idx, a_idx)` arrays and compute it with one gather plus a sum, replacing the per-timestep Python loop in `likelihood.py`.
- [ ] **Batch the soft-VI solves across clusters** with `jax.vmap`. The same code then runs on CPU or GPU (§7).
- [ ] **Exact parallelism:**
  - keep the assignment sweep sequential, with *global* CRP counts;
  - parallelize only the work that is conditionally independent: the per-cluster `log_pi` solves, the N × K per-trajectory likelihood matrix, and the per-cluster MH updates given assignments;
  - write the two-paragraph argument that the stationary distribution is unchanged.
- [ ] Replace `merge_states` entirely. This fixes P2. Keep the old parallel runner in `legacy/` for the record only.
- [ ] **Exactness test.** For the same seeds, the parallel and sequential samplers produce posteriors over K that are statistically indistinguishable (KS test across chains).

**Done when:** the exactness test passes. Report the speedup in per-sweep wall time and in ESS per second, separately for caching alone, vectorization alone and parallelism.

### WS5: Experiments
Each experiment is owned end to end by one author: design, launch, analysis, figure and results paragraph. The full matrix is in §6.
- **SA:** E0 (replication), A (prior swamping), C (DP vs finite mixture).
- **SD:** B (misspecification and identifiability), D (scaling), E (real data, stretch).

### WS6: Writing (SA leads)
- [ ] **SA:**
  - abstract, intro and related work (Choi & Kim, Babes-Vroman, Michini & How);
  - the method section (model and sampler, with correct terminology);
  - results for E0, A and C, and the discussion.
- [ ] **SD:**
  - the fast-sampler section (caching, vectorization, the exactness argument);
  - results for B and D;
  - the reproducibility appendix (configs, compute, seeds, code link).
- [ ] **Both:** the limitations section, the arXiv v2 changelog, and an internal mock review (§10) one week before submission.

---

## 5. Shared interfaces and conventions

Agree on these in Phase 0, then don't break them without a PR that both authors approve.

```python
# likelihood_backend.py  — owner: SD
class LikelihoodBackend:
    """Owns every expensive computation. The sampler never calls soft VI directly."""
    def __init__(self, phi, T, gamma, beta, trajectories, device="cpu"): ...
    def log_pi(self, w) -> jnp.ndarray               # (S, A); cached by weight id
    def loglik_matrix(self, weights) -> jnp.ndarray  # (N, K) per-trajectory log-likelihood
    def loglik_cluster(self, w, idx) -> float        # sum over trajectories in idx
    def invalidate(self, weight_id) -> None          # call after an accepted MH move

# sampler.py  — owner: SA (sweep logic); calls LikelihoodBackend only
def run_chain(backend, cfg, rng_key) -> Trace

class Trace:                  # post-burn-in samples only
    assignments: np.ndarray   # (n_kept, N) int
    weights:     list         # per kept sample: list of K arrays
    log_post:    np.ndarray   # (n_kept,)
    K:           np.ndarray   # (n_kept,)
    accept_rate: np.ndarray   # (n_kept,)
    wall_time_s: float

# summarize.py — owner: SA
def summarize(traces: list[Trace]) -> PosteriorSummary
    # posterior over K, PSM point estimate (Binder), MAP, R-hat, ESS

# evaluate.py — owner: SA. THE ONLY MODULE THAT SEES GROUND TRUTH.
def evaluate(summary, true_assignments, true_weights) -> dict
```

**Why the seam sits here.** SA can change what the sampler does without touching performance code, and SD can change how fast it runs without touching the statistics. The `LikelihoodBackend` boundary also lets SD develop WS4 while SA develops WS1, in parallel, with no merge conflicts.

### File ownership
The non-owner opens PRs against the owner's files and never pushes to them directly.

| Owner: SA | Owner: SD |
|---|---|
| `sampler.py`, `gibbs.py` (sweep logic), `summarize.py`, `evaluate.py`, `diagnostics.py`, `baselines/` | `likelihood_backend.py`, `likelihood.py`, `parallel.py`, `sweep.py`, `configs/`, `jobs/`, `expert_demos.py`, `tests/test_no_ground_truth_leak.py`, `Dockerfile`, `docs/` |

Shared, where either author may edit with the other reviewing: `figures.py` (each author owns their own figure functions) and `README.md`.

### Conventions
- **Branches.** One feature branch per checklist item, named `sa/<topic>` or `sd/<topic>`. Squash-merge into `main` after the other author reviews. Never push directly to `main`, apart from typo-level fixes to docs.
- **Commits.** Use an imperative subject line (≤ 72 characters), then a body explaining *why*. Reference the scope item, for example `WS4: cache log_pi per weight vector`.
- **Seeds.** Every seed is fixed in the config. A seed is never chosen by looking at results.
- **Results.** Results are append-only. Never overwrite a results JSON; re-runs get a new config hash or a new timestamped directory.
- **Figures.** Figures are generated only from `results/`, by `figures.py`. Never edit one by hand, and every figure function names the result directory it reads.
- **Docstrings.** Every public function documents its arguments, shapes and returns, in the style `gibbs.py` already uses. New modules start with a header saying what the module is for and who owns it.

---

## 6. Experiment matrix

Unless noted otherwise, every configuration runs **4 chains × 5 seeds**, and results are reported as mean ± std over seeds.

**Environment-validity filter (fixes P8).** A grid is valid only if every feature referenced by an expert type has at least one object present. This rule is stated in the paper and applied before any inference runs.

| Exp | Question | Grid | Owner | Approx. chains | Hardware |
|---|---|---|---|---|---|
| **E0** Replication | What do the v1 setups give under a valid estimator? | K ∈ {2,3} × demos/type ∈ {10,20} | SA | 80 | CPU |
| **A** Prior swamping | Is over-clustering driven by the likelihood-to-prior ratio, rather than by α? | traj length L ∈ {5,10,20,40} × demos/type ∈ {5,10,20} × α ∈ {1,10} × tempering τ ∈ {1, 1/L}, at K=3 | SA | 960 | CPU |
| **B** Misspecification and identifiability | How much of the failure is generator/inference mismatch? Does the DP split experts who share a reward but differ in rationality? | (i) β_gen ∈ {0.5,1,2,5} × β_inf ∈ {1,5} × VI ∈ {hard, soft}, at K=2. (ii) Same `w` with β ∈ {1,5}, plus a three-group variant. | SD | ~420 | CPU |
| **C** When does the DP win? | Where is the boundary between the DP and a finite mixture with BIC? | reward separation (cosine) ∈ {0, 0.5, 0.8, 0.95} × demos/type ∈ {5,10,20,40} × K ∈ {2,3,5}, for the DP and all WS3 baselines | SA | 960 + baselines | CPU |
| **D** Scaling | Does exact parallelism matter at realistic N? | N ∈ {40, 200, 1000, 5000} × workers ∈ {1,2,4,8,16}; strong and weak scaling; 3 seeds. Plus one CPU-vs-GPU backend comparison at N ∈ {1000, 5000}. | SD | ~60 long runs | CPU, plus one A100 benchmark |
| **E** *(stretch)* Real data | Does the method recover known heterogeneity in real demonstrations? | Discretized GPS/taxi trajectories gridded into a tabular MDP, with driver ID as the ground-truth cluster. Cut this first if time is short. | SD (SA on reward-feature design) | TBD | CPU |

**Ordering rule.**
- E0 runs first.
- A and B then start in parallel.
- C starts only after A: if prior swamping holds, C must use the tempered likelihood.

**Figure plan.** Each figure has the same owner as its experiment.

| Fig | Content | Source |
|---|---|---|
| 1 | Sampler cost breakdown and speedup | D / WS4 |
| 2 | Exactness check: parallel vs sequential posterior over K | WS4 |
| 3 | K inflation vs likelihood-to-prior ratio | A |
| 4 | Misspecification heatmap and the identifiability result | B |
| 5 | Phase diagram, DP vs BIC | C |
| App. | Convergence diagnostics (R-hat, ESS, traces) | WS1 |

---

## 7. Compute plan (Zaratan, CPU vs GPU)

### Who runs what
- **SD is the compute owner.** That means:
  - the Zaratan environment;
  - the job templates in `jobs/`;
  - the service-unit (SU) budget;
  - the large-scale and GPU runs for Experiment D;
  - keeping `results/` in sync.
- **Each experiment's owner launches their own sweeps** using SD's templates, so nobody waits on the other person to press "run":
  - SA submits E0, A and C;
  - SD submits B, D and E.

**SA and SD are the only people running jobs.** If a job fails because of the environment or templates, SD fixes it. If it fails because of the sampler or the statistics, SA fixes it.

### Why almost everything runs on CPU, not GPU
- **Each chain is small, sequential work.** The tabular soft VI on a small ObjectWorld grid is a handful of small matrix operations. GPU kernel-launch and transfer overhead dominate at this size, so a GPU would mostly sit idle.
- **Throughput comes from running many independent chains at once.** That's an embarrassingly parallel CPU workload. A standard Zaratan CPU node has 128 cores, so it runs 128 chains at the same time.
- **The numbers.** E0–C total roughly 2.5–3k chains. At current speed (about 1–3 minutes per chain at N ≈ 30), that's on the order of 100 core-hours, or about one CPU node for an hour or two. It will be less after WS4. *This is an estimate: re-measure it after WS4 lands.*

### Where a GPU actually helps
- **Experiment D at large N (1000–5000),** once WS4 has made the likelihood a batched N × K gather and the soft-VI solves are `vmap`-ed across clusters. One controlled CPU-vs-GPU comparison of the `LikelihoodBackend` at those sizes is itself a useful data point for the systems section.
- **The WS2 HMC stretch,** only if it ends up gradient-heavy at large N. Try CPU first.
- **The shelved MuJoCo branch** would need GPUs, which is one more reason it's out of scope.

### Zaratan hardware and GPU cost
From the [Zaratan cluster page](https://hpcc.umd.edu/home/clusters/zaratan/):

| Resource | What it is | Use it for |
|---|---|---|
| Standard CPU node | 128 cores, 512 GiB RAM (AMD Zen3) | **All sweeps** (E0, A, B, C, E) and the Ray scaling curves in D |
| Fractional A100 (`gpu:a100_1g.5gb`) | 1/7 of an A100, 7 SU/hr | Developing and debugging the GPU backend |
| Full A100 (`gpu:a100`) | 48 SU/hr | The final CPU-vs-GPU benchmark in D only |
| H100 (`gpu:h100`) | 144 SU/hr | **Not needed** — avoid it |

**Proposed GPU budget: at most 20 A100-hours (960 SU)** for the whole project. Any spend beyond that needs a decision-log entry. Check the CPU core-hour rate on your own allocation before Phase 3.

### Job conventions
- **One SLURM array task = one (config, seed, chain).** Each task writes its own JSON, so a crashed task can be resubmitted alone.
- **Resubmission is idempotent:** a task whose output JSON already exists exits immediately.
- **Scaling runs in D get a whole node** (`--exclusive`), so co-tenants don't contaminate the timings. Every result records the node type.
- **Paths:** job scripts live in `jobs/`, logs go to `results/<exp>/logs/` (git-ignored), and the result JSONs are committed.
- **Ray runs** start a head node inside the job allocation. Never use a long-lived Ray cluster for timing runs.

---

## 8. Timeline and decision gates

| Phase | Dates (2026) | SA | SD | Gate at end |
|---|---|---|---|---|
| **0** Consolidate | Oct 12–16 | Audit the `Result` branch; draft the interfaces | `legacy/` commit, sweep runner, CI leakage test, Zaratan environment and templates | Skeleton PR with the interfaces merged |
| **1** Valid sampler | Oct 19–30 | WS1 estimator and diagnostics; WS2 mixing | WS4 caching and vectorization, behind the same interface | **G1, Oct 30:** E0 results in hand |
| **2** First science | Nov 2–13 | Experiment A; WS3 metrics and baselines | WS4 exact parallelism and exactness test; Experiment B | **G2, Nov 13:** does prior swamping hold? (decides C's likelihood) |
| **3** Main results | Nov 16–Dec 4 | Experiment C (DP vs baselines) | Experiment D on Zaratan, including the GPU benchmark; E if on track | **G3, Dec 4:** results frozen; no new experiments after this |
| **4** Write and submit | Dec 7–18 | Lead writing: intro, method, E0, A, C | Systems section, B, D, reproducibility appendix | Mock review Dec 11 · arXiv v2 and **TMLR submission Dec 18** |

### What each gate decides
- **G1 (Oct 30).** If the valid estimator shows the method fails even at K=2, the paper's framing pivots. A and B become the main result: *"why DPM-BIRL over-clusters, plus a faster exact sampler to study it."* That paper is still publishable at TMLR. The ARI = 1.000 framing is not.
- **G2 (Nov 13).** If prior swamping holds, C runs with the tempered likelihood as the default. If it doesn't, A is reported as a negative ablation and C runs with the untempered likelihood.
- **G3 (Dec 4).** Results are frozen. Anything unfinished either becomes a limitation in the paper or gets cut, with a decision-log entry.

### Calendar risks
- Thanksgiving falls on Nov 26.
- UMD finals run through mid-December, overlapping Phase 4.
- If Phase 3 slips by more than a week, move to the Jan 15 fallback rather than compressing the writing.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Honest numbers show the method barely works | Medium–high | Pivot at G1 (§8). The diagnostic paper is still a contribution. |
| Prior swamping doesn't hold | Medium | Report A as a negative ablation; C and D still stand. |
| Exact parallelism gives little speedup at small N | High at N ≤ 40 | That's why D goes up to N = 5000. Caching is likely the dominant win, so report each component separately. |
| GPU gives no speedup even at N = 5000 | Medium | Report it honestly as a finding; the main scaling claims are CPU-based anyway. |
| Reviewers object to novelty relative to Choi & Kim | High if mis-framed | Frame the contribution as the sampler and the analysis, never the model (§1). |
| Bandwidth (coursework, finals) | High | Cut E first, then shrink C's grid (drop K=5). Writing time is never cut. |
| Zaratan allocation or queue delays | Medium | Sort out allocation access in Phase 0 (§12). Small sweeps also run on a laptop. |
| Branch sprawl or merge conflicts | Medium | Phase 0 consolidation, the ownership table and the interface contract. |

---

## 10. Pre-submission checklist (mock review, Dec 11)

- [ ] No metric anywhere is computed from a state chosen using ground truth (the CI leakage test passes).
- [ ] Every number in the paper traces to a JSON in `results/` that records a git SHA.
- [ ] All results report mean ± std over at least 5 seeds, and no seed was selected by outcome.
- [ ] R-hat < 1.05 is reported for every configuration in the main text.
- [ ] Every speedup claim compares against the *sequential sampler on the same model*, and is reported in ESS per second as well as wall time.
- [ ] The exactness argument for the parallel sampler is in the method section.
- [ ] The word "collapsed" no longer appears, unless the sampler really is collapsed.
- [ ] The generator/inference mismatch is either removed or is the explicit subject of Experiment B.
- [ ] Related work states plainly that the model comes from Choi & Kim (2012).
- [ ] The code is public, with a one-command reproduction for each figure.
- [ ] The compute used (core-hours and GPU-hours) is reported in the appendix.
- [ ] The arXiv v2 abstract matches the new results.

---

## 11. Decision log

Append new rows; never edit old ones. If a decision is reversed, add a new row that references the old one.

| # | Date | Decision | Rationale | Decided by |
|---|---|---|---|---|
| D1 | 2026-10-09 | Target TMLR, not a conference | Its bar is rigor, not novelty; it has no deadline and no travel. | SD (proposed) |
| D2 | 2026-10-09 | Reframe the contribution as sampler + analysis | The model is from Choi & Kim (2012), and v1's headline result relies on P1. | SD (proposed) |
| D3 | 2026-10-09 | Shelve `mujoco-neural-scale` | It can't mix, it's cost-prohibitive, and it's a different paper (§3). | SD (proposed) |
| D4 | 2026-10-09 | Ground truth is allowed only in `evaluate.py`, enforced by CI | This prevents P1 from ever recurring. | SD (proposed) |
| D5 | 2026-10-09 | Exact parallelism instead of an approximate merge | It's provably correct, simple to argue and easy to test. | SD (proposed) |
| D6 | 2026-10-09 | CPU array jobs as primary compute; GPU only for D's benchmark | The tabular workload is small per chain and has many independent chains (§7). | SD (proposed) |
| D7 | 2026-10-09 | Submit by Dec 18, with Jan 15 as fallback | The expanded experiments plus finals don't fit a November date. | SD (proposed) |

*Each "proposed" entry becomes "agreed (SA, SD)" once both authors approve the PR that adds this file.*

---

## 12. Open questions

These must be resolved in Phase 0.

1. **Does SA agree with the split in §4?** In particular: SA as writing lead, and SD owning Experiment B.
2. **Who holds the Zaratan allocation, and under which PI's project?** Students get access through a PI's project. Whoever isn't on it needs to be added before Phase 0 ends.
3. **What is the SU budget on that allocation,** and does the proposed GPU cap (§7) fit within it?
4. **Did the `Result` branch use ARI-based selection?** If so, its numbers are dropped for good.
5. **Is SA happy shelving `mujoco-neural-scale`?** If SA wrote it, SA has the final say on whether it stays on the remote.
6. **Who handles the arXiv v2 submission?** SA is the submitting author on v1.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **ARI** | Adjusted Rand Index. It measures agreement between two clusterings and doesn't depend on how the labels are numbered; 1.0 is a perfect match. Computing it needs the true labels, so it can **never** be used inside the sampler. |
| **Binder loss** | A loss function over clusterings. Minimizing it against the posterior similarity matrix gives a point-estimate clustering that summarizes the whole posterior. |
| **Boltzmann expert** | The demonstrator model: π(a\|s) ∝ exp(β·Q(s,a)). Larger β means a more rational expert. |
| **CRP** | Chinese Restaurant Process, the clustering prior induced by a DP. A trajectory joins cluster k with probability ∝ (size of k), or starts a new cluster with probability ∝ α. |
| **DPM-BIRL** | Dirichlet-process mixture Bayesian IRL: a DP prior over reward functions, so that each demonstration comes from one of an unknown number of reward types. |
| **ESS** | Effective sample size: how many independent draws the autocorrelated MCMC samples are worth. ESS per second is the fair speed metric for samplers. |
| **Exact parallelism** | Parallelizing only computations that are conditionally independent given the current state, so the chain's stationary distribution is unchanged. |
| **Neal Algorithm 8** | A DP mixture sampler that keeps explicit cluster parameters and uses m auxiliary components to propose new clusters. It's what `gibbs.py` actually implements, with m = 1. It is not "collapsed". |
| **PSM** | Posterior similarity matrix: entry (i, j) is the posterior probability that trajectories i and j share a cluster. It doesn't depend on label numbering, so label switching doesn't affect it. |
| **R-hat** | A convergence diagnostic that compares between-chain and within-chain variance. Values below 1.05 suggest the chains agree. |
| **Soft VI** | Soft value iteration, which uses log-sum-exp in place of max. It's the inference model's planner; the data generator currently uses hard max (P6). |
| **SU** | Service unit, the currency Zaratan uses to bill compute time. |
| **Tempered likelihood** | The likelihood raised to a power τ (for example 1/L), which rebalances it against the prior. It's a candidate fix in Experiment A. |

---

## 14. References

- Babes-Vroman, M., Marivate, V., Subramanian, K., & Littman, M. (2011). Apprenticeship learning about multiple intentions. *ICML*.
- Chang, J., & Fisher, J. W. (2013). Parallel sampling of DP mixture models using sub-cluster splits. *NeurIPS*.
- Choi, J., & Kim, K.-E. (2012). Nonparametric Bayesian inverse reinforcement learning for multiple reward functions. *NeurIPS*.
- Dahl, D. B., Johnson, D. J., & Müller, P. (2022). Search algorithms and loss functions for Bayesian clustering. *Journal of Computational and Graphical Statistics*.
- Michini, B., & How, J. P. (2012). Bayesian nonparametric inverse reinforcement learning. *ECML-PKDD*.
- Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. *Journal of Computational and Graphical Statistics*.
- Ramachandran, D., & Amir, E. (2007). Bayesian inverse reinforcement learning. *IJCAI*.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*.
- Williamson, S., Dubey, A., & Xing, E. (2013). Parallel Markov chain Monte Carlo for nonparametric mixture models. *ICML*.
- Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement learning. *AAAI*.

---

*Doc changelog:* **2026-10-09** first version, covering the audit, scope, split, experiments, compute plan, timeline, decision log, open questions and glossary.
