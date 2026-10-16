"""
run_serial.py  |  Owner: SA
End-to-end serial DPM-BIRL on ObjectWorld: several chains, one posterior
summary, scored against ground truth only at the end.

Replaces the v1 script, which reported the max-ARI state and restarted the
chain to it (P1, see docs/audits/result_branch.md). Ground truth is used in
exactly one place here: the call to evaluate().

    python run_serial.py --chains 4 --sweeps 500 --burn-in 100 --out run.json
"""

import argparse
import json
import time

from evaluate import evaluate
from expert_demos import generate_dataset
from likelihood_backend import LikelihoodBackend
from objectworld import ObjectWorld
from sampler import SamplerConfig, run_chain
from summarize import summarize


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serial DPM-BIRL on ObjectWorld.")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--sweeps", type=int, default=500)
    ap.add_argument("--burn-in", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--step-size", type=float, default=0.05)
    ap.add_argument("--n-mh-steps", type=int, default=1)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--n-per-type", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="chain c uses seed + c")
    ap.add_argument("--out", default=None, help="write traces + metrics as JSON")
    args = ap.parse_args(argv)

    env = ObjectWorld()
    phi, T = env.features(), env.transitions()
    trajs, true_w, true_z = generate_dataset(env, phi, T, n_per_type=args.n_per_type,
                                             seed=args.data_seed)
    backend = LikelihoodBackend(phi, T, args.gamma, args.beta, trajs)
    cfg = SamplerConfig(alpha=args.alpha, n_sweeps=args.sweeps, burn_in=args.burn_in,
                        step_size=args.step_size, n_mh_steps=args.n_mh_steps)

    traces = []
    for c in range(args.chains):
        t0 = time.perf_counter()
        traces.append(run_chain(backend, cfg, rng_key=args.seed + c))
        tr = traces[-1]
        print(f"chain {c}: {time.perf_counter() - t0:.1f}s | K mean {tr.K.mean():.2f} "
              f"| accept {tr.accept_rate.mean():.2f}")

    summary = summarize(traces)
    metrics = evaluate(summary, true_z, true_w)  # the only use of ground truth
    for k in ("K_true", "K_mode", "p_K_true", "K_point", "ari_point", "l2_matched",
              "n_extra", "rhat_K", "rhat_log_post", "ess_log_post", "accept_rate"):
        print(f"{k:>14s}: {metrics[k]}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "config": cfg.to_dict(), "metrics": metrics,
                       "traces": [t.to_dict() for t in traces]}, f)
    return metrics


if __name__ == "__main__":
    main()
