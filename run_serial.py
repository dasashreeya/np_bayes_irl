"""
run_serial.py  |  Owner: Both  |  Built: Day 5
End-to-end serial NP-Bayes IRL experiment.
Logs all metrics to W&B.
"""
import wandb
from objectworld import ObjectWorld
from expert_demos import generate_dataset
from gibbs import gibbs_sweep
from dp_prior import sample_new_weights
from eval import log_metrics
import jax


def run_experiment(n_sweeps=500, alpha=1.0, seed=0):
    env  = ObjectWorld()
    phi  = env.features()
    T    = env.transitions()
    trajs, true_w, true_z = generate_dataset(env, phi, T, n_per_type=10)

    run = wandb.init(project='np-bayes-irl', name='serial',
                     config={'n_sweeps': n_sweeps, 'alpha': alpha, 'seed': seed,
                             'env': 'ObjectWorld', 'mode': 'serial'})

    rng = jax.random.PRNGKey(seed)
    rng, sk = jax.random.split(rng)
    state = {
        'assignments':    [0] * len(trajs),
        'weight_vectors': [sample_new_weights(sk, phi.shape[1])]
    }

    for sweep in range(n_sweeps):
        rng, sk = jax.random.split(rng)
        state, sk = gibbs_sweep(trajs, state, phi, T,
                                alpha=alpha, rng_key=sk)
        if sweep % 10 == 0:
            log_metrics(sweep, state, true_w, true_z, run)

    run.finish()
    return state


if __name__ == '__main__':
    state = run_experiment(n_sweeps=300)
    print('Final n_clusters:', len(state['weight_vectors']))
    print('Final assignments:', state['assignments'])
