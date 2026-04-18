"""Tests for gibbs.py -- run after Step 11 on Day 4."""
import pytest
import jax
import jax.numpy as jnp

def test_gibbs_sweep_runs():
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset
    from gibbs import gibbs_sweep
    from dp_prior import sample_new_weights
    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()
    trajs, _, _ = generate_dataset(env, phi, T, n_per_type=5)
    rng = jax.random.PRNGKey(0)
    rng, sk = jax.random.split(rng)
    state = {'assignments': [0]*len(trajs), 'weight_vectors': [sample_new_weights(sk, 16)]}
    rng, sk = jax.random.split(rng)
    new_state, _ = gibbs_sweep(trajs, state, phi, T, rng_key=sk)
    assert 'assignments' in new_state
    assert 'weight_vectors' in new_state
    assert len(new_state['assignments']) == len(trajs)

def test_gibbs_pure_function():
    """gibbs_sweep must be serializable for Ray."""
    import pickle
    from gibbs import gibbs_sweep
    pickle.dumps(gibbs_sweep)  # must not raise

def test_cluster_count_reasonable():
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset
    from gibbs import run_gibbs
    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()
    trajs, _, _ = generate_dataset(env, phi, T, n_per_type=5)
    state, _ = run_gibbs(trajs, phi, T, n_sweeps=50, alpha=1.0)
    n_clusters = len(state['weight_vectors'])
    assert 1 <= n_clusters <= 10, f"Unreasonable cluster count: {n_clusters}"
