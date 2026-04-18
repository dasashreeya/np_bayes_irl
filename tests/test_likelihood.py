"""Tests for likelihood.py -- run after Step 8 on Day 3."""
import pytest
import jax
import jax.numpy as jnp

def test_likelihood_is_finite():
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset
    from likelihood import log_likelihood
    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()
    trajs, true_w, _ = generate_dataset(env, phi, T, n_per_type=5)
    ll = log_likelihood(true_w[0], trajs[:5], phi, T)
    assert jnp.isfinite(ll), f"log_likelihood must be finite, got {ll}"

def test_true_beats_random():
    from objectworld import ObjectWorld
    from expert_demos import generate_dataset
    from likelihood import log_likelihood
    from dp_prior import sample_new_weights
    env = ObjectWorld()
    phi = env.features()
    T   = env.transitions()
    trajs, true_w, _ = generate_dataset(env, phi, T, n_per_type=10)
    w_random = sample_new_weights(jax.random.PRNGKey(99), 16)
    ll_true   = log_likelihood(true_w[0], trajs[:10], phi, T)
    ll_random = log_likelihood(w_random,  trajs[:10], phi, T)
    assert ll_true > ll_random, "True weights must score higher than random"
