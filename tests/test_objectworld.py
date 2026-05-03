"""Tests for objectworld.py -- run after Step 2 on Day 1."""
import pytest
import jax.numpy as jnp

def test_transitions_shape():
    from objectworld import ObjectWorld
    env = ObjectWorld()
    T = env.transitions()
    assert T.shape == (100, 4, 100), f"Expected (100,4,100) got {T.shape}"

def test_transitions_sum_to_one():
    from objectworld import ObjectWorld
    env = ObjectWorld()
    T = env.transitions()
    sums = T.sum(axis=2)
    assert jnp.allclose(sums, 1.0), "Transition rows must sum to 1"

def test_features_shape():
    from objectworld import ObjectWorld
    env = ObjectWorld()
    phi = env.features()
    assert phi.shape == (100, 16), f"Expected (100,16) got {phi.shape}"

def test_features_sum():
    from objectworld import ObjectWorld
    env = ObjectWorld()
    phi = env.features()
    row_sums = phi.sum(axis=1)
    assert jnp.allclose(row_sums, 2.0), "Each state should have exactly 2 features active"

def test_reward_from_weights_shape():
    from objectworld import ObjectWorld
    env = ObjectWorld()
    phi = env.features()
    w = jnp.ones(16)
    R = env.reward_from_weights(w, phi)
    assert R.shape == (100, 4), f"Expected (100,4) got {R.shape}"
