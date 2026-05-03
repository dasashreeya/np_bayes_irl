"""Tests for eval.py -- run after Step 12 on Day 5."""
import pytest
import jax.numpy as jnp

def test_l2_error_zero():
    from eval import l2_weight_error
    w = jnp.array([1.0, -1.0, 0.0, 0.5])
    assert l2_weight_error(w, w) < 1e-5

def test_ari_perfect():
    from eval import adjusted_rand_index
    assert adjusted_rand_index([0,0,1,1], [0,0,1,1]) == 1.0

def test_ari_permutation():
    from eval import adjusted_rand_index
    # Label permutation should still be perfect
    assert adjusted_rand_index([0,0,1,1], [1,1,0,0]) == 1.0
