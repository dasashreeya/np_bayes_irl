"""
maxent_irl.py  |  Owner: P1  |  Built: Day 2
MaxEnt IRL baseline. Finds w such that policy feature expectations match expert.
This is the comparison baseline -- your method should beat this.
"""
import jax
import jax.numpy as jnp
from mdp_utils import soft_value_iteration, boltzmann_policy

def feature_expectations(trajectories, phi):
    """Empirical feature counts from expert trajectories."""
    # TODO Step 5 Day 2 -- P1
    pass

def policy_feature_expectations(pi, phi, T, gamma=0.95, n_iter=50):
    """Expected feature counts under policy pi."""
    # TODO Step 5 Day 2 -- P1
    pass

def maxent_irl(trajectories, phi, T, gamma=0.95, beta=1.0, lr=0.01, n_iters=100):
    """Run MaxEnt IRL. Returns recovered weight vector w."""
    # TODO Step 5 Day 2 -- P1
    pass
