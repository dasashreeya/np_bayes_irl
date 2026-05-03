import jax
from objectworld import ObjectWorld
from expert_demos import generate_dataset
from dp_prior import sample_new_weights
from likelihood import log_likelihood

env = ObjectWorld()
phi = env.features()
T   = env.transitions()
trajs, true_w, true_z = generate_dataset(env, phi, T)

rng      = jax.random.PRNGKey(0)
w_random = sample_new_weights(rng, 16)

ll_true   = log_likelihood(true_w[0], trajs[:10], phi, T)
ll_random = log_likelihood(w_random,  trajs[:10], phi, T)

print(f"ll_true:   {ll_true:.2f}")
print(f"ll_random: {ll_random:.2f}")
print(f"True scores higher: {ll_true > ll_random}")