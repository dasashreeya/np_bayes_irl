 
from parallel import run_parallel
from objectworld import ObjectWorld
from expert_demos import generate_dataset

env = ObjectWorld()
phi = env.features()
T   = env.transitions()
trajs, _, _ = generate_dataset(env, phi, T, n_per_type=5)
run_parallel(trajs, phi, T, n_workers=2, n_sweeps=3)
print('parallel smoke test ✓')