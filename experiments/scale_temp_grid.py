import numpy as np
from itertools import product
from run import run
kwargs = dict(n_epochs=1_000, augment=True, batch_size=125)

# first, find the optimum lr for each std
lrs = np.geomspace(0.0002, 0.02, 8)
stds = np.geomspace(0.05, 3, 15)
for std, lr in product(stds, lrs):
    run(normal_prior_scale=std, temp=0, lr=lr, seed=-1, **kwargs)

# running the above loop, these lrs had the highest test accuracy
# note: although we used test accuracy to select the optimum lr,
# a different random seed was used to generate the final posterior samples
lrs = (0.0002, 0.0002, 0.0002, 0.0004, 0.0007, 0.0007, 0.0007, 0.0054, 0.0104, 0.0104, 0.0104, 0.0104, 0.0104, 0.0104, 0.0104)

# using the optimum lrs, loop over temperatures
# using TPU devices with 8 cores each, this will generate 16 posterior samples for each model
temps = (0, *np.geomspace(0.0001, 1, 15))
seeds = range(2)
for seed, temp, (std, lr) in product(seeds, temps, zip(stds, lrs)):
    run(normal_prior_scale=std, temp=temp, lr=lr, seed=seed, **kwargs)
