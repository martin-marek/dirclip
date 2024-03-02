from run import run

"""
comparison of tempering, ndg, and dirclip; all trained from random initialization
"""
kwargs = dict(lr=1e-4, n_epochs=10_000, augment=True, normal_prior_scale=0.1)

# categorical, varying temp.
for seed in range(3):
    for T in (0, 0.0001, 0.001, 0.01, 0.1, 0.3, 1):
        run(distribution='categorical', T=T, seed=seed, **kwargs)

# factorized NDG, varying alpha
for seed in range(3):
    for distribution in ('ndg-logits', 'ndg-logprobs', 'ndg-likelihood', 'ndg-prior')
        for alpha in (1e-6, 1e-5, 1e-4, 1e-3, 0.0018, 0.0032 , 0.0056, 0.01, 0.022, 0.046, 0.1):
            run(distribution=distribution, distribution_param=alpha, seed=seed, **kwargs)

# dirichlet clipped, varying alpha
for seed in range(3):
    for alpha in (0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.84, 0.86, 0.88, 0.91, 0.94, 0.97, 0.98, 0.99, 1):
        run(distribution='dirclip-50', distribution_param=alpha, seed=seed, **kwargs)

# for two specific models, get 200 posterior samples (25 seeds x 8 TPU cores) to test
# how many posterior samples we actually need for an accurate approximation
for seed in range(25):
    run(distribution='categorical', temp=0.1, seed=seed, **kwargs)
    run(distribution='dirclip-50', distribution_param=0.85, seed=seed, **kwargs)
