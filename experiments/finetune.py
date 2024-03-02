from run import run

# fine-tune DirClip models
kwargs = dict(lr=1e-7, n_epochs=10_000, augment=True, normal_prior_scale=0.1, lr_decay=False, temp_warmup=False)
for seed in range(3):
    for clip_val in (10, 20, 50):
        for alpha in (1, 0.85, 0.4, -0.5):
            run(distribution=f'dirclip-{clip_val}', distribution_param=alpha, seed=seed, init_params=f'/PATH_TO_PRETRAINED_WEIGTHS/chain{seed}.npy', **kwargs)
