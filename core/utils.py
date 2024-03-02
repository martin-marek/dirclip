import pathlib
import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.special import logsumexp


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    # https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_map(
        lambda l, k: mean + std*jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def all_shards_close(x):
    return all(jnp.allclose(x[0], x[1]) for i in range(1, len(x)))


def split_into_batches(*arrays, n=None, bs=None):
    """
    - n: number of batches
    - bs: batch size
    Either `n` or `bs` has to be specified.
    """

    # compute num. of batches or batch size based on the other parameter
    assert (n is None) ^ (bs is None)
    bs = bs or len(arrays[0]) // n
    n = n or len(arrays[0]) // bs

    # batch data
    return [x[:n*bs].reshape([n, bs, *x.shape[1:]]) for x in arrays]


def average_predictions(logits):
    chain_len, ds_len, n_classes = logits.shape
    logprobs = jax.nn.log_softmax(logits, 2) # map logits -> class probs
    logprobs = logsumexp(logprobs, 0) - jnp.log(chain_len) # average predictions
    return logprobs


def augment(key, images):
    """
    Matches the paper "How Good is the Bayes Posterior in Deep Neural Networks Really?" exactly.
    https://github.com/google-research/google-research/blob/fa49cd0231954458e2b02278d95528b70245f06d/cold_posterior_bnn/datasets.py#L113
    """
    n, h, w, c = images.shape
    key_flip, key_crop = jax.random.split(key)
    
    # left-right flip
    do_flip = jax.random.bernoulli(key_flip, 0.5, [n])
    images = jnp.where(do_flip[:, None, None, None], jnp.flip(images, 2), images)
    
    # pad + crop
    images = jnp.pad(images, [[0, 0], [4, 4], [4, 4], [0, 0]])
    dx, dy = jax.random.randint(key_crop, [2], 0, 7)
    images = jax.lax.dynamic_slice(images, (0, dy, dx, 0), (n, h, w, c))

    return images


def image_grid(images, grid_width, grid_height):
    """
    Utility function to plot several images as a grid.
    """
    n, h, w, c = images.shape
    grid = images[:grid_width*grid_height] # discard images
    grid = grid.reshape([grid_width, grid_height, h, w, c])
    grid = jnp.transpose(grid, (1, 2, 0, 3, 4))
    grid = grid.reshape([grid_height*h, grid_width*w, c])
    return grid


def save_model(chain, logits_train, logits_test, val_hist, model_dir):
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    jnp.save(f'{model_dir}/chain.npy', chain)
    jnp.save(f'{model_dir}/logits_train.npy', logits_train)
    jnp.save(f'{model_dir}/logits_test.npy', logits_test)
    jnp.save(f'{model_dir}/val_hist.npy', val_hist)
