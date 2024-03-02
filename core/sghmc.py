import jax
import jax.numpy as jnp
from jax.lax import fori_loop
import models, utils


def make_epoch_step(log_prob_fn, x_train, y_train, n_class, augment, batch_size,
           max_temp, max_lr, n_epochs, momentum_decay=0.98, lr_decay=True, temp_warmup=True):
    """
    Based on the paper "How Good is the Bayes Posterior in Deep Neural Networks Really?"
    Default params are identical to those in Appendix A.1.
    The mass matrix M is always the identity matrix.
    # momentum decay = 0 -> SGLD
    # momentum decay = 1 -> HMC
    # T=0 -> SGD with momentum
    """
    n_train = len(x_train)

    @jax.jit
    def step_epoch(epoch, args):
        key, params, momentum, val_history, logits = args
        key, key_augment, key_shuffle = jax.random.split(key, 3)

        # update lr and T based on schedule, the schedules are:
        # - lr: ‾‾‾\_
        # - temp.: __/‾‾
        lr = max_lr * jnp.where(lr_decay & (epoch > n_epochs/2), jnp.sin(jnp.pi*(epoch/n_epochs)), 1)
        T = max_temp * jnp.where(temp_warmup, jnp.clip(3*(epoch/n_epochs)-1, 0, 1), 1)
        h = jnp.sqrt(lr / n_train)

        # augment data
        x, y = x_train, y_train
        if augment: x = utils.augment(key_augment, x)

        # shuffle and split data into batches
        idx = jax.random.permutation(key_shuffle, n_train)
        x_batched, y_batched, logits = utils.split_into_batches(x[idx], y[idx], logits, bs=batch_size)

        # iterate through batches
        # https://github.com/google-research/google-research/blob/5391a6bc2dd8630b91462b0e86b8e2f524ed878c/cold_posterior_bnn/core/sgmcmc.py#L924
        def step_batch(i, args):
            key, params, momentum, val, logits = args

            # sample noise
            key, key_noise = jax.random.split(key)
            noise = utils.normal_like_tree(key_noise, params, std=jnp.sqrt(2*T*(1-momentum_decay)))

            # update momentum
            (val_batch, logits_batch), grad = jax.value_and_grad(log_prob_fn, has_aux=True)(params, x_batched[i], y_batched[i])
            grad = jax.tree_map(lambda x: -x/batch_size, grad)
            momentum = jax.tree_map(lambda m, g, n: momentum_decay*m - h*n_train*g + n, momentum, grad, noise)
            logits = logits.at[i].set(logits_batch)
            val += val_batch
            
            # update params
            params = jax.tree_map(lambda p, m: p+h*m, params, momentum)

            return key, params, momentum, val, logits
        args = (key, params, momentum, 0, logits)
        args = fori_loop(0, len(x_batched), step_batch, args)
        key, params, momentum, val, logits = args
        val_hist = val_history.at[epoch].set(val)

        # return logits in the same order as the unshuffled dataset
        logits = logits.reshape([n_train, n_class])[jnp.argsort(idx)]

        return key, params, momentum, val_hist, logits

    return step_epoch
