import os
import fire
import jax
import jax.numpy as jnp
from jax import pmap
from time import time
from typing import List, Optional, Union
from tqdm import tqdm
import models, datasets, distributions, sghmc, utils


def run(model_name: str = 'resnet20',
        ds_name: str = 'cifar10',
        distribution: str = 'categorical',
        distribution_param: float = 0,
        normal_prior_scale: Optional[float] = None,
        posterior: bool = True,
        temp: float = 1,
        temp_warmup: bool = True,
        augment: bool = False,
        lr: Union[list, float] = 0.001,
        lr_decay: bool = True,
        batch_size: int = 125,
        n_epochs: int = 1000,
        seed: int = 0,
        save: bool = True,
        init_params: Optional[str] = None,
    ):
    """
    Runs a single model configuration.
    """
    train_config = locals()
    print(f'{train_config=}')
    key = jax.random.PRNGKey(seed)
    n_dev = jax.device_count()
    t0 = time()

    # load dataset
    t = time()
    ds_key = jax.random.PRNGKey(0)
    x_train, y_train, n_class = datasets.load(ds_name, 'train')
    x_test, y_test, n_class = datasets.load(ds_name, 'test')
    print(f'time load ds: {time()-t:.2f}s')

    # load model
    t = time()
    key, key_params, key_momentum = jax.random.split(key, 3)
    if model_name=='mlp': net_fn = models.make_mlp_fn([20, 20, 20], n_class)
    if model_name=='cnn': net_fn = models.make_cnn_fn(n_class)
    if model_name=='resnet20': net_fn = models.make_resnet20_fn(n_class)
    predict_fn, params = models.make_nn(net_fn, key_params, x_train[:1])
    print('num. model params:', sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f'time load model: {time()-t:.2f}s')

    # initialize a different model for each device
    f = lambda key: models.make_nn(net_fn, key, x_train[:1])[1]
    params = pmap(f)(jax.random.split(key_params, n_dev))
    momentum = utils.normal_like_tree(key_momentum, params)

    # optionally start training from an existing checkpoint
    if init_params is not None:
        params = jnp.load(init_params, allow_pickle=True).item()

    # set likelihood
    if distribution=='categorical': log_prob_fn = distributions.make_categorical_pdf(predict_fn)
    if distribution=='dirichlet': log_prob_fn = distributions.make_dirichlet_pdf(predict_fn, distribution_param, posterior)
    if distribution.startswith('dirclip'): log_prob_fn = distributions.make_dirichlet_pdf(predict_fn, distribution_param, posterior, clip_logp=-float(distribution.split('-')[1]))
    if distribution=='ndg-logits': log_prob_fn = distributions.make_ndg_post_fn(predict_fn, distribution_param)
    if distribution=='ndg-logprobs': log_prob_fn = distributions.make_ndg_post_fn(predict_fn, distribution_param, from_logprobs=True)
    if distribution=='ndg-prior': log_prob_fn = distributions.make_ndg_factorized_fn(predict_fn, distribution_param, ndg_prior=True, ndg_likelihood=False)
    if distribution=='ndg-likelihood': log_prob_fn = distributions.make_ndg_factorized_fn(predict_fn, distribution_param, ndg_prior=False, ndg_likelihood=True)
    if distribution=='confidence': log_prob_fn = distributions.make_confidence_pdf(predict_fn, distribution_param, posterior)

    # optionally add normal prior over weights
    log_prob_fn = distributions.add_normal_prior(log_prob_fn, normal_prior_scale, len(x_train))

    # run sghmc
    keys_sghmc = jax.random.split(key, n_dev)
    val_hist = jnp.zeros([n_dev, n_epochs])
    logits_train = jnp.zeros([n_dev, len(x_train), n_class])
    step_epoch = sghmc.make_epoch_step(log_prob_fn, x_train, y_train, n_class, augment, batch_size, temp, lr, n_epochs, lr_decay=lr_decay, temp_warmup=temp_warmup)
    step_epoch = pmap(step_epoch, in_axes=(None, 0))
    pbar = tqdm(range(n_epochs))
    for i in pbar:
        keys_sghmc, params, momentum, val_hist, logits_train = step_epoch(i, (keys_sghmc, params, momentum, val_hist, logits_train))
        acc = (logits_train.argmax(2) == y_train[None]).mean()
        conf = jax.nn.softmax(logits_train).max(2).mean()
        val = val_hist[:, i].mean()
        logit_range = logits_train.max() - logits_train.min()
        pbar.set_postfix_str(f'{acc=:6.2%}, {conf=:6.2%}, {val=:.0f}, {logit_range=:.1f}')

    # get test accuracy
    logits_test = pmap(predict_fn, in_axes=(None, 0))(x_test, params)
    logprobs_test = utils.average_predictions(logits_test)
    test_acc = (logprobs_test.argmax(1) == y_test).mean()
    print(f'{test_acc=:6.2%}')

    # save model
    if save:
        finetuned = init_params is not None
        out_dir = os.path.expanduser(f"~/weights/model='{model_name}',ds='{ds_name}',dist='{distribution}',dist_param={distribution_param:g},std={normal_prior_scale},{temp=:g},{augment=},{finetuned=},{seed=}")
        utils.save_model(params, logits_train, logits_test, val_hist, out_dir)
    print(f'total time: {(time()-t0)/60:.2f} min.')

if __name__ == "__main__":
    fire.Fire(run)
