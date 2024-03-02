import jax
import jax.numpy as jnp
import operator as op
from jax import vmap
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from functools import partial
import utils


def add_normal_prior(posterior_fn, std, ds_size):

    def out_fn(params, x, y):
        
        # get original posterior
        log_posterior, logits = posterior_fn(params, x, y)
        batch_size = len(logits)

        # add normal prior rescaled to batch size
        # the sum of this prior over batches equals the full-batch prior
        if std is not None:
            dx = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), params))
            log_prior = -0.5 * dx / std**2
            log_prior *= batch_size / ds_size
            log_posterior += log_prior

        return log_posterior, logits

    return out_fn


def make_categorical_pdf(predict_fn):

    def out_fn(params, x, y):
        logits = predict_fn(x, params)
        batch_size, n_class = logits.shape
        y_one_hot = jax.nn.one_hot(y, n_class)
        log_like = (y_one_hot * jax.nn.log_softmax(logits)).sum()
        return log_like, logits

    return out_fn


def make_dirichlet_pdf(predict_fn, alpha_prior, posterior=True, clip_val=None, clip_logp=None):
    
    def out_fn(params, x, y):

        # predict
        logits = predict_fn(x, params)
        batch_size, n_class = logits.shape
        log_probs = jax.nn.log_softmax(logits)
    
        # get prior
        alpha = alpha_prior * jnp.ones(n_class)
        log_probs_clipped = log_probs
        if clip_logp is not None: log_probs_clipped = log_probs_clipped.clip(min=clip_logp)
        log_prior = ((alpha-1) * log_probs_clipped).sum(-1)
        if clip_val is not None: log_prior = log_prior.clip(max=clip_val)
        log_prob = log_prior.sum()

        # get posterior
        if posterior:
            y_one_hot = jax.nn.one_hot(y, n_class)
            log_like = (y_one_hot * log_probs).sum()
            log_prob += log_like

        return log_prob, logits

    return out_fn


def make_ndg_post_fn(predict_fn, alpha_prior, from_logprobs=False):
    """
    Gaussian approximation of the Dirichlet posterior.
    """

    def out_fn(params, x, y):

        # predict
        logits = predict_fn(x, params)
        batch_size, n_class = logits.shape

        # the guassian approximation can either be applied over logits (defualt)
        # or over logprobs, which is just logits shifted by a scalar value
        if from_logprobs: logits = jax.nn.log_softmax(logits)

        # get posterior
        y_one_hot = jax.nn.one_hot(y, n_class)
        alpha_post = alpha_prior + y_one_hot # [batch_size, n_class]
        var = jnp.log(1 / alpha_post + 1) # [batch_size, n_class]
        mean = jnp.log(alpha_post) - var / 2 # [batch_size, n_class]
        if from_logprobs: mean -= mean.max() # logprobs are (-inf, 0)
        log_posterior = (-0.5 * (logits - mean)**2 / var).sum()

        return log_posterior, logits

    return out_fn


def make_ndg_factorized_fn(predict_fn, alpha, ndg_prior=True, ndg_likelihood=True):

    # get gaussian mean and variance
    # for convenience, let [1] correspond to the true class and [0] correspond to the other classes
    alpha = jnp.array([alpha, 1+alpha])
    var = jnp.log(1 / alpha + 1)
    mu = jnp.log(alpha) - var / 2
    mu -= mu[1] # set mu1 to 0

    # get likelihood polynomial terms
    a = (var[0]*mu[1]-var[1]*mu[0]) / (var[0]*var[1])
    b = (var[1]-var[0])/(2*var[0]*var[1])
    print(f'likelihood decomposition: {a=}, {b=}')

    def out_fn(params, x, y):

        # predict
        logits = predict_fn(x, params)
        log_probs = jax.nn.log_softmax(logits)
        batch_size, n_class = log_probs.shape

        # get prior
        log_prob = 0
        if ndg_prior:
            log_prior = (-0.5 * (log_probs - mu[0])**2 / var[0]).sum()
            log_prob += log_prior

        # get likelihood
        # if ndg_likelihood=True use NDG likelihood; otherwise use standard categorical likelihood
        y_one_hot = jax.nn.one_hot(y, n_class)
        log_like = (y_one_hot * log_probs).sum(-1)
        if ndg_likelihood: log_like = a*log_like + b*log_like**2
        log_prob += log_like.sum(-1)

        return log_prob, logits

    return out_fn


def make_confidence_pdf(predict_fn, T, posterior):
    
    def out_fn(params, x, y):

        # predict
        logits = predict_fn(x, params)
        batch_size, n_class = logits.shape
        log_probs = jax.nn.log_softmax(logits)
    
        # get prior
        log_prob = (1/T-1) * log_probs.max(-1).sum()

        # optionally add likelihood
        if posterior:
            y_one_hot = jax.nn.one_hot(y, n_class)
            categ_likelihood = (y_one_hot * log_probs).sum()
            log_prob += categ_likelihood

        return log_prob, logits

    return out_fn
