import jax
import jax.numpy as jnp
import haiku as hk
from jax.flatten_util import ravel_pytree


he_normal = hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")


def make_mlp_fn(layer_dims, output_dim=10, leaky_relu_slope=0.01):
    """Returns a forward function for an MLP of given dimensions."""

    def forward(x):
        """
        Input: [B, ...]
        Output: [B, output_dim]]
        """

        # flatten input
        x = hk.Flatten()(x) # [B, ...] -> [B, -1]

        # hidden layers
        for layer_dim in layer_dims:
            x = hk.Linear(layer_dim)(x) # [B, layer_dim]
            x = jax.nn.leaky_relu(x, leaky_relu_slope)

        # last layer
        x = hk.Linear(output_dim)(x) # [B, 1]

        return x

    return forward


def make_cnn_fn(output_dim=10):
    """
    Returns a Haiku forward function for a small MNIST CNN.
    Has 4085 params for 10 classes.
    """

    def forward(x):
        """
        Input: [B, C, L]
        Output: [B, output_dim]
        """
        # x = hk.Reshape(output_shape=[28, 28, 1])(x) # [B, 784] -> [B, 28, 28, 1]
        x = hk.Conv2D(10, 5, 5, w_init=he_normal)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(10, 5, 5, w_init=he_normal)(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(5, 5, 5, w_init=he_normal)(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(output_dim, w_init=he_normal)(x)

        return x

    return forward


class FeatureResponseNorm(hk.Module):
    """
    FRN has trainable params, so can be used in a BNN
    https://github.com/google-research/google-research/blob/dbeb9c0e0c4f6db8561966377f25412d773cbfc8/bnn_hmc/utils/models.py#L76
    """
    def __init__(self, eps=1e-6, name='frn'):
        super().__init__(name=name)
        self.eps = eps

    def __call__(self, x):
        par_shape = (1, 1, 1, x.shape[-1])  # [1,1,1,C]
        tau = hk.get_parameter('tau', par_shape, x.dtype, init=jnp.zeros)
        beta = hk.get_parameter('beta', par_shape, x.dtype, init=jnp.zeros)
        gamma = hk.get_parameter('gamma', par_shape, x.dtype, init=jnp.ones)
        nu2 = jnp.mean(jnp.square(x), axis=[1, 2], keepdims=True)
        x = x * jax.lax.rsqrt(nu2 + self.eps)
        y = gamma * x + beta
        z = jnp.maximum(y, tau)
        return z


def resnet_layer(x, output_channels, kernel_size=3, stride=1):
    """
    basic layer (illustrated as rectangles in Fig. 3 of the orig. paper)
    """
    x = hk.Conv2D(output_channels, kernel_size, stride, w_init=he_normal)(x)
    x = FeatureResponseNorm()(x)
    x = jax.nn.relu(x)
    return x


def resnet_block(x, output_channels, downscale=False):
    """
    block consists of two conv layers, ie has a single residual connection
    """
    # downstream layers
    y = resnet_layer(x, output_channels, 3, 1+downscale)
    y = resnet_layer(y, output_channels, 3, 1)

    # residual connection
    if downscale: x = resnet_layer(x, output_channels, 1, 2)

    return jax.nn.relu(x + y)


def make_resnet20_fn(num_classes=10):
    """
    Has 273,754 params for 10 classes.
    based on https://github.com/google-research/google-research/blob/dbeb9c0e0c4f6db8561966377f25412d773cbfc8/bnn_hmc/utils/models.py#L177
    """

    def forward(x):
        # conv1
        output_channels = 16
        x = resnet_layer(x, output_channels)

        # three stacks of blocks
        for stack in range(3):
            x = resnet_block(x, output_channels, stack>0)
            x = resnet_block(x, output_channels)
            x = resnet_block(x, output_channels)
            output_channels *= 2

        # (adaptive) average pooling
        x = x.mean([1, 2], keepdims=True)

        # fully-connected output layer
        x = hk.Flatten()(x)
        logits = hk.Linear(num_classes, w_init=he_normal)(x)

        return logits

    return forward


def make_nn(net_fn, key, x):
    # initialize the NN
    net = hk.transform(net_fn)
    params = net.init(key, x)
    predict_fn = lambda x, params: net.apply(params, None, x)
    return predict_fn, params
