import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Literal

image_stats = {
    'mnist': ((0.1307,), (0.3081,)),
    'cifar10': ((0.49, 0.48, 0.44), (0.2, 0.2, 0.2)),
}

def load(name: Literal['mnist', 'cifar10'], split: Literal['train', 'test']):
    """
    Based on https://github.com/izmailovpavel/bnn_covariate_shift/blob/main/bnn_hmc/utils/data_utils.py
    """

    # load dataset
    ds, dataset_info = tfds.load(name, split=split, as_supervised=True, with_info=True)
    n_examples = dataset_info.splits[split].num_examples
    n_classes = dataset_info.features['label'].num_classes

    # cast from int8 to float
    def img_to_float(image, label):
        using_float64 = jnp.zeros(1).dtype == jnp.float64
        dtype = tf.float64 if using_float64 else tf.float32
        return tf.image.convert_image_dtype(image, dtype), label
    ds = ds.map(img_to_float)

    # normalize to zero mean and unit variance
    def img_normalize(image, label):
        mean, std = image_stats[name]
        image -= tf.constant(mean, dtype=image.dtype)
        image /= tf.constant(std, dtype=image.dtype)
        return image, label
    ds = ds.map(img_normalize)

    # put whole dataset into a single batch
    ds = ds.batch(n_examples)

    # return dataset as a jax array
    ds = tfds.as_numpy(ds)
    x_train, y_train = next(iter(ds))
    return jnp.array(x_train), jnp.array(y_train), n_classes


def unnormalize(image, ds_name):
    mean, std = image_stats[ds_name]
    return jnp.array(mean) + jnp.array(std)*image
