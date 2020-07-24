import tensorflow as tf
import numpy as np
from gvae.layers import (
    full0,
    conv0, conv1, conv2, conv3,
)

layers_dict = {
    'conv0': conv0.layers,
    'conv1': conv1.layers,
    'conv2': conv2.layers,
    'conv3': conv3.layers,
    'full0': full0.layers,
}

def get_layers(model_name, config, scope='layers'):
    cond_dist = config['cond_dist']
    if cond_dist == 'gauss':
        num_out = 2
    else:
        num_out = 1
    model_func = layers_dict[model_name]
    return tf.make_template(
        scope, model_func, config=config, num_out=num_out
    )

def get_layers_mean(layers_out, config):
    return tf.sigmoid(layers_out[0])

def get_layers_samples(layers_out, config):
    cond_dist = config['cond_dist']
    batch_size = config['batch_size']
    output_size = config['output_size']
    c_dim = config['c_dim']

    if cond_dist == 'gauss':
        eps = tf.random_normal([batch_size, output_size, output_size, c_dim])
        loc = tf.sigmoid(layers_out[0])
        logscale = layers_out[1]
        samples = loc + tf.exp(logscale) * xeps
    elif cond_dist == 'bernouille':
        p = tf.sigmoid(layers_out[0])
        eps = tf.random_uniform([batch_size, output_size, output_size, c_dim])
        samples = tf.cast(eps <= p, tf.float32)

    return samples

def get_reconstr_err(layers_out, x, config):
    cond_dist = config['cond_dist']
    if cond_dist == 'gauss':
        loc = tf.sigmoid(layers_out[0])
        logscale = layers_out[1]
        err = (x - loc)*tf.exp(-logscale)
        reconst_err = tf.reduce_sum(
            0.5 * err * err + logscale + 0.5 * np.log(2*np.pi),
            [1, 2, 3]
        )
    elif cond_dist == 'bernouille':
        p_logits = layers_out[0]
        reconst_err = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=x),
          [1, 2, 3]
        )

    return reconst_err

def get_interpolations(layers, z1, z2, N, config):
    z_dim = config['z_dim']
    alpha = tf.reshape(tf.linspace(0., 1., N), [1, N, 1])

    z1 =  tf.reshape(z1, [-1, 1, z_dim])
    z2 =  tf.reshape(z2, [-1, 1, z_dim])

    z_interp = alpha * z1 + (1 - alpha) * z2
    z_interp = tf.reshape(z_interp, [-1, z_dim])

    return layers(z_interp)
