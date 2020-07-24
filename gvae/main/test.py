import tensorflow as tf
from gvae.layers import get_reconstr_err, get_layers_mean, get_interpolations
from gvae.utils import *
from gvae.ops import *
from gvae.validate import run_tests
from gvae.validate.ais import AIS
from gvae.gvae import gvae
from tqdm import tqdm
import time

def test(encoder, layers, adversary, x_test, config):
    log_dir = config['log_dir']
    eval_dir = config['eval_dir']
    results_dir = os.path.join(eval_dir, "results")
    z_dim = config['z_dim']
    batch_size = config['batch_size']
    ais_nchains = config['test_ais_nchains']
    test_nais = config['test_nais']
    is_ac = config['is_ac']

    z_sampled = tf.random_normal([batch_size, z_dim])
    gvae_test = gvae(encoder, layers, adversary, x_test, z_sampled, config, is_training=False)

    stats_scalar = {
        'loss_primal': gvae_test.loss_primal,
        'loss_dual': gvae_test.loss_dual,
    }

    stats_dist = {
        'ELBO': gvae_test.ELBO,
        'KL': gvae_test.KL,
        'reconst_err': gvae_test.reconst_err,
        'z': gvae_test.z_real,
    }

    params_posterior = [gvae_test.z_mean, tf.log(gvae_test.z_std)]
    eps_scale = gvae_test.z_std

    def energy0(z, theta):
        z_mean = theta[0]
        log_z_std = theta[1]
        return -get_pdf_gauss(z_mean, log_z_std, z)

    def get_z0(theta):
        z_mean = theta[0]
        z_std = tf.exp(theta[1])
        return z_mean + z_std * tf.random_normal([batch_size, z_dim])

    run_tests(layers, stats_scalar, stats_dist,
        gvae_test.x_real, params_posterior, energy0, get_z0, config,
        eps_scale=eps_scale
    )
