
import numpy as np
import tensorflow as tf
from tensorflow import distributions as distr

def gpreg(y, dist, params, niter, Sigma_proposal):
    """TODO: Docstring for gp.
    :returns: TODO

    """

    # Precomputation outside tensorflow
    n = y.size
    X = np.concatenate([np.ones([n, 1]), np.eye(n)], 1)
    samples = np.zeros((niter, 2))

    # Constants
    Sigma_proposal = tf.constant(Sigma_proposal, dtype = tf.float32)
    prior_c_sigma2 = tf.constant(1.0)
    Sigma_proposal_chol = tf.cholesky(Sigma_proposal)

    # Parameters
    params_log = tf.Variable(params, dtype = tf.float32)
    sigma2 = tf.exp(params_log[0,0])
    phi = tf.exp(params_log[1,0])

    # Data
    tf_dis = tf.placeholder(dtype = tf.float32)
    tf_y = tf.placeholder(dtype = tf.float32)
    feed_dict = {
            tf_dis: dist,
            tf_y: y
            }

    # Computation
    Sigma_gp = tf_cov_exp(tf_dis, sigma2, phi, 0.0)
    Sigma_marginal = prior_c_sigma2 + Sigma_gp
    Sigma_z = Sigma_marginal + tf.eye(n)

    # Metropolis Hastings sampler
    updater = update(params_log, Sigma_proposal_chol, tf_dis, Sigma_z,
            tf_y, prior_c_sigma2)

    # Iterate trough MH sampler
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(niter):
        params_new = sess.run(updater, feed_dict)
        samples[i, :] = np.reshape(np.exp(params_new), 2)

    sess.run(updater, feed_dict)
    return samples

def update(params_log, Sigma_proposal_chol, tf_dis, Sigma_z, tf_y,
        prior_c_sigma2):
    """TODO: Docstring for update.

    :x: TODO
    :returns: TODO

    """
    n = tf.size(tf_y)
    zeros_n = tf.zeros([n, 1])
    params_log_aux = params_log + \
            tf.matmul(Sigma_proposal_chol,
                    tf.distributions.Normal(0.0, 1.0).sample(tf.shape(params_log)))
    sigma2_aux = tf.exp(params_log_aux[0,0])
    phi_aux = tf.exp(params_log_aux[1,0])

    Sigma_gp_aux = tf_cov_exp(tf_dis, sigma2_aux, phi_aux, 0.0)
    Sigma_marginal_aux = prior_c_sigma2 + Sigma_gp_aux
    Sigma_z_aux = Sigma_marginal_aux + tf.eye(n)
    acceptance_prob_log = dmvnorm(tf_y, zeros_n, Sigma_z_aux) + \
            distr.Normal(tf.log(1.0), 0.4).log_prob(params_log_aux[0,0]) + \
            distr.Normal(tf.log(0.08), 0.4).log_prob(params_log_aux[1,0]) - \
            dmvnorm(tf_y, zeros_n, Sigma_z) - \
            distr.Normal(tf.log(1.0), 0.4).log_prob(params_log[0,0]) - \
            distr.Normal(tf.log(0.08), 0.4).log_prob(params_log[1,0])

    uniform_log = tf.log(tf.distributions.Uniform().sample())
    params_log_new = tf.cond(tf.greater(acceptance_prob_log,uniform_log),\
            lambda: params_log_aux, lambda: params_log)

    return tf.assign(params_log, params_log_new)


def dmvnorm(y, mean, sigma):
    L = tf.cholesky(sigma)
    kern_sqr = tf.matrix_triangular_solve(L, y - mean, lower = True)
    n = tf.cast(tf.shape(sigma)[1], tf.float32)
    loglike = - 0.5 * n * tf.log( 2.0 * np.pi)
    loglike += - tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
    loglike += - 0.5 * tf.reduce_sum(tf.square(kern_sqr))
    return(loglike)

def tf_cov_exp(d, sigma2, phi, nugget):
    S = sigma2 * tf.exp(- d / phi) + nugget * tf.eye(tf.shape(d)[1])
    return(S)

def test():
    """TODO: Docstring for test.

    :x: TODO
    :returns: TODO

    """

    sess0 = tf.Session()
    bla = tf.ones(10) - tf.ones([10,1])
    return sess0.run(bla)

