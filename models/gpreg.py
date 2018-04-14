
import numpy as np
import tensorflow as tf
# import tensorflow.distributions as distr

def gp(y, dist, params, niter, Sigma_proposal):
    """TODO: Docstring for gp.
    :returns: TODO

    """

    samples = np.zeros((niter, 2))

    # Constants
    Sigma_proposal = tf.constant(Sigma_proposal, dtype = tf.float32)
    prior_c_sigma2 = tf.constant(1.0)

    # Auxiliary constants

    # Parameters
    sigma2_log = tf.Variable(params[0])
    phi_log = tf.Variable(params[1])

    # Transformations
    sigma2 = tf.exp(sigma2_log)
    phi = tf.exp(phi_log)

    # Data
    tf_dis = tf.placeholder(dtype = tf.float32)
    # tf_x = tf.placeholder(dtype = tf.float32)
    tf_y = tf.placeholder(dtype = tf.float32)

    # Computation
    n = tf.size(y)
    ones_n = tf.ones([n, 1])
    zeros_n = tf.zeros([n, 1])
    eye_n = tf.eye(n)
    X = tf.concat([ones_n, eye_n], 1)
    Sigma_proposal_chol = tf.cholesky(Sigma_proposal)


    params_log = tf.expand_dims(tf.stack([sigma2_log, phi_log], 0), 1)
    Sigma_gp = tf_cov_exp(tf_dis, sigma2, phi, 0)
    Sigma_marginal = prior_c_sigma2 + Sigma_gp
    Sigma_z = Sigma_marginal + eye_n
    updater = update(params_log, Sigma_proposal_chol, tf_dis, eye_n, Sigma_z,
            zeros_n, tf_y, prior_c_sigma2)

    # Arguments
    feed_dict = {
            tf_dis: dist,
            tf_y: y
            }

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(niter):
        bla = sess.run(updater, feed_dict)
        samples[i, :] = np.reshape(bla, 2)

    return samples

def update(params_log, Sigma_proposal_chol, tf_dis, eye_n, Sigma_z, zeros_n, tf_y,
        prior_c_sigma2):
    """TODO: Docstring for update.

    :x: TODO
    :returns: TODO

    """
    params_log_aux = params_log + \
            tf.matmul(Sigma_proposal_chol,
                    tf.distributions.Normal(0.0, 1.0).sample(tf.shape(params_log)))
    sigma2_aux = tf.exp(params_log[0])
    phi_aux = tf.exp(params_log[1])
    Sigma_gp_aux = tf_cov_exp(tf_dis, sigma2_aux, phi_aux, 0)
    Sigma_marginal_aux = prior_c_sigma2 + Sigma_gp_aux
    Sigma_z_aux = Sigma_marginal_aux + eye_n
    # Sigma_z_aux_chol = tf.cholesky(Sigma_z_aux)
    acceptance_prob_log = dmvnorm(tf_y, zeros_n, Sigma_z_aux) + \
            tf.distributions.Normal(tf.log(1.0),
                    0.4).log_prob(tf.reshape(params_log_aux[0], [])) + \
            tf.distributions.Normal(tf.log(0.03),
                    0.4).log_prob(tf.reshape(params_log_aux[1],[])) - \
            dmvnorm(tf_y, zeros_n, Sigma_z) + \
            tf.distributions.Normal(tf.log(1.0),
                    0.4).log_prob(tf.reshape(params_log[0], [])) + \
            tf.distributions.Normal(tf.log(0.03),
                    0.4).log_prob(tf.reshape(params_log[1], []))

    uniform = tf.log(tf.distributions.Uniform().sample())
    params_log_new = tf.cond(tf.greater(acceptance_prob_log,uniform),\
            lambda: params_log_aux, lambda: params_log)

            #         lambda: params_log_aux,
            #         lambda: tf.zeros(tf.shape(params_log)))
            #
            # 1.0
    # params_log_aux = tf.matmul(Sigma_proposal_chol, tf.random_normal([2,1]))
    # params_log
    # +

    return tf.exp(params_log_new)



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


