
def gpreg(y, dist, params, niter, Sigma_proposal):
    """TODO: Docstring for gp.
    :returns: TODO

    """

    # Precomputation outside tensorflow
    n = y.size
    X = np.concatenate([np.ones([n, 1]), np.eye(n)], 1)

    # Allocate memory for samples
    samples = np.zeros((niter, 2))

    # Constants
    Sigma_proposal = tf.constant(Sigma_proposal, dtype = tf.float32)
    prior_c_sigma2 = tf.constant(1.0)
    Sigma_proposal_chol = tf.cholesky(Sigma_proposal)

    # Parameters
    params_log = tf.Variable(params, dtype = tf.float32)

    # Data
    tf_dis = tf.placeholder(dtype = tf.float32)
    tf_y = tf.placeholder(dtype = tf.float32)
    feed_dict = {
            tf_dis: dist,
            tf_y: y
            }

    # Log posterior of initial parameters
    aux = initialize(posterior(params_log, tf_dis, tf_y, prior_c_sigma2), feed_dict)
    params_log_posterior = tf.Variable(aux, dtype = tf.float32)

    # Metropolis Hastings sampler
    updater = update(params_log, params_log_posterior,
            Sigma_proposal_chol, tf_dis, tf_y, prior_c_sigma2)

    # Iterate trough MH sampler
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(niter):
        params_new, params_log_posterior_new = sess.run(updater, feed_dict)
        samples[i, :] = np.reshape(np.exp(params_new), 2)
        # print(sess.run(params_log_posterior, feed_dict))

    sess.run(updater, feed_dict)
    return samples


def update(params_log, params_log_posterior, Sigma_proposal_chol, tf_dis, tf_y,
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

    params_log_aux_posterior = posterior(params_log_aux, tf_dis, tf_y, prior_c_sigma2)
    acceptance_prob_log = params_log_aux_posterior - params_log_posterior

    uniform_log = tf.log(tf.distributions.Uniform().sample())
    params_log_new = tf.cond(tf.greater(acceptance_prob_log,uniform_log),\
            lambda: params_log_aux, lambda: params_log)
    params_log_posterior_new = tf.cond(tf.greater(acceptance_prob_log,uniform_log),\
            lambda: params_log_aux_posterior, lambda: params_log_posterior)

    op1 = tf.assign(params_log, params_log_new)
    op2 = tf.assign(params_log_posterior, params_log_posterior_new)
    return op1, op2
    # return params_log_new


def posterior(params_log, tf_dis, tf_y, prior_c_sigma2):
    """TODO: Docstring for posterior.

    :params_log: TODO
    :tf_dis: TODO
    :tf: TODO
    :returns: TODO

    """
    n = tf.size(tf_y)
    zeros_n = tf.zeros([n, 1])
    sigma2 = tf.exp(params_log[0,0])
    phi = tf.exp(params_log[1,0])

    Sigma_gp = tf_cov_exp(tf_dis, sigma2, phi, 0.0)
    Sigma_marginal = prior_c_sigma2 + Sigma_gp
    Sigma_z = Sigma_marginal + tf.eye(n)
    posterior_prob_log = dmvnorm(tf_y, zeros_n, Sigma_z) + \
            distr.Normal(tf.log(1.0), 0.4).log_prob(params_log[0,0]) + \
            distr.Normal(tf.log(0.08), 0.4).log_prob(params_log[1,0])
    return posterior_prob_log

