
import numpy as np
import tensorflow as tf
from tensorflow import distributions as distr

class geo_normal(object):

    """Univariate Geostatistical Linear Model"""

    def __init__(self, y, dist, params):
        """
        Constructor:

        :y: response variable
        :dist: distance matrix
        :params: initial parameters

        """
        self.n = y.size
        self.X = np.concatenate([np.ones([self.n, 1]), np.eye(self.n)], 1)
        self.params_log = tf.Variable(params, dtype = tf.float32)
        self.tf_y = tf.placeholder(dtype = tf.float32)
        self.tf_dis = tf.placeholder(dtype = tf.float32)
        self.feed_dict = {
                self.tf_dis: dist,
                self.tf_y: y
                }
        self.prior_c_sigma2 = tf.constant(1.0)


    def posterior(self, params_log):
        """TODO: Docstring for posterior.

        :params_log: TODO
        :tf_dis: TODO
        :tf: TODO
        :returns: TODO

        """
        n = tf.size(self.tf_y)
        zeros_n = tf.zeros([n, 1])
        sigma2 = tf.exp(params_log[0,0])
        phi = tf.exp(params_log[1,0])

        Sigma_gp = tf_cov_exp(self.tf_dis, sigma2, phi, 0.0)
        Sigma_marginal = self.prior_c_sigma2 + Sigma_gp
        Sigma_z = Sigma_marginal + tf.eye(n)
        posterior_prob_log = dmvnorm(self.tf_y, zeros_n, Sigma_z) + \
                distr.Normal(tf.log(1.0), 0.4).log_prob(params_log[0,0]) + \
                distr.Normal(tf.log(0.08), 0.4).log_prob(params_log[1,0])
        return posterior_prob_log


    def update(self, Sigma_proposal_chol):
        """TODO: Docstring for update.

        :x: TODO
        :returns: TODO

        """
        n = tf.size(self.tf_y)
        zeros_n = tf.zeros([n, 1])
        params_log_aux = self.params_log + \
                tf.matmul(Sigma_proposal_chol,
                        tf.distributions.Normal(0.0, 1.0).sample(tf.shape(self.params_log)))

        params_log_aux_posterior = self.posterior(params_log_aux)
        acceptance_prob_log = params_log_aux_posterior - self.params_log_posterior

        uniform_log = tf.log(tf.distributions.Uniform().sample())
        params_log_new = tf.cond(tf.greater(acceptance_prob_log,uniform_log),\
                lambda: params_log_aux, lambda: self.params_log)
        params_log_posterior_new = tf.cond(tf.greater(acceptance_prob_log,uniform_log),\
                lambda: params_log_aux_posterior, lambda: self.params_log_posterior)

        op1 = tf.assign(self.params_log, params_log_new)
        op2 = tf.assign(self.params_log_posterior, params_log_posterior_new)
        return op1, op2
        # return params_log_new

    def sample(self, Sigma_proposal, niter):
        """TODO: Docstring for sample.
        :returns: TODO

        """
        # Allocate memory for samples
        samples = np.zeros((niter, 2))

        # Constants
        Sigma_proposal = tf.constant(Sigma_proposal, dtype = tf.float32)
        Sigma_proposal_chol = tf.cholesky(Sigma_proposal)

        # Log posterior of initial parameters
        aux = initialize(self.posterior(self.params_log), self.feed_dict)
        self.params_log_posterior = tf.Variable(aux, dtype = tf.float32)

        # Metropolis Hastings sampler
        updater = self.update(Sigma_proposal_chol)

        # Iterate trough MH sampler
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(niter):
            params_new, params_log_posterior_new = sess.run(updater, self.feed_dict)
            samples[i, :] = np.reshape(np.exp(params_new), 2)
            # print(sess.run(params_log_posterior, feed_dict))

        # sess.run(updater, feed_dict)
        return samples

    def callpost(self):
        """TODO: Docstring for callpost.
        :returns: TODO

        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess.run(self.posterior(self.params_log), self.feed_dict)




def initialize(tensor, dictionary):
    """TODO: Docstring for initialize.
    :tensor: TODO
    :returns: TODO

    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(tensor, dictionary)
    return output

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

def test(params, dist, y):
    """TODO: Docstring for test.

    :x: TODO
    :returns: TODO

    """
    # Data
    tf_dis = tf.placeholder(dtype = tf.float32)
    tf_y = tf.placeholder(dtype = tf.float32)
    feed_dict = {
            tf_dis: dist,
            tf_y: y
            }

    prior_c_sigma2 = tf.constant(1.0)
    params_log = tf.Variable(params, dtype = tf.float32)

    sess_init = tf.Session()
    sess_init.run(tf.global_variables_initializer())
    plop = sess_init.run(posterior(params_log, tf_dis, tf_y, prior_c_sigma2),
            feed_dict)
    return plop

