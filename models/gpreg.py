
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

        # Data as dictionary for tensorflow
        self.tf_y = tf.placeholder(dtype = tf.float32)
        self.tf_dis = tf.placeholder(dtype = tf.float32)
        self.feed_dict = {
                self.tf_dis: dist,
                self.tf_y: y
                }


    def sample(self, Sigma_proposal, niter, sampler = "rwmh"):
        """TODO: Docstring for sample.
        :returns: TODO

        """

        # Prior hyperparameters
        self.prior_c_sigma2 = tf.constant(1.0)

        # Constants
        Sigma_proposal = tf.constant(Sigma_proposal, dtype = tf.float32)
        Sigma_proposal_chol = tf.cholesky(Sigma_proposal)

        # Log posterior of initial parameters
        initial_logpost = eval(self.logposterior(self.params_log), self.feed_dict)
        self.params_logpost = tf.Variable(initial_logpost, dtype = tf.float32)

        # Gradient log posterior of initial parameters and mala sampler
        if sampler == "mala":
            initial_logpost_grad = \
                    eval(tf.gradients(self.logposterior(self.params_log),
                        self.params_log)[0], self.feed_dict)
            print(initial_logpost_grad)
            self.params_logpost_grad = tf.Variable(initial_logpost_grad)
            updater = self.update_mala(Sigma_proposal_chol)

        # Metropolis Hastings sampler
        if sampler == "rwmh":
            updater = self.update_rwmh(Sigma_proposal_chol)

        # Allocate memory for samples
        samples = np.zeros((niter, 2))

        # Iterate trough sampler and save samples
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(niter):
                params_new = sess.run(updater, self.feed_dict)
                samples[i, :] = np.reshape(np.exp(params_new[0]), 2)

        return samples


    def logposterior(self, params_log):
        """TODO: Docstring for posterior.

        :params_log: TODO
        :tf_dis: TODO
        :tf: TODO
        :returns: TODO

        """
        zeros_n = tf.zeros([self.n, 1])
        sigma2 = tf.exp(params_log[0,0])
        phi = tf.exp(params_log[1,0])

        Sigma_gp = tf_cov_exp(self.tf_dis, sigma2, phi, 0.0)
        Sigma_marginal = self.prior_c_sigma2 + Sigma_gp
        Sigma_z = Sigma_marginal + tf.eye(self.n)
        posterior_prob_log = dmvnorm(self.tf_y, zeros_n, Sigma_z) + \
                distr.Normal(tf.log(1.0), 0.4).log_prob(params_log[0,0]) + \
                distr.Normal(tf.log(0.08), 0.4).log_prob(params_log[1,0])
        return posterior_prob_log


    def update_rwmh(self, L_proposal):
        """TODO: Docstring for update_rwmh.

        :L_proposal: Lower traingular cholesky decomposition of the covariance of the
        proposal distribution
        :returns: TODO

        """
        zeros_n = tf.zeros([self.n, 1])
        dims = tf.shape(self.params_log)

        candidate = self.params_log
        candidate += tf.matmul(L_proposal, distr.Normal(0.0, 1.0).sample(dims))
        cand_logpost = self.logposterior(candidate)
        logprob = cand_logpost - self.params_logpost

        log_unif = tf.log(distr.Uniform().sample())
        new, new_logpost = tf.cond(
                tf.greater(logprob, log_unif),
                lambda: (candidate, cand_logpost),
                lambda: (self.params_log, self.params_logpost)
                )

        op_param = tf.assign(self.params_log, new)
        op_logpost = tf.assign(self.params_logpost, new_logpost)
        return op_param, op_logpost


    def update_mala(self, L_proposal):
        """TODO: Docstring for update_rwmh.

        :L_proposal: Lower traingular cholesky decomposition of the covariance of the
        proposal distribution
        :returns: TODO

        """
        zeros_n = tf.zeros([self.n, 1])
        dims = tf.shape(self.params_log)

        L_inv = tf.matrix_inverse(L_proposal)

        candidate = self.params_log + 0.03 * self.params_logpost_grad
        candidate += tf.matmul(L_proposal, distr.Normal(0.0, 1.0).sample(dims))
        cand_logpost = self.logposterior(candidate)
        cand_logpost_grad = tf.gradients(self.logposterior(candidate), candidate)[0]

        center_current =  self.params_log - candidate - 0.03 * cand_logpost_grad
        center_cand =  candidate - self.params_log - 0.03 * self.params_logpost_grad

        logprob = cand_logpost - self.params_logpost
        logprob -= 0.5 * tf.reduce_sum(tf.square(tf.matmul(L_inv, center_current)))
        logprob += 0.5 * tf.reduce_sum(tf.square(tf.matmul(L_inv, center_cand)))

        log_unif = tf.log(distr.Uniform().sample())
        new, new_logpost, new_logpost_grad = tf.cond(
                tf.greater(logprob, log_unif),
                lambda: (candidate, cand_logpost, cand_logpost_grad),
                lambda: (self.params_log, self.params_logpost,
                    self.params_logpost_grad)
                )

        op_param = tf.assign(self.params_log, new)
        op_logpost = tf.assign(self.params_logpost, new_logpost)
        op_logpost_grad = tf.assign(self.params_logpost_grad, new_logpost_grad)
        return op_param, op_logpost, new_logpost_grad


    def callpost(self):
        """TODO: Docstring for callpost.
        :returns: TODO

        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        return sess.run(self.posterior(self.params_log), self.feed_dict)




def eval(tensor, dictionary):
    """TODO: Docstring for eval.
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

