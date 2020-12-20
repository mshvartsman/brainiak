from sklearn.base import BaseEstimator
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
import logging

__all__ = [
    "FastTFA",
]

logger = logging.getLogger(__name__)


@tf.function()
def get_factormat(R, centers, widths):
    rdists = tf.reduce_sum(tf.square(R), axis=-1)
    cdists = tf.reduce_sum(tf.square(centers), axis=-1)

    distmat = cdists[:, None] - 2.0 * \
        (centers @ tf.transpose(R)) + rdists[None, :]

    return tf.math.exp(-distmat / widths)


class FastTFA(BaseEstimator):

    def __init__(
            self,
            k=5):
        self.k = k

    @tf.function()
    def marginal_tfa_loss(self, theta, X, R, s):
        """ marginal log likelihood of TFA generative model,
        omitting constant terms and priors for now
        """
        centers = tf.reshape(theta[:(3*self.k)], (self.k, 3))
        widths = tf.reshape(theta[(3*self.k):], (self.k, 1))
        F = get_factormat(R, centers, widths)

        cholesky_term = tf.linalg.cholesky(
            F @ tf.transpose(F) / s + tf.eye(self.k, dtype="float64"))
        logdet_term = 2 * \
            tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky_term))
                          ) + 2 * self.sz * tf.math.log(tf.math.sqrt(s))
        solve_term = tf.linalg.trace(X @ (tf.eye(self.sz, dtype="float64") / s - tf.transpose(
            F) @ tf.linalg.cholesky_solve(cholesky_term, F) / s**2) @ tf.transpose(X))

        return logdet_term + solve_term

    @tf.function()
    def _val_and_grad(self, theta, X, R, s):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            loss = self.marginal_tfa_loss(theta, X, R, s)

        grad = tape.gradient(loss, theta)
        return loss, grad

    def fit(self, X, R, n_iter=1, subsamp_size_v=None, subsamp_size_t=None):
        """ Topographical Factor Analysis (TFA)[Manning2014]

        Parameters
        ----------
        X : 2D array, in shape [TRs, voxels]
            The fMRI data of one subject

        R : 2D array, in shape [n_voxel, n_dim]
            The voxel coordinate matrix of fMRI data

        """
        self.t, self.v = X.shape

        init_centers = np.random.randint(
            np.min(R), high=np.max(R), size=(self.k, 3)).astype("float64")
        init_widths = np.abs(np.random.normal(
            loc=0, scale=np.std(R)**2, size=(self.k, 1)))

        theta0 = np.concatenate(
            [init_centers.flatten(), init_widths.flatten()])

        lb = [np.min(R)] * 3 * self.k + [1e-5] * self.k
        ub = [np.max(R)] * 3 * self.k + [np.std(R)**2] * self.k
        bounds = list(zip(lb, ub))

        s = tf.math.reduce_std(X)**2
        subsamp_size_v = subsamp_size_v or self.v
        subsamp_size_t = subsamp_size_t or self.t
        self.sz = subsamp_size_v
        n_iter = n_iter if subsamp_size_v else 1
        xconst = tf.constant(X)
        rconst = tf.constant(R)

        for i in range(n_iter):
            if subsamp_size_v < self.v:
                subsamp_v = np.random.choice(
                    self.v, subsamp_size_v, replace=False)
                subsamp_t = np.random.choice(
                    self.t, subsamp_size_t, replace=False)
                xconst = tf.constant(X[subsamp_t, :][:, subsamp_v])
                rconst = tf.constant(R[subsamp_v, :])
                self.sz = subsamp_size_v

            def val_and_grad(theta):
                theta_ = tf.constant(theta)
                loss, grad = self._val_and_grad(theta_, xconst, rconst, s)
                return loss.numpy(), grad.numpy()
            result = minimize(fun=val_and_grad, x0=theta0,
                              method="L-BFGS-B", jac=True, bounds=bounds)
            theta0 = result.x

        self.centers_ = result.x[:(3*self.k)].reshape(self.k, 3)
        self.widths_ = result.x[(3*self.k):].reshape(self.k, 1)
        self.F_ = get_factormat(R, self.centers_, self.widths_).numpy()
        self.W_ = np.linalg.solve(self.F_ @ self.F_.T, self.F_ @ X.T).T
        return self
