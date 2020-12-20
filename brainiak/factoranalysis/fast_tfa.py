from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
import math
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

    def init_centers_widths(self, R):
        """Initialize prior of centers and widths

        Returns
        -------

        centers : 2D array, with shape [K, n_dim]
            Prior of factors' centers.

        widths : 1D array, with shape [K, 1]
            Prior of factors' widths.

        """

        max_sigma = 2.0 * math.pow(np.nanmax(np.std(R, axis=0)), 2)
        kmeans = KMeans(
            init='k-means++',
            n_clusters=self.k,
            n_init=10,
            random_state=100)
        kmeans.fit(R)
        centers = kmeans.cluster_centers_
        widths = max_sigma * np.ones((self.k, 1)) + np.random.normal(self.k)
        return centers, widths

    @tf.function()
    def marginal_tfa_loss(self, theta, X, R):
        """ marginal log likelihood of TFA generative model,
        omitting constant terms and priors for now
        """
        centers = tf.reshape(theta[:(3*self.k)], (self.k, 3))
        widths = tf.reshape(theta[(3*self.k):], (self.k, 1))
        F = get_factormat(R, centers, widths)
        s = tf.math.reduce_std(X)**2

        cholesky_term = tf.linalg.cholesky(
            F @ tf.transpose(F) / s + tf.eye(self.k, dtype="float64"))
        logdet_term = 2 * \
            tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky_term))
                          ) + 2 * self.v * tf.math.log(tf.math.sqrt(s))
        solve_term = tf.linalg.trace(tf.transpose(X) @ X / s - tf.transpose(
            F) @ tf.linalg.cholesky_solve(cholesky_term, F) / s**2 @ tf.transpose(X) @ X)

        return logdet_term + solve_term

    @tf.function()
    def _val_and_grad(self, theta, X, R):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            loss = self.marginal_tfa_loss(theta, X, R)

        grad = tape.gradient(loss, theta)
        return loss, grad

    def fit(self, X, R):
        """ Topographical Factor Analysis (TFA)[Manning2014]

        Parameters
        ----------
        X : 2D array, in shape [TRs, voxels]
            The fMRI data of one subject

        R : 2D array, in shape [n_voxel, n_dim]
            The voxel coordinate matrix of fMRI data

        """
        self.t, self.v = X.shape

        init_centers, init_widths = self.init_centers_widths(R)

        theta0 = np.concatenate(
            [init_centers.flatten(), init_widths.flatten()])

        lb = [np.min(R)] * 3 * self.k + [1e-5] * self.k
        ub = [np.max(R)] * 3 * self.k + [np.std(R)**2] * self.k
        bounds = list(zip(lb, ub))

        def val_and_grad(theta):
            theta_ = tf.constant(theta)
            loss, grad = self._val_and_grad(theta_, X, R)
            return loss.numpy(), grad.numpy()

        result = minimize(fun=val_and_grad, x0=theta0,
                          method="L-BFGS-B", jac=True, bounds=bounds)

        self.centers_ = result.x[:(3*self.k)].reshape(self.k, 3)
        self.widths_ = result.x[(3*self.k):].reshape(self.k, 1)
        self.F_ = get_factormat(R, self.centers_, self.widths_).numpy()
        self.W_ = np.linalg.solve(self.F_ @ self.F_.T, self.F_ @ X.T).T
        return self
