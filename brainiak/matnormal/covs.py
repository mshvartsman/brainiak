import tensorflow as tf
import numpy as np
import abc
import scipy.linalg
import scipy.sparse
from tensorflow.contrib.distributions import InverseGamma, WishartCholesky
from brainiak.matnormal.utils import x_tx, xx_t
from brainiak.utils.kronecker_solvers import tf_solve_lower_triangular_kron,\
                          tf_solve_upper_triangular_kron, \
                          tf_solve_lower_triangular_masked_kron, \
                          tf_solve_upper_triangular_masked_kron

__all__ = ['CovBase',
           'CovIdentity',
           'CovAR1',
           'CovIsotropic',
           'CovDiagonal',
           'CovDiagonalGammaPrior',
           'CovUnconstrainedCholesky',
           'CovUnconstrainedCholeskyWishartReg',
           'CovUnconstrainedInvCholesky',
           'CovKroneckerFactored']


class CovBase(object):
    """Base metaclass for noise covariances
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, size):
        self.size = size
        
        # Log-likelihood of this covariance (useful for regularization)
        self.logp = tf.constant(0, dtype=tf.float64)

    @abc.abstractmethod
    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        pass

    def logdet(self):
        """ log|Sigma|
        """
        pass

    @abc.abstractmethod
    def solve(self, X):
        """Given this covariance and some X, compute :math:`Sigma^{-1} * x`
        """
        pass

    @property
    def _prec(self):
        """Expose the precision explicitly (mostly for testing / 
        visualization)
        """
        return self.solve(tf.eye(self.size, dtype=tf.float64))

    @property
    def _cov(self):
        """Expose the covariance explicitly (mostly for testing / 
        visualization)
        """
        return tf.linalg.inv(self._prec)


# class CovTFWrap(CovBase):
#     """ thin wrapper around a TF tensor
#     """
#     def __init__(self, Sigma):

#         self.L = tf.cholesky(Sigma)
#         self.logdet = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))

#     def get_optimize_vars(self):
#         """ Returns a list of tf variables that need to get optimized to fit
#             this covariance
#         """
#         return []

#     def solve(self, X):
#         """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
#         """
#         return tf.cholesky_solve(self.L, X)


class CovIdentity(CovBase):
    """Identity noise covariance.
    """
    def __init__(self, size):
        super(CovIdentity, self).__init__(size)
        self.logdet = tf.constant(0.0, 'float64')

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to
            fit this covariance
        """
        return []

    def solve(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return X


class CovAR1(CovBase):
    """AR1 covariance
    """
    def __init__(self, size, rho=None, sigma=None, scan_onsets=None):

        super(CovAR1, self).__init__(size)

        # Similar to BRSA trick I think
        if scan_onsets is None:
            self.run_sizes = [size]
            self.offdiag_template = tf.constant(scipy.linalg.toeplitz(np.r_[0,
                                                1, np.zeros(size-2)]),
                                                dtype=tf.float64)
            self.diag_template = tf.constant(np.diag(np.r_[0,
                                                           np.ones(size-2),
                                                           0]))
        else:
            self.run_sizes = np.ediff1d(np.r_[scan_onsets, size])
            sub_offdiags = [scipy.linalg.toeplitz(np.r_[0, 1, np.zeros(r-2)])
                            for r in self.run_sizes]
            self.offdiag_template = tf.constant(scipy.sparse.
                                                block_diag(sub_offdiags)
                                                .toarray())
            subdiags = [np.diag(np.r_[0, np.ones(r-2), 0])
                        for r in self.run_sizes]
            self.diag_template = tf.constant(scipy.sparse.
                                             block_diag(subdiags)
                                             .toarray())

        self._identity_mat = tf.constant(np.eye(size))

        if sigma is None:
            self.log_sigma = tf.Variable(tf.random_normal([1],
                                         dtype=tf.float64), name="sigma")
        else:
            self.log_sigma = tf.Variable(np.log(sigma), name="sigma")

        if rho is None:
            self.rho_unc = tf.Variable(tf.random_normal([1], dtype=tf.float64),
                                       name="rho")
        else:
            self.rho_unc = tf.Variable(np.log(rho), name="rho")

        # make logdet, first unconstrain rho and sigma
        rho = 2 * tf.sigmoid(self.rho_unc) - 1
        sigma = tf.exp(self.log_sigma)
        # now compute logdet
        self.logdet = tf.reduce_sum(2 * tf.constant(self.run_sizes,
                                    dtype=tf.float64) *
                                    tf.log(sigma) - tf.log(1 - tf.square(rho)))

        # precompute sigma_inv op
        # Unlike BRSA we assume stationarity within block so no special case
        # for first/last element of a block. This makes constructing this
        # matrix easier.
        # reprsimil.BRSA says (I - rho1 * D + rho1**2 * F) / sigma**2

        rho = 2 * tf.sigmoid(self.rho_unc) - 1
        sigma = tf.exp(self.log_sigma)
        self.Sigma_inv = (self._identity_mat - rho * self.offdiag_template + rho**2 *
                self.diag_template) / tf.square(sigma)


    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to
            fit this covariance
        """
        return [self.rho_unc, self.log_sigma]

    def solve(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return tf.matmul(self.Sigma_inv, X)


class CovIsotropic(CovBase):
    """Scaled identity (isotropic) noise covariance.
    """

    def __init__(self, size, sigma=None):
        super(CovIsotropic, self).__init__(size)
        if sigma is None:
            self.log_sigma = tf.Variable(tf.random_normal([1],
                                         dtype=tf.float64), name="sigma")
        else:
            self.log_sigma = tf.Variable(np.log(sigma), name="sigma")

        self.logdet = self.size * self.log_sigma

        self.sigma = tf.exp(self.log_sigma)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.log_sigma]


    def solve(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return X / self.sigma


class CovDiagonal(CovBase):
    """Uncorrelated (diagonal) noise covariance
    """
    def __init__(self, size, sigma=None):
        super(CovDiagonal, self).__init__(size)
        if sigma is None:
            self.logprec = tf.Variable(tf.random_normal([size],
                                       dtype=tf.float64), name="precisions")
        else:
            self.logprec = tf.Variable(np.log(1/sigma), name="log-precisions")

        self.logdet = -tf.reduce_sum(self.logprec)
        self.prec = tf.exp(self.logprec)
        self.prec_dimaugmented = tf.expand_dims(self.prec, -1)

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.logprec]

    def solve(self, X):
        """Given this Sigma and some X, compute :math:`Sigma^{-1} * x`
        """
        return tf.multiply(self.prec_dimaugmented, X)


class CovDiagonalGammaPrior(CovDiagonal):
    """Uncorrelated (diagonal) noise covariance
    """
    def __init__(self, size, sigma=None, alpha=1.5, beta=1e-10):
        super(CovDiagonalGammaPrior, self).__init__(size, sigma)

        self.ig = InverseGamma(concentration=tf.constant(alpha,
                                                         dtype=tf.float64),
                               rate=tf.constant(beta, dtype=tf.float64))

        self.logp = tf.reduce_sum(self.ig.log_prob(self.prec))


class CovUnconstrainedCholesky(CovBase):
    """Unconstrained noise covariance parameterized in terms of its cholesky
    """

    def __init__(self, size=None, Sigma=None):
        
        if size is None and Sigma is None:
            raise RuntimeError("Must pass either Sigma or size")

        if size is not None and Sigma is not None:
            raise RuntimeError("Must pass either Sigma or size but not both")

        if Sigma is not None:
            size = Sigma.shape[0]

        super(CovUnconstrainedCholesky, self).__init__(size)
        
        if Sigma is None:
            self.L_full = tf.Variable(tf.random_normal([size, size],
                                      dtype=tf.float64),
                                      name="L_full", dtype="float64")
        
        else:
            # in order to respect the Sigma we got passed in, we log the diag
            # which we will later exp. a little ugly but this
            # is a rare use case
            L = np.linalg.cholesky(Sigma)
            L[np.diag_indices_from(L)] = np.log(np.diag(L))
            self.L_full = tf.Variable(L, name="L_full",
                                      dtype="float64")


        # Zero out triu of L_full to get cholesky L.
        # This seems dumb but TF is smart enough to set the gradient to zero
        # for those elements, and the alternative (fill_lower_triangular from
        # contrib.distributions) is inefficient and recommends not doing the
        # packing (for now).
        # Also: to make the parameterization unique we exp the diagonal so
        # it's positive.
      
        L_indeterminate = tf.matrix_band_part(self.L_full, -1, 0)
        self.L = tf.matrix_set_diag(L_indeterminate,
                                  tf.exp(tf.matrix_diag_part(L_indeterminate)))

        self.logdet = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L)))

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
             this covariance
        """
        return [self.L_full]

    def solve(self, X):
        """
        Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
        cholesky solve
        """
        return tf.cholesky_solve(self.L, X)

class CovUnconstrainedCholeskyWishartReg(CovUnconstrainedCholesky):
    """Unconstrained noise covariance parameterized in terms of its
       cholesky factor.
       Regularized using the trick from Chung et al. 2015 such that as the
       covariance approaches singularity, the likelihood goes to 0.
    """

    def __init__(self, size, Sigma=None):
        super(CovUnconstrainedCholeskyWishartReg, self).__init__(size)
        self.wishartReg = WishartCholesky(df=tf.constant(size+2,
                                                         dtype=tf.float64),
                                          scale=tf.constant(1e5 * np.eye(size),
                                          dtype=tf.float64))

        Sigma = xx_t(self.L)
        self.logp = self.wishartReg.log_prob(Sigma)

class CovUnconstrainedInvCholesky(CovBase):
    """Unconstrained noise covariance parameterized
       in terms of its precision cholesky. Use this over the 
       regular cholesky unless you have a good reason not to, since
       you save a solve on every step. 
    """

    def __init__(self, size, invSigma=None):
        super(CovUnconstrainedInvCholesky, self).__init__(size)
        
        if invSigma is None:
            self.Linv_full = tf.Variable(tf.random_normal([size, size],
                                         dtype=tf.float64), name="Linv_full")
        else:
            self.Linv_full = tf.Variable(np.linalg.cholesky(invSigma),
                                         name="Linv_full")

        # Zero out triu of L_full to get cholesky L.
        # This seems dumb but TF is smart enough to set the gradient to zero
        # for those elements, and the alternative (fill_lower_triangular from
        # contrib.distributions) is inefficient and recommends not doing the
        # packing (for now).
        # Also: to make the parameterization unique we log the diagonal so
        # it's positive.
        L_indeterminate = tf.matrix_band_part(self.Linv_full, -1, 0)
        self.Linv = tf.matrix_set_diag(L_indeterminate,
                                  tf.exp(tf.matrix_diag_part(L_indeterminate)))
        self.logdet = -2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.Linv)))

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized to fit
            this covariance
        """
        return [self.Linv_full]

    def solve(self, X):
        """
        Given this Sigma and some X, compute :math:`Sigma^{-1} * x` using
        matmul (since we're parameterized by L_inv)
        """
        return tf.matmul(x_tx(self.Linv), X)


class CovKroneckerFactored(CovBase):
    """ Kronecker product noise covariance parameterized in terms
    of its component cholesky factors
    """

    def __init__(self, sizes, Sigmas=None, mask=None):
        """Initialize the kronecker factored covariance object.

        Arguments
        ---------
        sizes : list
            List of dimensions (int) of the factors
            E.g. ``sizes = [2, 3]`` will create two factors of
            sizes 2x2 and 3x3 giving us a 6x6 dimensional covariance
        Sigmas : list (default : None)
            Initial guess for the covariances. List of positive definite
            covariance matrices the same sizes as sizes.
        mask : int array (default : None)
            1-D tensor with length equal to product of sizes with 1 for
            valid elements and 0 for don't care

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If sizes is not a list
        """
        if not isinstance(sizes, list):
            raise TypeError('sizes is not a list')

        self.sizes = sizes
        self.nfactors = len(sizes)
        self.size = np.prod(np.array(sizes), dtype=np.int32)

        if Sigmas is None:
            self.L_full = [tf.Variable(tf.random_normal([sizes[i], sizes[i]],
                           dtype=tf.float64), name="L"+str(i)+"_full")
                           for i in range(self.nfactors)]
        else:
            self.L_full = [tf.Variable(np.linalg.cholesky(Sigmas[i]),
                           name="L"+str(i)+"_full")
                           for i in range(self.nfactors)]
        self.mask = mask

        # make a list of choleskys
        L_indeterminate = [tf.matrix_band_part(mat, -1, 0)
                           for mat in self.L_full]
        self.L = [tf.matrix_set_diag(mat, tf.exp(tf.matrix_diag_part(mat)))
                for mat in L_indeterminate]

        self.logdet = self._make_logdet()

    def get_optimize_vars(self):
        """ Returns a list of tf variables that need to get optimized
            to fit this covariance
        """
        return self.L_full

    def _make_logdet(self):
        """ log|Sigma| using the diagonals of the cholesky factors.
        """
        if self.mask is None:
            n_list = tf.stack([tf.to_double(tf.shape(mat)[0])
                               for mat in self.L])
            n_prod = tf.reduce_prod(n_list)
            logdet = tf.stack([tf.reduce_sum(tf.log(tf.diag_part(mat)))
                               for mat in self.L])
            logdetfinal = tf.reduce_sum((logdet*n_prod)/n_list)
        else:
            n_list = [tf.shape(mat)[0] for mat in self.L]
            mask_reshaped = tf.reshape(self.mask, n_list)
            logdet = 0.0
            for i in range(self.nfactors):
                indices = list(range(self.nfactors))
                indices.remove(i)
                logdet += tf.log(tf.diag_part(self.L[i])) *\
                    tf.to_double(tf.reduce_sum(mask_reshaped, indices))
            logdetfinal = tf.reduce_sum(logdet)
        return (2.0*logdetfinal)

    def solve(self, X):
        """ Given this Sigma and some X, compute Sigma^{-1} * x using
        traingular solves with the cholesky factors.
        Do 2 triangular solves - L L^T x = y as L z = y and L^T x = z
        """
        if self.mask is None:
            z = tf_solve_lower_triangular_kron(self.L, X)
            x = tf_solve_upper_triangular_kron(self.L, z)
        else:
            z = tf_solve_lower_triangular_masked_kron(self.L, X, self.mask)
            x = tf_solve_upper_triangular_masked_kron(self.L, z, self.mask)
        return x
