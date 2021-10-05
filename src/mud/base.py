import numpy as np
from numpy.linalg import inv
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


class LinearGaussianProblem(object):
    """Inverse Problems with Linear/Affine Maps with Gaussian distributions.

    """

    def __init__(self, A, b, y=None, mean=None, cov=None, data_cov=None):

        self.A = A
        self.b = b

        # Initialize to defaults
        n_samples, dim_input = self.A.shape
        self.data_cov = np.eye(n_samples) if data_cov is None else data_cov
        self.cov = np.eye(dim_input) if cov is None else cov
        self.mean = np.zeros((dim_input, 1)) if mean is None else mean.reshape(-1,1)
        self.b = np.zeros((n_samples, 1)) if b is None else b.reshape(-1,1)
        self.y = np.zeros(n_samples) if y is None else y.reshape(-1,1)

        n_data, n_targets = y.shape

        if n_samples != n_data:
            raise ValueError(
                "Number of samples in X and y does not correspond:"
                " %d != %d" % (n_samples, n_data)
            )

        # Compute residual
        self.z = self.y - self.b - self.A @ self.mean

        # Update to residual we initialize to None. Evaluate in fit function
        self.update = None

    def fit(self):
        raise NotImplementedError


    def estimate(self):
        raise NotImplementedError


class LinearMUD(LinearGaussianProblem):
    """

    """

    def __init__(self, A, b, y=None, mean=None, cov=None, data_cov=None):
        super.__init__(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov)
        self.mud_point = None


    def updated_cov(X, init_cov=None, data_cov=None):
        """
        We start with the posterior covariance from ridge regression
        Our matrix R = init_cov^(-1) - X.T @ pred_cov^(-1) @ X
        replaces the init_cov from the posterior covariance equation.
        Simplifying, this is given as the following, which is not used
        due to issues of numerical stability (a lot of inverse operations).

        up_cov = (X.T @ np.linalg.inv(data_cov) @ X + R )^(-1)
        up_cov = np.linalg.inv(\
            X.T@(np.linalg.inv(data_cov) - inv_pred_cov)@X + \
            np.linalg.inv(init_cov) )

        We return the updated covariance using a form of it derived
        which applies Hua's identity in order to use Woodbury's identity.

        >>> updated_cov(np.eye(2))
        array([[1., 0.],
               [0., 1.]])
        >>> updated_cov(np.eye(2)*2)
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        >>> updated_cov(np.eye(3)[:, :2]*2, data_cov=np.eye(3))
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        >>> updated_cov(np.eye(3)[:, :2]*2, init_cov=np.eye(2))
        array([[0.25, 0.  ],
               [0.  , 0.25]])
        """
        if init_cov is None:
            init_cov = np.eye(X.shape[1])
        else:
            assert X.shape[1] == init_cov.shape[1]

        if data_cov is None:
            data_cov = np.eye(X.shape[0])
        else:
            assert X.shape[0] == data_cov.shape[1]

        pred_cov = X @ init_cov @ X.T
        inv_pred_cov = np.linalg.pinv(pred_cov)
        # pinv b/c inv unstable for rank-deficient A

        # Form derived via Hua's identity + Woodbury
        K = init_cov @ X.T @ inv_pred_cov
        up_cov = init_cov - K @ (pred_cov - data_cov) @ K.T

        return up_cov


    def mud_point(self, method='default'):
        """
        Compute MUD Point

        """
        if method=='default':
            inv_pred_cov = np.linalg.pinv(self.A @ self.cov @ self.A.T)
            self.update = self.cov @ self.A.T @ inv_pred_cov
        else:
            up_cov = updated_cov(X=A, init_cov=cov, data_cov=data_cov)
            self.update = up_cov @ A.T @ np.linalg.inv(data_cov)

        self.mud = self.mean + self.update @ self.z

        return self.mud


    def estimate(self):
        if self.mud_point is None:
            return self.mud_point()


class LinearBayes(LinearGaussianProblem):
    """

    """

    def __init__(self, A, b, y=None, mean=None, cov=None, data_cov=None, w=1):
        super.__init__(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov)
        self.w = w
        self.map_point = None


    def map_point(self):
        dc_i = inv(self.data_cov)
        post_cov = inv(self.A.T @ dc_i @ self.A + self.w * inv(self.cov))
        self.update = post_cov @ A.T @ dc_i
        self.map_point = mean + update @ z


    def estimate(self):
        if self.map_point is None:
            return self.map_point()



class DensityProblem(object):
    """Data-Consistent Inverse Problem for parameter identification

    Data-Consistent inversion is a way to infer most likely model paremeters
    using observed data and predicted data from the model.

    Parameters
    ----------
    X : ndarray
        2D array containing parameter samples from an initial distribution. Rows
        represent each sample while columns represent parameter values.
    y : ndarray
        array containing push-forward values of paramters samples through the
        forward model. These samples will form the `predicted distribution`.
    domain : array_like, optional
        2D Array containing ranges of each paramter value in the parameter
        space. Note that the number of rows must equal the number of parameters,
        and the number of columns must always be two, for min/max range.

    Example Usage
    -------------
    Generate test 1-D parameter estimation problem. Model to produce predicted
    data is the identity map and observed signal comes from true value plus
    some random gaussian nose:

    >>> from mud.base import DensityProblem
    >>> from mud.funs import wme
    >>> import numpy as np
    >>> def test_wme_data(domain, num_samples, num_obs, noise, true):
    ...     # Parameter samples come from uniform distribution over domain
    ...     X = np.random.uniform(domain[0], domain[1], [num_samples,1])
    ...     # Identity map model, so predicted values same as param values.
    ...     predicted = np.repeat(X, num_obs, 1)
    ...     # Take Observed data from true value plus random gaussian noise
    ...     observed = np.ones(num_obs)*true + np.random.randn(num_obs)*noise
    ...     # Compute weighted mean error between predicted and observed values
    ...     y = wme(predicted, observed)
    ...     # Build density problem, with wme values as the model data
    ...     return DensityProblem(X, y, [domain])

    Set up well-posed problem:
    >>> D = test_wme_data([0,1], 1000, 50, 0.05, 0.5)

    Estimate mud_point -> Note since WME map used, observed implied to be the
    standard normal distribution and does not have to be set explicitly from
    observed data set.
    >>> np.round(D.mud_point()[0],1)
    0.5

    Expecation value of r, ratio of observed and predicted distribution, should
    be near 1 if predictabiltiy assumption is satisfied.
    >>> np.round(D.exp_r(),0)
    1

    Set up ill-posed problem -> Searching out of range of true value
    >>> D = test_wme_data([0.6, 1], 1000, 50, 0.05, 0.5)

    Mud point will be close as we can get within the range we are searching for
    >>> np.round(D.mud_point()[0],1)
    0.6

    Expectation of r is close to zero since predictability assumption violated.
    >>> np.round(D.exp_r(),1)
    0.0

    """

    def __init__(self, X, y, domain=None):
        self.X = X
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.domain = domain
        self._r = None
        self._up = None
        self._in = None
        self._pr = None
        self._ob = None


    def set_observed(self, distribution=dist.norm()):
        """Set distribution for the observed data.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=scipy.stats.norm()
            scipy.stats continuous distribution like object representing the
            likelihood of observed data. Defaults to a standard normal
            distribution N(0,1).

        """
        self._ob = distribution.pdf(self.y).prod(axis=1)


    def set_initial(self, distribution=None):
        """Set initial distribution of model parameter values.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, optional
            scipy.stats continuous distribution object from where initial
            parameter samples were drawn from. If non provided, then a uniform
            distribution over domain of density problem is assumed. If no domain
            is specified for density, then a standard normal distribution is
            assumed.

        """
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        initial_dist = distribution
        self._in = initial_dist.pdf(self.X).prod(axis=1)
        self._up = None
        self._pr = None


    def set_predicted(self, distribution=None,
            bw_method=None, weights=None, **kwargs):
        """Sets the predicted distribution from predicted data `y`.

        Parameters
        ----------
        distribution : scipy.stats.rv_continuous, default=None
            A scipy.stats continuous probability distribution. If non specified,
            then the distribution for the predicted data is computed using
            gaussina kernel density estimation.
        bw_method : str, scalar, or callable, optional
            Bandwidth method to use in gaussian kernel density estimation.
        weights : array_like, optional
            Weights to apply to predicted samples `y` in gaussian kernel density
            estimation.
        **kwargs : dict, optional
            If a distribution is passed, then any extra keyword arguments will
            be passed to the pdf() method as keyword arguments.

        Returns
        -------
        """
        if distribution is None:
            distribution = gkde(self.y.T, bw_method=bw_method, weights=weights)
            pred_pdf = distribution.pdf(self.y.T).T
        else:
            pred_pdf = distribution.pdf(self.y, **kwargs)
        self._pr = pred_pdf
        self._up = None


    def fit(self):
        """Update initial distribution using ratio of observed and predicted.

        Applies [] to compute the updated distribution using the ratio of the
        observed to the predicted multiplied by the initial according to the
        data-consistent framework. Note that if initail, predicted, and observed
        distributiosn have not been set before running this method, they will
        be run with default values. To set specific predicted, observed, or
        initial distributions use the `set_` methods.

        Parameteres
        -----------

        Returns
        -----------

        """
        if self._in is None:
            self.set_initial()
        if self._pr is None:
            self.set_predicted()
        if self._ob is None:
            self.set_observed()

        # Store ratio of observed/predicted
        self._r = np.divide(self._ob, self._pr)

        # Compute only where observed is non-zero: NaN -> 0/0 -> set to 0.0
        self._r[np.argwhere(np.isnan(self._r))] = 0.0

        # Multiply by initial to get updated pdf
        self._up = np.multiply(self._in, self._r)


    def mud_point(self):
        """Maximal Updated Density (MUD) Point

        Returns the Maximal Updated Density or MUD point as the parameter sample
        from the initial distribution with the highest update density value.

        Parameters
        ----------

        Returns
        -------
        mud_point : ndarray
            Maximal Updated Density (MUD) point.
        """
        if self._up is None:
            self.fit()
        m = np.argmax(self._up)
        return self.X[m, :]


    def estimate(self):
        """Estimate

        Returns the best estimate for most likely paramter values for the given
        model data using the data-consistent framework.

        Parameters
        ----------

        Returns
        -------
        mud_point : ndarray
            Maximal Updated Density (MUD) point.
        """
        return self.mud_point()


    def exp_r(self):
        """Expectation Value of R

        Returns the expectation value of the R, the ratio of the observed to the
        predicted density values. If the predictability assumption for the data-
        consistent framework is satisfied, then this value should be close to 1
        up to sampling errors.

        Parameters
        ----------

        Returns
        -------
        exp_r : float
            Value of the E(r). Should be close to 1.0.
        """
        if self._up is None:
            self.fit()
        return np.mean(self._r)


class BayesProblem(object):
    """
    Sets up Bayesian Inverse Problem for parameter identification


    Example Usage
    -------------

    >>> from mud.base import BayesProblem
    >>> import numpy as np
    >>> from scipy.stats import distributions as ds
    >>> X = np.random.rand(100,1)
    >>> num_obs = 50
    >>> Y = np.repeat(X, num_obs, 1)
    >>> y = np.ones(num_obs)*0.5 + np.random.randn(num_obs)*0.05
    >>> B = BayesProblem(X, Y, np.array([[0,1], [0,1]]))
    >>> B.set_likelihood(ds.norm(loc=y, scale=0.05))
    >>> np.round(B.map_point()[0],1)
    0.5

    """

    def __init__(self, X, y, domain=None):
        self.X = X
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.y = y
        self.domain = domain
        self._ps = None
        self._pr = None
        self._ll = None

    def set_likelihood(self, distribution, log=False):
        if log:
            self._log = True
            self._ll = distribution.logpdf(self.y).sum(axis=1)
            # below is an equivalent evaluation (demonstrating the expected symmetry)
            # std, mean = distribution.std(), distribution.mean()
            # self._ll = dist.norm(self.y, std).logpdf(mean).sum(axis=1)
        else:
            self._log = False
            self._ll = distribution.pdf(self.y).prod(axis=1)
            # equivalent
            # self._ll = dist.norm(self.y).pdf(distribution.mean())/distribution.std()
            # self._ll = self._ll.prod(axis=1)
        self._ps = None

    def set_prior(self, distribution=None):
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        prior_dist = distribution
        self._pr = prior_dist.pdf(self.X).prod(axis=1)
        self._ps = None

    def fit(self):
        if self._pr is None:
            self.set_prior()
        if self._ll is None:
            self.set_likelihood()

        if self._log:
            ps_pdf = np.add(np.log(self._pr), self._ll)
        else:
            ps_pdf = np.multiply(self._pr, self._ll)

        assert ps_pdf.shape[0] == self.X.shape[0]
        if np.sum(ps_pdf) == 0:
            raise ValueError("Posterior numerically unstable.")
        self._ps = ps_pdf

    def map_point(self):
        if self._ps is None:
            self.fit()
        m = np.argmax(self._ps)
        return self.X[m, :]

    def estimate(self):
        return self.map_point()


class WeightedDensityProblem(DensityProblem):
    """
    Sets up a Weighted Data-Consistent Inverse Problem for parameter
    identification.


    Example Usage
    -------------

    >>> from mud.base import WeightedDensityProblem as WDP
    >>> from mud.funs import wme
    >>> import numpy as np
    >>> num_sapmles = 100
    >>> X = np.random.rand(num_samples,1)
    >>> num_obs = 50
    >>> Y = np.repeat(X, num_obs, 1)
    >>> y = np.ones(num_obs)*0.5 + np.random.randn(num_obs)*0.05
    >>> W = wme(Y, y)
    >>> weights = np.ones(num_samples)
    >>> B = WDP(X, W, domain=np.array([[0,1], [0,1]]), weights=weights)
    >>> np.round(B.mud_point()[0],1)
    0.5
    >>> np.round(B.exp_r(),1)
    1.2

    """
    def __init__(self, X, y, domain=None, weights=None):
        super().__init__(X, y, domain=domain)
        self.set_weights(weights)


    def set_weights(self, weights=None):
        # weights is array of ones if non specified, and 2D always
        w = np.ones(self.X.shape[0]) if weights is None else weights
        w = w.reshape(1, -1) if w.ndim==1 else weights

        # Verify length of each weight vectors match number of samples in X
        assert self.X.shape[0]==w.shape[1]

        # Multiply weights column wise to get one weight row vector
        w = np.prod(w, axis=0)

        # Normalize weight vector
        self._weights  = np.divide(w, np.sum(w,axis=0))

        # Re-set initial, predicted, and updated
        self._in = None
        self._pr = None
        self._up = None


    def set_initial(self, distribution=None):
        super().set_initial(distribution=distribution)
        self._in = self._in * self._weights


    def set_predicted(self, distribution=None, bw_method=None):
        super.set_predicted(distribution=distribution, weights=self._weights)


    def exp_r(self):
        if self._up is None:
            self.fit()
        return np.average(self._r, weights=self._weights)


