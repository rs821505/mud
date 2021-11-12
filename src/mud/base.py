import pdb
import numpy as np
from mud.util import null_space, make_2d_unit_mesh, updated_cov
from numpy.linalg import inv
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


class LinearGaussianProblem(object):
    """Inverse Problems with Linear/Affine Maps with Gaussian distributions.

    """

    def __init__(self,
            A=np.array([1, 1]).reshape(-1,1),
            b=None,
            y=None,
            mean_i =None,
            cov_i=None,
            cov_o=None,
            alpha=1.0):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)
        ns, di = self.A.shape

        # Initialize to defaults - Reshape everything into 2D arrays.
        self.b = np.zeros((ns, 1)) if b is None else b.reshape(-1, 1)
        self.y = np.zeros((ns, 1)) if y is None else y.reshape(-1, 1)
        self.mean_i = np.zeros((di, 1)) if mean_i is None else mean_i.reshape(-1, 1)
        self.cov_i = np.eye(di) if cov_i is None else cov_i
        self.cov_o = np.eye(ns) if cov_o is None else cov_o

        # How much to scale regularization terms
        self.alpha = alpha

        # Check appropriate dimensions of inputs
        n_data, n_targets = self.y.shape
        if ns != n_data:
            raise ValueError(
                "Number of samples in X and y does not correspond:"
                " %d != %d" % (ns , n_data)
            )

        # Initialize to no solution
        self.sol = None


    def compute_functionals(self, X, terms='all'):
        """
        For a given input and observed data, compute functionals or
        individual terms in functionals that are minimized to solve the
        linear gaussian problem.
        """
        # Compute observed mean
        mean_o = self.y - self.b

        # Define inner-producted induced by vector norm
        ip = lambda X, mat : np.sum(X * (np.linalg.inv(mat) @ X), axis=0)

        # First compute data mismatch norm
        data_term = ip((self.A @ X.T + self.b) - mean_o.T,
                       self.cov_o)
        if terms=='data': return data_term

        # Tikhonov Regularization Term
        reg_term =  self.alpha * ip((X- self.mean_i.T).T, self.cov_i)
        if terms=='reg': return reg_term

        # Data-Consistent Term - "unregularizaiton" in data-informed directions
        dc_term = self.alpha * ip(self.A @ (X - self.mean_i.T).T,
                             self.A @ self.cov_i @ self.A.T)
        if terms=='dc_term': return dc_term

        # Modified Regularization Term
        reg_m_terms = reg_term - dc_term
        if terms=='reg_m': return reg_m_terms

        bayes_fun = data_term + reg_term
        if terms=='bayes': return bayes_fun

        dc_fun  = bayes_fun - dc_term
        if terms=='dc': return dc_fun

        return (data_term, reg_term, dc_term, bayes_fun, dc_fun)


    def solve(self, method='mud', output_dim=None):
        """
        Explicitly solve linear problem using given method.

        """
        # Reduce output dimension if desired
        od = self.A.shape[0] if output_dim is None else output_dim
        _A = self.A[:od, :]
        _b = self.b[:od, :]
        _y = self.y[:od, :]
        _cov_o = self.cov_o[:od, :od]

        # Compute observed mean
        mean_o = _y - _b

        # Compute residual
        z = mean_o - _A @ self.mean_i

        # Weight initial covariance to use according to alpha parameter
        a_cov_i = self.alpha * self.cov_i

        # Solve according to given method, or solve all methods
        if method == 'mud' or method == 'all':
            inv_pred_cov = np.linalg.pinv(_A @ a_cov_i @ _A.T)
            update = a_cov_i @ _A.T @ inv_pred_cov
            self.mud = self.mean_i + update @ z

        # if method == 'mud_alt' or method == 'all':
        #     up_cov = updated_cov(X=_A, init_cov=a_cov_i, data_cov=_cov_o)
        #     update = up_cov @ _A.T @ np.linalg.inv(_cov_o)
        #     self.mud_alt = self.mean_i + update @ z

        if method == 'map' or method == 'all':
            co_inv = inv(_cov_o)
            cov_p = inv(_A.T @ co_inv @ _A + inv(a_cov_i))
            update = cov_p @ _A.T @ co_inv
            self.map = self.mean_i + update @ z

        if method == 'ls' or method == 'all':
            # Compute ls solution from pinv method
            self.ls = (np.linalg.pinv(_A) @ mean_o)

        # Return solution or all solutions
        if method =='all':
            return (self.mud, self.map, self.ls)
            # return (self.mud, self.mud_alt, self.map, self.ls)
        else:
            return self.__getattribute__(method)


    def plot_sol(self,
            point='mud',
            ax=None,
            label=None,
            note_loc=None,
            pt_opts = {'color':'k', 's':100, 'marker':'o'},
            ln_opts = {'color':'xkcd:blue', 'marker':'d', 'lw':1, 'zorder':10},
            annotate_opts={'fontsize':14, 'backgroundcolor':'w'}):
        """
        Plot solution points
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get solution point or initial poitn to plot.
        pt = self.mean_i if point=='initial' else self.solve(method=point)
        pt_opts['label'] = point

        # Plot point
        ax.scatter(pt[0], pt[1], **pt_opts)

        # Plot line connecting iniital value and solution
        if ln_opts is not None and point!='initial':
            ax.plot([self.mean_i.ravel()[0], pt.ravel()[0]],
                    [self.mean_i.ravel()[1], pt.ravel()[1]],
                    **ln_opts)

        if label is not None:
            # Annotate point with a label if desired
            nc = note_loc
            nc = (pt[0] - 0.02, pt[1] + 0.02) if nc is None else nc
            ax.annotate(label, nc, **annotate_opts)


    def plot_contours(self,
            ref=None,
            subset=None,
            ax=None,
            annotate=False,
            note_loc = None,
            w=1,
            label = "{i}",
            plot_opts={'color':"k", 'ls':":", 'lw':1, 'fs':20},
            annotate_opts={'fontsize':20}):
        """
        Plot Linear Map Solution Contours
        """
        # Initialize a plot if one hasn't been already
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # All rows of A are default subset of contours to plot
        subset = np.arange(self.A.shape[0]) if subset is None else subset

        # Ref is the reference point to plot each contour line through.
        ref = ref if ref is not None else self.solve(method='ls')

        # Build null-space (contour lines) for each subset row of A
        A = self.A[np.array(subset), :]
        numQoI = A.shape[0]
        AA = np.hstack([null_space(A[i, :].reshape(1, -1)) for i in range(numQoI)]).T

        # Plot each contour line going through ref point
        for i, contour in enumerate(subset):
            xloc = [ref[0] - w * AA[i, 0], ref[1] + w * AA[i, 0]]
            yloc = [ref[0] - w * AA[i, 1], ref[1] + w * AA[i, 1]]
            ax.plot(xloc, yloc, **plot_opts)

            # If annotate is set, then label line with given annotations
            if annotate:
                nl = (xloc[0], yloc[0]) if note_loc is None else note_loc
                ax.annotate(label.format(i=contour + 1), nl, **annotate_opts)


    def plot_fun_contours(self, mesh=None,
            terms='dc', ax=None, N=250, r=1, **kwargs):
        """
        Plot contour map offunctionals being minimized over input space
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Get mesh if one hasn't been passed
        if mesh is None:
            _, _, mesh = make_2d_unit_mesh(N, r)

        # Compute functional terms desired over range
        term = self.compute_functionals(mesh, terms=terms)

        # Plot contours
        _ = ax.contour(mesh[:, 0].reshape(N, N),
                       mesh[:, 1].reshape(N, N),
                       term.reshape(N, N), **kwargs)


class IterativeLinearProblem(LinearGaussianProblem):


    def __init__(self,
            A,
            b,
            y=None,
            initial_mean=None,
            cov=None,
            data_cov=None,
            idx_order=None):

        # Make sure A is 2D array
        self.A = A if A.ndim == 2 else A.reshape(1, -1)

        # Initialize to defaults - Reshape everythin into 2D arrays.
        n_samples, dim_input = self.A.shape
        self.data_cov = np.eye(n_samples) if data_cov is None else data_cov
        self.cov = np.eye(dim_input) if cov is None else cov
        self.initial_mean = np.zeros((dim_input, 1)) if initial_mean is None else initial_mean.reshape(-1,1)
        self.b = np.zeros((n_samples, 1)) if b is None else b.reshape(-1,1)
        self.y = np.zeros(n_samples) if y is None else y.reshape(-1,1)
        self.idx_order = range(self.A.shape[0]) if idx_order is None else idx_order

        # Verify arguments?

        # Initialize chain to initial mean
        self.epochs = []
        self.solution_chains = []
        self.errors = []


    def solve(self, num_epochs=1, method='mud'):
        """
        Iterative Solutions
        Performs num_epochs iterations of estimates

        """
        m_init = self.initial_mean if len(self.solution_chains)==0 else self.solution_chains[-1][-1]
        solutions = [m_init]
        for _ in range(0, num_epochs):
            epoch = []
            solutions = [solutions[-1]]
            for i in self.idx_order:
                # Add next sub-problem to chain
                epoch.append(LinearGaussianProblem(self.A[i, :],
                    self.b[i],
                    self.y[i],
                    mean=solutions[-1],
                    cov=self.cov,
                    data_cov=self.data_cov))

                # Solve next mud problem
                solutions.append(epoch[-1].solve(method=method))

            self.epochs.append(epoch)
            self.solution_chains.append(solutions)

        return self.solution_chains[-1][-1]


    def get_errors(self, ref_param):
        """
        Get errors with resepct to a reference parameter

        """
        solutions = np.concatenate([x[1:] for x in self.solution_chains])
        if len(solutions)!=len(self.errors):
            self.errors = [np.linalg.norm(s - ref_param) for s in solutions]
        return self.errors


    def plot_chain(self, ref_param, ax=None, color="k", s=100, **kwargs):
        """
        Plot chain of solutions and contours
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)
        for e, chain in enumerate(self.solution_chains):
            num_steps = len(chain)
            current_point = chain[0]
            ax.scatter(current_point[0], current_point[1], c="b", s=s)
            for i in range(0, num_steps):
                next_point = chain[i]
                points = np.hstack([current_point, next_point])
                ax.plot(points[0, :], points[1, :], c=color)
                current_point = next_point
            ax.scatter(current_point[0], current_point[1], c="g", s=s)
            ax.scatter(ref_param[0], ref_param[1], c="r", s=s)
        self.plot_contours(ref_param, ax=ax, subset=self.idx_order,
                color=color, s=s, **kwargs)


    def plot_chain_error(self, ref_param, ax=None, alpha=1.0,
            color="k", label=None, s=100, fontsize=12):
        """
        Plot error over iterations
        """
        _ = self.get_errors(ref_param)
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.set_yscale('log')
        ax.plot(self.errors, color=color, alpha=alpha, label=label)
        ax.set_ylabel("$||\lambda - \lambda^\dagger||$", fontsize=fontsize)
        ax.set_xlabel("Iteration step", fontsize=fontsize)



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


    def updated_cov(self, X, init_cov=None, data_cov=None):
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

