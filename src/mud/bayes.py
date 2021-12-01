import pdb
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import distributions as dist
from scipy.stats import gaussian_kde as gkde


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

        # Set inputs
        self.X = X
        self.y = y
        self.domain = domain

        if self.y.ndim == 1:
            # Reshape 1D to 2D array to keep things consistent
            self.y = self.y.reshape(-1, 1)

        # Get dimensions of inverse problem
        self.param_dim = self.X.shape[1]
        self.obs_dim = self.y.shape[1]

        if self.domain is not None:
            # Assert our domain passed in is consistent with data array
            assert domain.shape[0]==self.X.shape[1]

        # Initialize distributions and descerte values to None
        self._ps = None
        self._pr = None
        self._ll = None
        self._pr_dist = None
        self._ll_dist = None


    def set_likelihood(self, data, sigma, log=False):
        """

        """
        if log:
            self._log = True
            self._ll = distribution.logpdf(self.y).sum(axis=1)
        else:
            self._log = False
            self._ll = distribution.pdf(self.y).prod(axis=1)
        self._ll_dist = distribution
        self._ps = None


    def set_prior(self, distribution=None):
        if distribution is None:  # assume standard normal by default
            if self.domain is not None:  # assume uniform if domain specified
                mn = np.min(self.domain, axis=1)
                mx = np.max(self.domain, axis=1)
                distribution = dist.uniform(loc=mn, scale=mx - mn)
            else:
                distribution = dist.norm()
        self._pr_dist = distribution
        self._pr = self._pr_dist.pdf(self.X).prod(axis=1)
        self._ps = None


    def fit(self, data=None, log=False):
        if self._pr is None:
            self.set_prior()
        if self._ll is None:
            if data is None:
                raise ValueError("likelihood not set and data was not passed")
            self.set_likelihood(data, log)

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


    def plot_param_space(self,
            param_idx=0,
            ax=None,
            x_range=None,
            aff=1000,
            pr_opts={'color':'b', 'linestyle':'--', 'linewidth':4},
            ps_opts={'color':'g', 'linestyle':':', 'linewidth':4}):
        """
        Plot probability distributions over parameter space

        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Default x_range to full domain of all parameters
        x_range = x_range if x_range is not None else self.domain
        x_plot = np.linspace(x_range.T[0], x_range.T[1], num=aff)

        if pr_opts is not None:
            # Compute initial plot based off of stored initial distribution
            pr_plot = self._pr_dist.pdf(x_plot)

            # Plot prior distribution over parameter space
            ax.plot(x_plot[:,param_idx], pr_plot[:,param_idx], **pr_opts)

        if ps_opts is not None:
            # Compute posterior if it hasn't been already
            if self._ps is None:
                self.fit()

            # ps_plot - kde over params weighted by posterior computed pdf
            ps_plot = gkde(self.X.T, weights=self._ps)(x_plot.T)
            if self.param_dim==1:
                # Reshape two two-dimensional array if one-dim output
                ps_plot = ps_plot.reshape(-1,1)

            # Plot posterior distribution over parameter space
            ax.plot(x_plot[:,param_idx], ps_plot[:,param_idx], **ps_opts)


    def plot_obs_space(self,
            obs_idx=0,
            ax=None,
            y_range=None,
            aff=1000,
            ll_opts = {'color':'r', 'linestyle':'-', 'linewidth':4},
            pf_ps_opts = {'color':'g', 'linestyle':':', 'linewidth':4}):
        """
        Plot probability distributions defined over observable space.
        """

        if ax is None:
            _, ax = plt.subplots(1, 1)

        # Default range is (-1,1) over each observable variable
        if y_range is None:
            y_range = np.repeat([[-1,1]], self.y.shape[1], axis=0)

        # Default x_range to full domain of all parameters
        y_plot = np.linspace(y_range.T[0], y_range.T[1], num=aff)

        if ll_opts is not None:
            # Compute Likelihoood values
            ll_plot = self._ll_dist.pdf(y_plot)
            if self.obs_dim==1:
                # Reshape two two-dimensional array if one-dim output
                ll_plot = ll_plot.reshape(-1,1)

            # Plot pf of initial
            ax.plot(y_plot[:,obs_idx], ll_plot[:,obs_idx], **ll_opts)

        if pf_ps_opts is not None:
            # Compute PF of updated
            pf_ps_plot = gkde(self.y.T, weights=self._ps)(y_plot.T)
            if self.obs_dim==1:
                # Reshape two two-dimensional array if one-dim output
                pf_ps_plot = pf_ps_plot.reshape(-1,1)


            # Plut pf of updated
            ax.plot(y_plot[:,obs_idx], pf_ps_plot[:,obs_idx], **pf_ps_opts)
