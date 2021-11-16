import pdb
import numpy as np
from mud.linear import LinearGaussianProblem
from matplotlib import pyplot as plt


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



