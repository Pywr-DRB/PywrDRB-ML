# A collection of thermal control policies for the thermal control system.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

class BasePolicy:
    def __init__(self):
        pass

    def plot_response_surface(self, dim1=0, dim2=1, resolution=100):
        """
        Plot 3D surface for a generalized piecewise linear policy with two selected input dimensions.

        Parameters
        ----------
        dim1 : int
            Index of the first input dimension to vary (x-axis).
        dim2 : int
            Index of the second input dimension to vary (y-axis).
        resolution : int
            Number of points per axis for the surface plot grid.

        Assumes
        -------
        - All input dimensions are normalized to [0, 1].
        - Other input dimensions are fixed at 0.5.
        """

        # Grid of values for the two selected dimensions
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.linspace(0, 1, resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        Z = np.zeros_like(X_grid)
        for i in range(resolution):
            for j in range(resolution):
                # Construct full input vector with all values = 0.5, except selected dims
                X_input = [0.5] * self.n_dim
                X_input[dim1] = X_grid[i, j]
                X_input[dim2] = Y_grid[i, j]

                z = self.run(X_input)
                Z[i, j] = z

        # Plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_grid, Y_grid, Z, cmap='plasma')

        ax.set_xlabel(f'x{dim1} input', labelpad=10, fontsize=12)
        ax.set_ylabel(f'x{dim2} input', labelpad=10, fontsize=12)
        ax.set_zlabel('Response', labelpad=15, fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        plt.tight_layout()
        plt.show()

    def plot_response_contour(self, dim1=0, dim2=1, resolution=100, filled=True, levels=20, cmap='plasma'):
        """
        Plot 2D contour map for a generalized piecewise linear policy
        with two selected input dimensions.

        Parameters
        ----------
        dim1 : int
            Index of the first input dimension to vary (x-axis).
        dim2 : int
            Index of the second input dimension to vary (y-axis).
        resolution : int
            Number of points per axis for the surface plot grid.
        filled : bool
            If True, uses filled contours (contourf); otherwise, contour lines.
        levels : int
            Number of contour levels.
        cmap : str
            Colormap to use.
        """
        # Generate grid
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.linspace(0, 1, resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X_grid)

        # Evaluate policy on grid
        for i in range(resolution):
            for j in range(resolution):
                X_input = [0.5] * self.n_dim
                X_input[dim1] = X_grid[i, j]
                X_input[dim2] = Y_grid[i, j]
                Z[i, j] = self.run(X_input)

        # Plot 2D contour
        fig, ax = plt.subplots(figsize=(8, 6))
        if filled:
            contour = ax.contourf(X_grid, Y_grid, Z, levels=levels, cmap=cmap)
            fig.colorbar(contour, ax=ax, label='Response')
        else:
            contour = ax.contour(X_grid, Y_grid, Z, levels=levels, cmap=cmap)
            ax.clabel(contour, inline=True, fontsize=8)

        ax.set_xlabel(f'x{dim1} input')
        ax.set_ylabel(f'x{dim2} input')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_response_line(self, dim=0, resolution=200):
        """
        Plot 1D line of the policy response as a function of a single input dimension.

        Parameters
        ----------
        dim : int
            Index of the input dimension to vary.
        resolution : int
            Number of points to evaluate in the range [0, 1].
        """
        # Generate input values
        x_vals = np.linspace(0, 1, resolution)
        y_vals = []

        for x in x_vals:
            X_input = [0.5] * self.n_dim
            X_input[dim] = x
            y = self.run(X_input)
            y_vals.append(y)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, y_vals, label=f'x{dim} input', lw=2)
        ax.set_xlabel(f'x{dim} input')
        ax.set_ylabel('Response')
        ax.grid(True)
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def gen_params(self, n=1, seed=None):
        """
        Generate random parameters for the policy.

        Parameters
        ----------
        n : int
            Number of parameter sets to generate.
        seed : int, optional
            Random seed for reproducibility. If None, uses a random seed.

        Returns
        -------
        np.ndarray
            Array of shape (n, self.n_params) containing the generated parameters.
        """
        if seed is not None:
            rng = np.random.default_rng(seed=42)
            params = rng.uniform(0, 1, (n, self.n_params))
        else:
            params = np.random.uniform(0, 1, (n, self.n_params))
        return params

class RuleBasedPolicy(BasePolicy):
    """
    A class representing a rule-based thermal control policy.

    This policy implements a simple rule-based approach to determine the amount of water to release
    based on the current temperature. If the temperature exceeds a specified threshold, it releases
    a fixed amount of water; otherwise, it does not release any water.

    Parameters
    ----------
    threshold : float, optional
        The temperature threshold for releasing water (default is 24 degrees Celsius).
    thermal_release : float, optional
        The amount of water to release when the temperature exceeds the threshold (default is 300 mgd).
    """
    def __init__(self, threshold=24, thermal_release_amount=300):
        self.threshold = threshold
        self.thermal_release_amount = thermal_release_amount

    def run(self, X,  *args, **kwargs):
        """
        Execute the rule-based policy.

        Parameters
        ----------
        X : array-like of shape (1,)
            Input feature vector, where X[0] is the current temperature in degrees Celsius.

        Returns
        -------
        float
            The amount of water to release in million gallons per day (mgd).
        """
        if X[0] >= self.threshold:
            return self.thermal_release_amount  # mgd
        else:
            return 0  # No release

class GeneralizedPiecewiseLinearPolicy(BasePolicy):
    """
    A generalized piecewise linear policy for thermal control.

    For each input dimension, the parameters specify:
        - distances between breakpoints (positive values summing to 1 after normalization)
        - corresponding z_values for each breakpoint

    The x_breaks are constructed to span [0, 1], with the first break at 0 and the last at 1.

    Parameters
    ----------
    n_dim : int
        Number of input dimensions.
    n_steps : int or list of int
        Number of steps (linear segments) per dimension.
    """

    def __init__(self, n_dim, n_steps):
        super().__init__()
        self.n_dim = n_dim

        if isinstance(n_steps, int):
            n_steps = [n_steps] * n_dim
        self.n_steps = n_steps
        self.n_params = self._compute_n_params()

    def set_params(self, *params):
        """
        Set the parameters for the policy.

        Parameters
        ----------
        params : array-like of shape (n_params,)
            New parameters to set for the policy.
        """
        params = np.asarray(params)
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(params)}."
        self.params = params

    def _compute_n_params(self):
        # Each dimension has:
        # - n_step distances
        # - (n_step + 1) z values
        return sum(n_step + (n_step + 1) for n_step in self.n_steps)

    def run(self, X):
        """
        Compute the policy output as the average of piecewise linear outputs.

        Parameters
        ----------
        X : array-like of shape (n_dim,)
            Input vector with values in [0, 1].

        Returns
        -------
        z : float
            Averaged output from all piecewise linear mappings.
        """
        X = np.asarray(X)
        assert len(X) == self.n_dim, "Input dimension mismatch."

        z_total = 0
        i = 0
        for d in range(self.n_dim):
            n_step = self.n_steps[d]
            n_points = n_step + 1

            # Extract distance and z values
            distances = np.array(self.params[i : i + n_step])
            i += n_step
            z_values = np.array(self.params[i : i + n_points])
            i += n_points

            # Normalize distances to sum to 1 and compute breakpoints
            distances = distances / np.sum(distances)
            x_breaks = np.concatenate(([0], np.cumsum(distances)))

            # Ensure x_breaks ends exactly at 1
            x_breaks[-1] = 1.0

            # Clip and interpolate
            xi = np.clip(X[d], 0, 1)
            zi = np.interp(xi, x_breaks, z_values)
            z_total += zi

        return z_total / self.n_dim

class RegressionPolicy(BasePolicy):
    """
    A class representing a regression-based policy for thermal control.

    This policy uses polynomial regression to map a normalized input vector
    to a scalar output value. The model is defined by a fixed polynomial
    degree and corresponding coefficients.

    Parameters
    ----------
    degree : int, optional
        The degree of the polynomial used for regression (default is 2).
    *params : list
        A flattened 1D list of polynomial coefficients, where the first element
        is the constant term, followed by coefficients for each degree term.
    Example
    -------
    For a quadratic polynomial with coefficients [1, -2, 3]:
        params = [1, -2, 3]
        policy = RegressionPolicy(degree=2, *params)
        output = policy.run([0.5, 0.5])  # Normalized input vector

    """

    def __init__(self, n_dim, degree=2):
        self.degree = degree
        self.n_dim = n_dim
        self.n_params = self._compute_n_params()
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

    def set_params(self, *params):
        """
        Set the parameters for the policy.

        Parameters
        ----------
        params : array-like of shape (n_params,)
            New parameters to set for the policy.
        """
        params = np.asarray(params)
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(params)}."
        self.params = params

    def _compute_n_params(self):
        """
        Compute the total number of parameters based on the polynomial degree and input dimensions.
        The number of parameters is given by the formula:
        (degree + 1) * (n_dim + degree) // 2
        This accounts for all combinations of input dimensions up to the specified degree.
        """
        return (self.degree + 1) * (self.n_dim + self.degree) // 2

    def run(self, X):
        """
        Predict the policy output for input vector X.

        Parameters
        ----------
        X : array-like of shape (n_features,)
            Normalized input feature vector in [0, 1].

        Returns
        -------
        z : float
            Scalar policy output.
        """
        X = np.asarray(X)
        assert X.shape[0] == self.n_dim, "Input dimension mismatch."

        X = X.reshape(1, -1)
        X_poly = self.poly.fit_transform(X)
        y = float(X_poly @ self.params)

        # Force y to be in [0, 1]
        y = np.clip(y, 0, 1)
        return y

class GaussianRBFPolicy(BasePolicy):
    """
    A class representing a Gaussian Radial Basis Function (RBF) policy for thermal control.

    Parameters
    ----------
    *params : list
        A 1D array of RBF parameters, flattened. The array contains:
            - First A values: weights (w_i)
            - Next A*B values: centers (c_j,i)
            - Next A*B values: basis (b_j,i)

    Example
    -------
    For 2D input and 3 RBFs:
        params = [
            0.2, 0.2, 0.1, 0.5,    # center1, sigma1, weight1
            0.5, 0.5, 0.15, 1.0,   # center2, sigma2, weight2
            0.8, 0.8, 0.1, 0.3     # center3, sigma3, weight3
        ]
    """

    def __init__(self, n_dim, n_basis):
        self.n_dim = n_dim
        self.n_basis = n_basis
        self.n_params = self._compute_n_params()

    def set_params(self, *params):
        """
        Set the parameters for the policy.

        Parameters
        ----------
        params : array-like of shape (n_params,)
            New parameters to set for the policy.
        """
        params = np.asarray(params)
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(params)}."
        self.params = params
        self._parse_params(params)

    def _parse_params(self, params):
        n_dim = self.n_dim
        n_basis = self.n_basis

        # centers [0, 1]
        centers = params[:n_dim*n_basis].reshape(n_dim, n_basis)

        # b = 2*sigma^2 [0, 1]
        basises = params[n_dim*n_basis:n_dim*n_basis*2].reshape(n_dim, n_basis)
        basises = np.where(basises != 0.0, basises, 1e-6) # Avoid zero sigma

        weights = params[n_dim*n_basis*2:]
        weights /= (np.sum(weights)+1e-10)  # Normalizing weights to sum to 1

        self.weights = weights
        self.centers = centers
        self.basises = basises

    def _compute_n_params(self):
        """
        Compute the total number of parameters based on the number of dimensions and basis functions.
        Each RBF has n_dim centers, 1 sigma, and 1 weight.
        """
        return self.n_basis + 2*(self.n_dim*self.n_basis)

    def run(self, X):
        X = np.asarray(X)
        assert X.shape[0] == self.n_dim, "Input dimension mismatch."

        # Vectorized computation of squared differences
        # centers: (n_dim, n_basis), X[:, None]: (n_dim, 1)
        diffs = X[:, None] - self.centers  # shape: (n_dim, n_basis)
        # basises: (n_dim, n_basis)
        exponent = -np.sum((diffs ** 2) / (self.basises ** 2), axis=0)  # shape: (n_basis,)
        rbf_values = np.exp(exponent)  # shape: (n_basis,)

        # Weighted sum
        y = np.dot(self.weights, rbf_values)
        
        # Force y to be in [0, 1]
        y = np.clip(y, 0, 1)
        return float(y)

class CubicRBFPolicy(BasePolicy):
    """
    A class representing a Cubic Radial Basis Function (RBF) policy for thermal control.

    This policy uses a weighted sum of cubic RBFs:
        φ(r) = r³, where r is the Euclidean distance from the center.

    Parameters
    ----------
    n_dim : int
        Number of input features.
    n_basis : int
        Number of RBF centers.
    *params : list
        A flattened 1D list with the following format:
            - For each RBF (repeat n_basis times):
                - center_i : n_dim floats
                - weight_i : float

    Example
    -------
    For 2D input and 3 RBFs:
        params = [
            0.2, 0.2,     # center1, weight1
            0.5, 0.5,     # center2, weight2
            0.8, 0.8,     # center3, weight3
        ]
    """

    def __init__(self, n_dim, n_basis):
        self.n_dim = n_dim
        self.n_basis = n_basis
        self.n_params = self._compute_n_params()

    def set_params(self, *params):
        """
        Set the parameters for the policy.

        Parameters
        ----------
        params : array-like of shape (n_params,)
            New parameters to set for the policy.
        """
        params = np.asarray(params)
        assert len(params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(params)}."
        self.params = params
        self._parse_params(params)

    def _parse_params(self, params):
        n_dim = self.n_dim
        n_basis = self.n_basis

        # centers [0, 1]
        centers = params[:n_dim*n_basis].reshape(n_dim, n_basis)

        # b = 2*sigma^2 [0, 1]
        basises = params[n_dim*n_basis:n_dim*n_basis*2].reshape(n_dim, n_basis)
        basises = np.where(basises != 0.0, basises, 1e-6) # Avoid zero sigma

        weights = params[n_dim*n_basis*2:]
        weights /= (np.sum(weights)+1e-10)  # Normalizing weights to sum to 1

        self.weights = weights
        self.centers = centers
        self.basises = basises

    def _compute_n_params(self):
        """
        Compute the total number of parameters based on the number of dimensions and basis functions.
        Each RBF has n_dim centers, 1 sigma, and 1 weight.
        """
        return self.n_basis + 2*(self.n_dim*self.n_basis)

    def run(self, X):
        X = np.asarray(X)
        assert X.shape[0] == self.n_dim, "Input dimension mismatch."

        # Vectorized computation of squared differences
        # centers: (n_dim, n_basis), X[:, None]: (n_dim, 1)
        diffs = X[:, None] - self.centers  # shape: (n_dim, n_basis)
        # basises: (n_dim, n_basis)
        rbf_values = np.sum(np.abs((diffs ** 3) / (self.basises ** 3)), axis=0)  # shape: (n_basis,)

        # Weighted sum
        y = np.dot(self.weights, rbf_values)
        
        # Force y to be in [0, 1]
        y = np.clip(y, 0, 1)
        return float(y)