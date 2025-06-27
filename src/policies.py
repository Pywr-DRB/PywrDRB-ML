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
        params = self.params
        # First, parse number of dimensions from param structure
        i = 0
        parsed_params = []
        while i < len(params):
            n_step = int(params[i])
            n_points = n_step + 1
            total_len = 1 + 2 * n_points
            parsed_params.append(params[i : i + total_len])
            i += total_len
        ndim = len(parsed_params)

        # Grid of values for the two selected dimensions
        x_vals = np.linspace(0, 1, resolution)
        y_vals = np.linspace(0, 1, resolution)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

        Z = np.zeros_like(X_grid)
        for i in range(resolution):
            for j in range(resolution):
                # Construct full input vector with all values = 0.5, except selected dims
                X_input = [0.5] * ndim
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
        ax.set_zlabel('Thermal Release', labelpad=15, fontsize=12)
        ax.set_title('Policy Surface: Generalized Piecewise Linear Function')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
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
    A class representing a generalized piecewise linear policy for thermal control.

    This policy allows for multiple input dimensions, each with its own set of breakpoints
    and corresponding output values. The output is computed as the average of the piecewise
    linear interpolated values for each input dimension.
    
    Parameters
    ----------
    *params : variable length
        For each input dimension xi, the parameters are provided in the following order:
            - x_breaks_i : array-like of shape (n_step_i + 1,)
                Strictly increasing breakpoints along xi's domain.
            - z_values_i : array-like of shape (n_step_i + 1,)
                Output values corresponding to each breakpoint.
    The total number of parameters should match the sum over all dimensions:
    dim * (1 + 2 * (n_step_i + 1))  # varies depending on per-dimension n_step_i
    """

    def __init__(self, n_dim, n_steps, *params):
        self.params = np.asarray(params)
        self.n_dim = n_dim
        if isinstance(n_steps, int):
            n_steps = [n_steps] * n_dim
        self.n_steps = n_steps
        self.n_params = self._compute_n_params()
        assert len(self.params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(self.params)}."
        
    def _compute_n_params(self):
        """
        Compute the total number of parameters based on the number of dimensions and steps.
        Each dimension has:
            - 1 for n_step
            - n_step + 1 for x_breaks
            - n_step + 1 for z_values
        Total: dim * (1 + 2 * (n_step + 1))
        """
        return sum((1 + 2 * (n_step + 1)) for n_step in self.n_steps)
    
    def run(self, X):
        """
        Compute the output of the generalized piecewise linear policy.

        Parameters
        ----------
        X : array-like of shape (n_dim,)
            Input feature vector. Each element corresponds to a dimension that
            will be mapped using its own piecewise linear function.

        Returns
        -------
        z : float
            Scalar output value, computed as the average of all per-dimension
            piecewise linear interpolated values.

        Raises
        ------
        ValueError
            If x_breaks are not strictly increasing for any dimension.
        """
        X = np.asarray(X)
        assert len(X) == self.n_dim, "Input dimension mismatch."

        z_total = 0
        i = 0
        for d in range(self.n_dim):
            n_step = self.n_steps[d]
            n_points = n_step + 1

            x_breaks = np.array(self.params[i : i + n_points])
            i += n_points
            z_values = np.array(self.params[i : i + n_points])
            i += n_points

            if not np.all(np.diff(x_breaks) > 0):
                raise ValueError(f"x_breaks for input {d} must be strictly increasing.")

            xi = np.clip(X[d], x_breaks[0], x_breaks[-1])
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

    def __init__(self, n_dim, degree=2, *params):
        self.params = np.asarray(params)
        self.degree = degree
        self.n_dim = n_dim
        self.n_params = self._compute_n_params()
        assert len(self.params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(self.params)}."
        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

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
        return float(X_poly @ self.params)

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

    def __init__(self, n_dim, n_basis, *params):
        self.n_dim = n_dim
        self.n_basis = n_basis
        self.params = np.asarray(params)
        self.n_params = self._compute_n_params()
        assert len(self.params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(self.params)}."
        self._parse_params(*params)

    def _parse_params(self, *params):
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

    def __init__(self, n_dim, n_basis, *params):
        self.n_dim = n_dim
        self.n_basis = n_basis
        self.params = np.asarray(params)
        self._parse_params(*params)
        assert len(self.params) == self.n_params, \
            f"Expected {self.n_params} parameters, got {len(self.params)}."
        self.n_params = self._compute_n_params()

    def _parse_params(self, *params):
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
        return float(y)