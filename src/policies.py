# A collection of thermal control policies for the thermal control system.
import math
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
            rng = np.random.default_rng(seed=seed)
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
        The amount of water to release when the temperature exceeds the threshold (default is 65 mgd (~100 cfs)).
    """
    def __init__(self, threshold=24, thermal_release_amount=65):
        super().__init__()
        self.threshold = threshold
        self.thermal_release_amount = thermal_release_amount 
        
        self.name = "RuleBasedPolicy"

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
        
        # For Borg
        self.param_names = self._gen_param_names()
        self.bounds = self.assign_bounds()
        
        self.name = "GeneralizedPiecewiseLinearPolicy"

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
    
    def _gen_param_names(self):
        """
        Generate parameter names for the policy.

        Returns
        -------
        list of str
            List of parameter names for each piecewise linear segment.
        """
        param_names = []
        for d in range(self.n_dim):
            n_step = self.n_steps[d]
            for i in range(n_step):
                param_names.append(f'dist_{d}_{i}')
            for i in range(n_step + 1):
                param_names.append(f'z_{d}_{i}')
        self.param_names = param_names
        return param_names
    
    def assign_bounds(self, x_range=[0.01, 1], z_range=[0, 1]):
        """
        Assign bounds for the policy parameters.

        Parameters
        ----------
        x_range : list of tuple
            List of (min, max) bounds for each x parameter (distance).
            Default [0.01, 1] prevents zero distances which cause numerical issues.
        z_range : list of tuple, optional
            List of (min, max) bounds for each z parameter (default is [0, 1]).
        
        Note: Distance parameters are normalized to sum to 1, so they represent
        relative segment lengths. Minimum of 0.01 ensures all segments exist.
        Z parameters directly control output values at breakpoints.
        """
        bounds = []
        for d in range(self.n_dim):
            n_step = self.n_steps[d]
            for i in range(n_step):
                bounds.append([x_range[0], x_range[1]])  # distances
            for i in range(n_step + 1):
                bounds.append([z_range[0], z_range[1]])  # z values
        self.bounds = bounds
        return bounds

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
    The degree-2 polynomial features are [1, a, b, a^2, ab, b^2]
    f(x1, x2) = c0 + c1*x1 + c2*x2 + c3*x1² + c4*x1*x2 + c5*x2²

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
        
        self.name = "RegressionPolicy"
        
        # For Borg
        self.param_names = self._gen_param_names()
        self.bounds = self.assign_bounds()

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
        The number of parameters is given by the binomial coefficient:
        C(n_dim + degree, degree) = (n_dim + degree)! / (degree! * n_dim!)
        This accounts for all combinations of input dimensions up to the specified degree.
        """
        return math.comb(self.n_dim + self.degree, self.degree)

    def _gen_param_names(self):
        """
        Generate parameter names for the policy.

        Returns
        -------
        list of str
            List of parameter names for each piecewise linear segment.
        """
        param_names = [f"c{i}" for i in range(self.n_params)]
        return param_names
    
    def assign_bounds(self, output_range=[0, 1]):
        """
        Assign parameter bounds to ensure output stays within specified range
        for inputs in [0,1]^n_dim.
        
        For polynomial regression with normalized inputs [0,1] and outputs [0,1]:
        - Constant term (c0): [0, 1] to ensure non-negative baseline
        - Other coefficients: [-2, 2] to allow flexibility while maintaining output bounds
        
        Note: These bounds are conservative. Tighter bounds could be derived
        from constraint optimization, but would be computationally expensive.
        """
        bounds = []
        for i in range(self.n_params):
            if i == 0:  # Constant term
                bounds.append([output_range[0], output_range[1]])
            else:  # Linear, quadratic, and interaction terms
                # Allow negative coefficients for flexibility
                # The clipping in run() ensures output bounds are respected
                bounds.append([-2.0, 2.0])
        return bounds
    
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
        
        # Check for dimension mismatch and provide helpful error message
        if X_poly.shape[1] != len(self.params):
            raise ValueError(
                f"Dimension mismatch: polynomial features have {X_poly.shape[1]} terms "
                f"but parameters have {len(self.params)} values. "
                f"Expected {self.n_params} parameters for n_dim={self.n_dim}, degree={self.degree}. "
                f"Consider reinitializing the policy with correct parameters."
            )
        
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
        
        self.name = "GaussianRBFPolicy"
        
        # For Borg
        self.param_names = self._gen_param_names()
        self.bounds = self.assign_bounds()

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

        # Parse parameters in the order: weight_i, center_d_i, basis_d_i for each basis i
        # This matches the order from _gen_param_names and assign_bounds
        weights = np.zeros(n_basis)
        centers = np.zeros((n_dim, n_basis))
        basises = np.zeros((n_dim, n_basis))
        
        idx = 0
        for i in range(n_basis):
            # Weight for basis i
            weights[i] = params[idx]
            idx += 1
            
            # Centers and basises for all dimensions of basis i
            for d in range(n_dim):
                centers[d, i] = params[idx]
                idx += 1
                basises[d, i] = params[idx]
                idx += 1
        
        # Avoid zero sigma
        basises = np.where(basises != 0.0, basises, 1e-6)
        
        # Normalizing weights to sum to 1
        weights /= (np.sum(weights) + 1e-10)

        self.weights = weights
        self.centers = centers
        self.basises = basises
        
    def _gen_param_names(self):
        """
        Generate parameter names for the policy.

        Returns
        -------
        list of str
            List of parameter names for each RBF.
        """
        param_names = []
        for i in range(self.n_basis):
            param_names.append(f'weight_{i}')
            for d in range(self.n_dim):
                param_names.append(f'center_{d}_{i}')
                param_names.append(f'basis_{d}_{i}')
        return param_names
    
    def assign_bounds(self, weight_bound=[0, 1], center_bound=[0, 1], basis_bound=[1e-6, 2.0]):
        """
        Assign parameter bounds to ensure output stays within specified range
        for inputs in [0,1]^n_dim.
        
        For Gaussian RBFs with normalized inputs [0,1] and outputs [0,1]:
        - Weights: [0, 1] - ensures non-negative contributions, normalized to sum=1
        - Centers: [0, 1] - matches input domain for optimal coverage  
        - Basis (sigma): [1e-6, 2.0] - avoids numerical issues while allowing both
          narrow (fine details) and broad (global trends) RBFs
        
        Note: Larger sigma allows RBFs to influence larger regions, improving
        generalization for smooth functions.
        """
        bounds = []
        for i in range(self.n_basis):
            bounds.append(weight_bound)
            for d in range(self.n_dim):
                bounds.append(center_bound)
                bounds.append(basis_bound)
        return bounds

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
        φ(r) = |r|³, where r is the scaled distance from the center.

    Parameters
    ----------
    n_dim : int
        Number of input features.
    n_basis : int
        Number of RBF centers.
    *params : list
        A flattened 1D list with the following format:
            - For each RBF (repeat n_basis times):
                - weight_i : 1 float
                - center_i : n_dim floats  
                - basis_i : n_dim floats (scaling parameters)

    Example
    -------
    For 2D input and 2 RBFs:
        params = [
            0.5,           # weight_0
            0.2, 0.3,      # center_0 (x, y)
            1.0, 1.0,      # basis_0 (scale_x, scale_y)
            0.5,           # weight_1
            0.7, 0.8,      # center_1 (x, y)  
            0.5, 0.5,      # basis_1 (scale_x, scale_y)
        ]
    """

    def __init__(self, n_dim, n_basis):
        self.n_dim = n_dim
        self.n_basis = n_basis
        self.n_params = self._compute_n_params()
        
        self.name = "CubicRBFPolicy"
        
        # For Borg
        self.param_names = self._gen_param_names()
        self.bounds = self.assign_bounds()

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

    def _compute_n_params(self):
        """
        Compute the total number of parameters based on the number of dimensions and basis functions.
        Each RBF has n_dim centers, n_dim basis parameters, and 1 weight.
        """
        return self.n_basis + 2*(self.n_dim*self.n_basis)

    def _parse_params(self, params):
        n_dim = self.n_dim
        n_basis = self.n_basis

        # Parse parameters in the order: weight_i, center_d_i, basis_d_i for each basis i
        # This matches the order from _gen_param_names and assign_bounds
        weights = np.zeros(n_basis)
        centers = np.zeros((n_dim, n_basis))
        basises = np.zeros((n_dim, n_basis))
        
        idx = 0
        for i in range(n_basis):
            # Weight for basis i
            weights[i] = params[idx]
            idx += 1
            
            # Centers and basises for all dimensions of basis i
            for d in range(n_dim):
                centers[d, i] = params[idx]
                idx += 1
                basises[d, i] = params[idx]
                idx += 1
        
        # Avoid zero basis values
        basises = np.where(basises != 0.0, basises, 1e-6)
        
        # Normalizing weights to sum to 1
        weights /= (np.sum(weights) + 1e-10)

        self.weights = weights
        self.centers = centers
        self.basises = basises

    def _gen_param_names(self):
        """
        Generate parameter names for the policy.

        Returns
        -------
        list of str
            List of parameter names for each RBF.
        """
        param_names = []
        for i in range(self.n_basis):
            param_names.append(f'weight_{i}')
            for d in range(self.n_dim):
                param_names.append(f'center_{d}_{i}')
                param_names.append(f'basis_{d}_{i}')
        return param_names

    def assign_bounds(self, weight_bound=[0, 0.5], center_bound=[0, 1], basis_bound=[0.1, 1.0]):
        """
        Assign parameter bounds to ensure output stays within specified range
        for inputs in [0,1]^n_dim.
        
        For Cubic RBFs with normalized inputs [0,1] and outputs [0,1]:
        - Weights: [0, 0.5] - REDUCED from [0,1] because cubic RBFs can grow large
          With normalized weights, this helps prevent output explosion
        - Centers: [0, 1] - matches input domain for optimal coverage  
        - Basis (scale): [0.1, 1.0] - INCREASED minimum from 1e-6 to prevent
          explosive growth near centers. Max distance in [0,1]^n_dim is sqrt(n_dim),
          so (1.0/0.1)³ = 1000 max, manageable with reduced weights
        
        Note: For cubic RBFs φ(r) = |r|³, the function grows rapidly with distance.
        These bounds balance expressiveness with output constraint satisfaction.
        """
        bounds = []
        for i in range(self.n_basis):
            bounds.append(weight_bound)
            for d in range(self.n_dim):
                bounds.append(center_bound)
                bounds.append(basis_bound)
        return bounds

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