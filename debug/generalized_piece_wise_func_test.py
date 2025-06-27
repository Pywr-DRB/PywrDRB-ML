import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def generalized_piecewise_linear(X, *params):
    """
    Fully generalized piecewise linear function.

    Each input feature xi in X can have its own number of breakpoints and
    corresponding output values. The function computes a piecewise linear
    mapping for each input dimension independently and returns the average
    of the resulting values.

    Parameters
    ----------
    X : array-like of shape (n_features,)
        Input feature vector. Each element corresponds to a dimension that
        will be mapped using its own piecewise linear function.

    *params : variable length
        For each input dimension xi, the parameters are provided in the
        following order:
            - n_step_i : int
                Number of linear segments (i.e., number of breakpoints - 1)
            - x_breaks_i : array-like of shape (n_step_i + 1,)
                Strictly increasing breakpoints along xi's domain.
            - z_values_i : array-like of shape (n_step_i + 1,)
                Output values corresponding to each breakpoint.

        The total number of parameters should match the sum over all dimensions:
        dim * (1 + 2 * (n_step_i + 1))  // varies depending on per-dimension n_step_i

    Returns
    -------
    z : float
        Scalar output value, computed as the average of all per-dimension
        piecewise linear interpolated values.

    Raises
    ------
    ValueError
        If x_breaks are not strictly increasing for any dimension, or
        if the number of parameters is inconsistent with the structure described.

    Examples
    --------
    >>> X = [15.0, 25.0]
    >>> params = [
    ...     3, 0, 10, 20, 30,    0, 30, 15, 50,   # for x1
    ...     2, 0, 20, 40,        10, 5, 60        # for x2
    ... ]
    >>> generalized_piecewise_linear(X, *params)
    29.166666666666668
    """
    X = np.asarray(X)
    dim = len(X)
    i = 0  # pointer into params
    z_total = 0

    for d in range(dim):
        n_step = int(params[i])
        i += 1
        n_points = n_step + 1

        x_breaks = np.array(params[i : i + n_points])
        i += n_points
        z_values = np.array(params[i : i + n_points])
        i += n_points

        if not np.all(np.diff(x_breaks) > 0):
            raise ValueError(f"x_breaks for input {d} must be strictly increasing.")

        xi = np.clip(X[d], x_breaks[0], x_breaks[-1])
        zi = np.interp(xi, x_breaks, z_values)
        z_total += zi

    return z_total / dim


import numpy as np
import matplotlib.pyplot as plt

def plot_generalized_piecewise_surface(params, dim1=0, dim2=1, resolution=100):
    """
    Plot 3D surface for a generalized piecewise linear policy with two selected input dimensions.

    Parameters
    ----------
    params : list
        Flat parameter list for the generalized_piecewise_linear function.
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

            z = generalized_piecewise_linear(X_input, *params)
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

params = [
    3, 0.0, 0.33, 0.66, 1.0,    0, 0.3, 0.15, 0.5,   # x0
    2, 0.0, 0.5, 1.0,           0.1, 0.05, 0.6,      # x1
    1, 0.0, 1.0,                0.25, 0.25           # x2 (won't vary here)
]

plot_generalized_piecewise_surface(params, dim1=0, dim2=1)



