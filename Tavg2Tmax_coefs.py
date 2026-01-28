#%% Run TempLSTM and evaluate
import pandas as pd
import numpy as np
import pathnavigator
import matplotlib.pyplot as plt

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()

db_L = pd.read_csv(pn.data.raw.get("nwis_Lordville_degC_mgd.csv"), index_col=0, parse_dates=True)
mask = (
    (db_L['tavg_water_lordville_src'] == 'obs') &
    (db_L['tmmx_water_lordville_src'] == 'obs')
)

db_L = db_L.loc[mask, ['tavg_water_lordville', 'tmmx_water_lordville']]
#%%
def fit_linear_regression(x, y):
    """
    Simple linear regression

    mse = np.sum((y - y_fit)**2) / (n - 2)

    XTX_inv = np.linalg.inv(X.T @ X)
    X_new = np.vstack([x_new, np.ones_like(x_new)]).T
    se_pred = np.sqrt(mse * (1 + np.diag(X_new @ XTX_inv @ X_new.T)))

    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable

    Returns:
    --------
    tuple with slope, intercept, and XTX_inv for later prediction uncertainty
    """
    x = np.array(x)
    y = np.array(y)

    # Design matrix [x, 1] for y = ax + b
    X = np.vstack([x, np.ones_like(x)]).T

    # Least squares solution
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_fit = a * x + b

    # For later prediction uncertainty calculation
    n = len(y)
    mse = np.sum((y - y_fit)**2) / (n - 2)
    XTX_inv = np.linalg.inv(X.T @ X)

    # Calculate RMSE
    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.4f}")

    coefs = {"a": a, "b": b, "mse": mse, "XTX_inv": XTX_inv}
    return coefs

fit_linear_regression(
    db_L['tavg_water_lordville'], 
    db_L['tmmx_water_lordville']
    )

#{'a': 1.047946739057828,
# 'b': 0.42440192351902667,
# 'mse': 0.21422786125493337,
# 'XTX_inv': array([[ 2.37226498e-06, -2.40200717e-05],
#        [-2.40200717e-05,  3.70212479e-04]])}

#%%

def plot_linear_regression(x, y, coefs):
    """
    Plot data, fitted linear regression line, and RMSE.
    """
    x = np.array(x)
    y = np.array(y)

    a = coefs["a"]
    b = coefs["b"]
    mse = coefs["mse"]
    rmse = np.sqrt(mse)

    # Create fitted line
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b

    plt.figure(figsize=(4, 3.5))
    plt.scatter(x, y, alpha=0.7, s=1, label="Obs")
    plt.plot(x_line, y_line, linewidth=2, label=f"Fit: y = {a:.3f}x + {b:.3f}", c="r")
    plt.text(
        0.05, 0.95,
        f"RMSE = {rmse:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.xlabel("$T_{avg}$ (°C)")
    plt.ylabel("$T_{max}$ (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

x, y = db_L['tavg_water_lordville'], db_L['tmmx_water_lordville']
coefs = fit_linear_regression(x, y)
plot_linear_regression(x, y, coefs)