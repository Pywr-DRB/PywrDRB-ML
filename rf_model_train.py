import pandas as pd
import numpy as np
import pathnavigator
import clt

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.rf_model import RandomForestUncertaintyModel

folder = "RFModels"
pn.models.mkdir(folder)

#%%
db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_TempLSTM.copy()['1979-01-01': '2023-12-31']
df_res = pd.DataFrame(index=database.index)
n_bootstrap = 100


rf_settings = {
    "n_estimators": 20,#35,
    "max_features": "sqrt",
    "random_state": 4,
    "n_jobs": -2,
    "max_depth": 10,#14,
    }

# Temp1
df = database[["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "doy", "QbcTavg_Q_C", "QbcTavg_T_C", "tavg_water_cannonsville_src"]].dropna()
df.loc[df["tavg_water_cannonsville_src"] != "obs", "QbcTavg_T_C"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
x_vars = ["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "doy", "QbcTavg_Q_C"]
y_var = "QbcTavg_T_C"
X = df[x_vars].values
obs = df[y_var].values
rf_model1 = RandomForestUncertaintyModel(x_vars, y_var, **rf_settings)
#rf_model1.grid_search(X, obs)
rf_model1.fit(X, obs)
rf_model1.cross_val_rmse(X, obs, n_splits=5, shuffle=False, threshold=20, weight=2)
#rf_model1.compute_tau(X, obs, n_bootstrap=n_bootstrap, disable=False)
#rf_model1.downsample_tau(size=5000)
rf_model1.save(pn.models.get(folder) / "rf_model1.gz")
# RMSE on training set: 1.487
# Mean RMSE from cross-validation: 1.911
X = database[rf_model1.x_vars].values
df_res["T_C"], df_res["T_C_lb"], df_res["T_C_ub"] = rf_model1.predict(X, quantile=0.95)

# Temp2
df = database[["tmmn", "tmmx", "pr", "srad", "QbcTavg_Q_i", "doy", "QbcTavg_Q_C", "QbcTavg_T_i", "tavg_water_src"]].dropna()
df.loc[df["tavg_water_src"] != "obs", "QbcTavg_T_i"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
x_vars = ["tmmn", "tmmx", "pr", "srad", "QbcTavg_Q_i", "doy", "QbcTavg_Q_C"]
y_var = "QbcTavg_T_i"
X = df[x_vars].values
obs = df[y_var].values
rf_model2 = RandomForestUncertaintyModel(x_vars, y_var, **rf_settings)
#rf_model2.grid_search(X, obs)
rf_model2.fit(X, obs)
rf_model2.cross_val_rmse(X, obs, n_splits=5, shuffle=False, threshold=20, weight=2)
#rf_model2.compute_tau(X, obs, n_bootstrap=n_bootstrap, disable=False)
#rf_model2.downsample_tau(size=5000)
rf_model2.save(pn.models.get(folder) / "rf_model2.gz")
# RMSE on training set: 2.529
# Mean RMSE from cross-validation: 4.242
X = database[rf_model2.x_vars].values
df_res["T_i"], df_res["T_i_lb"], df_res["T_i_ub"] = rf_model2.predict(X, quantile=0.95)


# rf_settings = {
#     "n_estimators": 15,
#     "max_features": "sqrt",
#     "random_state": 4,
#     "n_jobs": -2,
#     "max_depth": 5,
#     }
df = database[["QbcTavg_T_L", "QbcTmax_T_L", "doy", 'tavg_water_src', 'tmmx_water_src']]
df.loc[df["tavg_water_src"] != "obs", "QbcTavg_T_L"] = np.nan
df.loc[df["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan
df = df.dropna()
x_vars = ["QbcTavg_T_L", "doy"]
y_var = "QbcTmax_T_L"
X = df[x_vars].values
obs = df[y_var].values
rf_model_map = RandomForestUncertaintyModel(x_vars, y_var, **rf_settings)
#rf_model_map.grid_search(X, obs)
rf_model_map.fit(X, obs)
rf_model_map.cross_val_rmse(X, obs, n_splits=5, shuffle=False)
#rf_model_map.compute_tau(X, obs, n_bootstrap=n_bootstrap, disable=False)
#rf_model_map.downsample_tau(size=5000)
rf_model_map.save(pn.models.get(folder) / "rf_model_map.gz")
# RMSE on training set: 0.217
# Mean RMSE from cross-validation: 0.477

Q_C = database["QbcTavg_Q_C"]
Q_i = database["QbcTavg_Q_i"]
Q_L = database["QbcTavg_Q_L"]

df_res["Tavg_L"] = (df_res["T_C"]*Q_C + df_res["T_i"]*Q_i)/(Q_C+Q_i)
X = pd.concat([df_res[["Tavg_L"]], database[["doy"]]], axis=1).values
df_res["T_L"], df_res["T_L_lb"], df_res["T_L_ub"] = rf_model_map.predict(X, quantile=0.95)
df_res.to_csv(pn.models.get(folder) / "simple_run_Temp.csv")

#%% Plot T_L
pn.mkdir(pn.models.get(folder) / "figures")

import matplotlib.pyplot as plt
df = database[["QbcTavg_T_L", "QbcTmax_T_L", "doy", 'tavg_water_src', 'tmmx_water_src']]
df.loc[df["tavg_water_src"] != "obs", "QbcTavg_T_L"] = np.nan
df.loc[df["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan
obs = df["QbcTmax_T_L"]

for y in range(1979, 2024):
    df_res_ = df_res[f"{y}":f"{y}"]
    fig, ax = plt.subplots()
    ax.plot(df_res_.index, df_res_["T_L"], label="T_L", color='tab:blue', lw=2)
    ax.fill_between(df_res_.index, df_res_["T_L_lb"], df_res_["T_L_ub"], color='tab:blue', alpha=0.2, label='95% CI')
    ax.plot(df_res_.index, obs[f"{y}":f"{y}"], label="obs", color='k', linestyle='--', lw=1)
    ax.axhline(24, c="grey", ls="-.", lw=1)
    ax.axhline(20, c="grey", ls="-.", lw=1)
    ax.set_ylim([-1, 30])
    ax.set_xlabel("Date")
    ax.set_ylabel("T_L (degC)")
    ax.legend(ncols=3, loc="upper left", frameon=False)
    plt.tight_layout()
    clt.fig.savefig(fig, filename=pn.models.get(f"{folder}/figures") / f"temp_{y}.png")
    plt.show()

#%% Train salinity
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_SalinityLSTM.copy()['1979-01-01': '2023-12-31']

df_res = pd.DataFrame(index=database.index)

rf_settings = {
    "n_estimators": 20,#35,
    "max_features": "sqrt",
    "random_state": 4,
    "n_jobs": -2,
    "max_depth": 10,#14,
    }
n_bootstrap = 100

df = database[["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront", "saltfront_src"]].dropna()
df.loc[df["saltfront_src"] != "obs", "saltfront"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
x_vars = ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"]
y_var = "saltfront"
X = df[x_vars].values
obs = df[y_var].values
rf_model_saltfront = RandomForestUncertaintyModel(x_vars, y_var, **rf_settings)
#rf_model_saltfront.grid_search(X, obs)
rf_model_saltfront.fit(X, obs)
rf_model_saltfront.cross_val_rmse(X, obs, n_splits=5, shuffle=False, threshold=80, weight=10)
#rf_model_saltfront.compute_tau(X, obs, n_bootstrap=n_bootstrap, disable=False)
#rf_model_saltfront.downsample_tau(size=5000)
#rf_model_saltfront.save(pn.models.get(folder) / "rf_model_saltfront.gz")

# X = database[rf_model_saltfront.x_vars].values
# df_res["saltfront"], df_res["saltfront_lb"], df_res["saltfront_ub"] = rf_model_saltfront.predict(X, quantile=0.95)
# df_res.to_csv(pn.models.get(folder) / "simple_run_Saltfront.csv")

# RMSE on training set: 2.967
# Mean RMSE from cross-validation: 4.176

#%% Plot saltfront
pn.mkdir(pn.models.get(folder) / "figures")

import matplotlib.pyplot as plt
df = database[["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront", "saltfront_src"]].dropna()
df.loc[df["saltfront_src"] != "obs", "saltfront"] = np.nan
obs = df["saltfront"]

for y in range(1979, 2024):
    df_res_ = df_res[f"{y}":f"{y}"]
    fig, ax = plt.subplots()
    ax.plot(df_res_.index, df_res_["saltfront"], label="saltfront", color='tab:blue', lw=2)
    ax.fill_between(df_res_.index, df_res_["saltfront_lb"], df_res_["saltfront_ub"], color='tab:blue', alpha=0.2, label='95% CI')
    ax.plot(df_res_.index, obs[f"{y}":f"{y}"], label="obs", color='k', linestyle='--', lw=1)
    ax.axhline(82.9, c="grey", ls="-.", lw=1)
    ax.axhline(87, c="grey", ls="-.", lw=1)
    ax.axhline(92.5, c="grey", ls="-.", lw=1)
    ax.set_ylim([30, 95])
    ax.set_xlabel("Date")
    ax.set_ylabel("Saltfront (river mile)")
    ax.legend(ncols=3, loc="lower left", frameon=False)
    plt.tight_layout()
    clt.fig.savefig(fig, filename=pn.models.get(f"{folder}/figures") / f"salt_front_{y}.png")
    plt.show()


#%% Hyperparameter tuning for Random Forest model (Pending)
# for i in range(1, 50, 10):
#     print(i)
#     rf_settings = {
#         "n_estimators": 100,
#         "max_features": "sqrt",
#         "random_state": 4,
#         "n_jobs": -2,
#         "max_depth": i,
#         }
    
    
#     X = df[["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "QbcTavg_Q_C", "doy"]].values
#     obs = df["QbcTavg_T_C"].values
    
#     rf_model = RandomForestUncertaintyModel(**rf_settings)
#     rf_model.fit(X, obs, test_size=0.2)
#     rf_model.cross_val_rmse(X, obs, n_splits=5, shuffle=True)
#rf_model.compute_tau(X, obs, n_bootstrap=100, n_jobs=-2, disable=False)



