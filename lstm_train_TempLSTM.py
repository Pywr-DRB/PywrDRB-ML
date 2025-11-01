#%%
import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()
from src.model_builder import (
    make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models,
    loop_to_eval_lstm_models, eval_TempLSTM, return_sim_obs_pair_for_T_L
    )

config_template = {
    'input_data_file': "data/database/TempLSTM_database.csv",
    'x_vars': [],
    'y_vars': [],
    'y_vars_src': [],
    'lag_days': 1,
    'min_date': '1979-01-01',
    'max_date': '2023-12-31',
    'start_date_train': '1979-01-01',
    'end_date_train': '2023-12-31',
    'start_date_val': '2017-01-01',
    'end_date_val': '2017-12-31',
    'start_date_test': '2017-01-01',
    'end_date_test': '2017-12-31',
    'pre_train': True,
    'fine_tune': True,
    'n_epochs_pre': 50,
    'n_epochs_fine': 350,
    'hidden_units': 16,
    'head_hidden_units': 16,
    'head_n_distr': 1,
    'weight_loss': True,
    'weight_threshold': 20,
    'weight_value': 5,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 2,
    }
lstm1_settings = {
    "model_id": "TempLSTM1",
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_C"],
    "y_vars_src": ["tavg_water_cannonsville_src"],
    'learn_rate_pre': 0.05,
    'learn_rate_fine': 0.05,
    'early_stopping_patience': 50,
    'dropout_rate': 0,
    }
# 50 0.05 0 2
lstm2_settings = {
    "model_id": "TempLSTM2",
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "QbcTavg_Q_i", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_i"],
    "y_vars_src": ["tavg_water_src"],
    'learn_rate_pre': 0.05,
    'learn_rate_fine': 0.05,
    'early_stopping_patience': 50,
    'dropout_rate': 0,
    }
#%% Create config file
subfolder = "TempLSTM"
lstm1_config = deepcopy(config_template)
lstm1_config.update(lstm1_settings)
lstm2_config = deepcopy(config_template)
lstm2_config.update(lstm2_settings)

lstm1_config_file = make_lstm_model(subfolder=subfolder, **lstm1_config)
lstm2_config_file = make_lstm_model(subfolder=subfolder, **lstm2_config)

#%% Train model and output simple run results
model_ids = ["TempLSTM1", "TempLSTM2"]
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=True)
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=True)
df_metric_train = loop_to_eval_lstm_models(lstms, period="train", subfolder=subfolder, only_months=None, mode="TempLSTM")

#               nrmse         r        r2      rmse
# model_id                                         
# TempLSTM1  0.060459  0.930463  0.862872  1.463116
# TempLSTM2  0.012391  0.931999  0.867740  4.167066

#               nrmse         r        r2      rmse
# model_id                                         
# TempLSTM1  0.064629  0.919808  0.843206  1.564033
# TempLSTM2  0.012347  0.932295  0.868736  4.152256

#               nrmse         r        r2      rmse
# model_id                                         
# TempLSTM1  0.065054  0.923122  0.842484  1.574299
# TempLSTM2  0.012621  0.929311  0.862777  4.244475


# #%%
# df_metric = eval_TempLSTM(
#     lstm1=lstms["TempLSTM1"],
#     lstm2=lstms["TempLSTM2"],
#     period="all", only_months=None, disable=False
#     )


# df_metric_summer = eval_TempLSTM(
#     lstm1=lstms["TempLSTM1"],
#     lstm2=lstms["TempLSTM2"],
#     period="all", only_months=[6,7,8], disable=False
#     )

# #%%
# import clt
# df_obs, df_sim = return_sim_obs_pair_for_T_L(lstm1=lstms["TempLSTM1"], lstm2=lstms["TempLSTM2"])

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(4, 4))
# x = df_obs["T_L"]
# y = df_sim["T_L"]
# x, y = clt.utils.dropna_any(x, y)
# clt.plots.scatter(ax, x=x, y=y, s=2)
# ax.set_xlabel("Observed Tmax at Lordville")
# ax.set_ylabel("Predicted Tmax at Lordville")
# ax.legend()
# plt.tight_layout()
# plt.show()

#%%
import numpy as np
import pandas as pd
import clt

import matplotlib.pyplot as plt
from src.lstm_model import WaterTempLSTMModel

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']
ml_model_temp = WaterTempLSTMModel(
    model1=pn.models.get() / f"{subfolder}/TempLSTM1.yml",
    model2=pn.models.get() / f"{subfolder}/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model_temp.load_data(db_TempLSTM)
ml_model_temp.update_until(date=pd.Timestamp('2024-01-01'))

fig, ax = plt.subplots()
sim = ml_model_temp.records["T_L_mu"]
obs = db_TempLSTM.copy()
obs.loc[obs['tmmx_water_src'] != "obs", 'QbcTmax_T_L'] = np.nan
obs = obs['QbcTmax_T_L'].values
sim, obs = clt.dropna_any(sim, obs)

# For Tavg_L_mu
sim_Tavg_L_mu = ml_model_temp.records["Tavg_L_mu"]
obs_Tavg_L_mu = db_TempLSTM.copy()
obs_Tavg_L_mu.loc[obs_Tavg_L_mu['tavg_water_src'] != "obs", 'QbcTavg_T_L'] = np.nan
obs_Tavg_L_mu = obs_Tavg_L_mu['QbcTavg_T_L'].values
sim_Tavg_L_mu, obs_Tavg_L_mu = clt.dropna_any(sim_Tavg_L_mu, obs_Tavg_L_mu)
round(clt.metrics.rmse(sim_Tavg_L_mu, obs_Tavg_L_mu), 2)

clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=(0, 30, 5), color="salmon")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.set_ylabel(f"RMSE\n(Overall: {round(clt.metrics.rmse(sim, obs), 2)})")
ax.set_xlabel("$T_{max}$ (°C)")

#%% Compare across weights 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse_over_obs_bins_in_barplot(
    ax, obs, sim, bins=10, align='center',
    labels=None, colors=None, empty_bin_value=np.nan, **kwargs
):
    """
    Plot RMSE against observed values in bins. If `sim` is a pandas DataFrame,
    draw grouped bars (one bar per column) within each bin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    obs : array-like
        Observed values.
    sim : array-like or pandas.DataFrame
        Simulated values. If DataFrame, columns are separate series to compare.
    bins : int or tuple
        Either number of bins (int), or a tuple (start, end, step).
    align : {'center','edge'}
        Bar alignment (default 'center').
    labels : list of str, optional
        Labels for legend when `sim` is DataFrame. Defaults to sim.columns.
    colors : list of color, optional
        Colors per series when `sim` is DataFrame.
    empty_bin_value : float
        Value to use when a bin has no samples (default np.nan).
    **kwargs :
        Passed to `ax.bar` (e.g., alpha).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'bin_edges': ndarray of bin edges
        - 'bin_centers': ndarray of bin centers
        - 'rmse': ndarray (S x B) for DataFrame input, or (B,) for 1-D input
        - 'handles': list of BarContainers (for legend), else []
    """
    obs = np.asarray(obs)

    # Build bin edges
    if isinstance(bins, tuple) and len(bins) == 3:
        start, end, step = bins
        bin_edges = np.arange(start, end + step, step)
    else:
        bin_edges = np.linspace(np.nanmin(obs), np.nanmax(obs), num=int(bins) + 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Digitize observed into bins
    bin_idx = np.digitize(obs, bin_edges) - 1
    # clamp to [0, n_bins-1]
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)

    # Helper to compute per-bin RMSE for one simulated series
    def per_bin_rmse(sim_vec):
        sim_vec = np.asarray(sim_vec)
        se = (obs - sim_vec) ** 2
        rmse = np.empty(len(bin_centers), dtype=float)
        for i in range(len(bin_centers)):
            m = bin_idx == i
            if np.any(m):
                rmse[i] = np.sqrt(np.nanmean(se[m]))
            else:
                rmse[i] = empty_bin_value
        return rmse

    handles = []

    if isinstance(sim, pd.DataFrame):
        series_list = [sim[c].to_numpy() for c in sim.columns]
        n_series = len(series_list)
        labels = labels if labels is not None else list(sim.columns)

        # Bar width per series inside each bin (with a small group padding)
        group_pad = 0.1
        bar_widths = (bin_widths * (1 - group_pad)) / max(n_series, 1)

        rmse_mat = np.vstack([per_bin_rmse(sv) for sv in series_list])  # (S, B)

        for j in range(n_series):
            overall_rmse = round(clt.metrics.rmse(series_list[j], obs), 2)
            # Offset each series j around the bin center
            offsets = (-0.5 * (n_series - 1) + j) * bar_widths
            # Compute x positions per bin
            x = bin_centers + offsets

            h = ax.bar(
                x, rmse_mat[j],
                width=bar_widths,
                align='center',
                label=labels[j] + f" (RMSE = {overall_rmse})",
                color=(None if colors is None else colors[j] if j < len(colors) else None),
                **kwargs
            )
            handles.append(h)

        ax.set_xlabel("Observed-value bins")
        ax.set_ylabel("RMSE")
        ax.legend(
            loc='upper center',          # position on top
            bbox_to_anchor=(0.5, 1.18),  # center it horizontally, move slightly above the plot
            ncol=2,                      # two columns
            frameon=False,               # no frame
            fontsize=10                  # optional: adjust font size
        )
        out_rmse = rmse_mat

    else:
        # 1D case
        rmse_vals = per_bin_rmse(sim)
        # default width ~ 90% of bin
        widths = bin_widths * 0.9
        ax.bar(bin_centers, rmse_vals, width=widths, align='center', **kwargs)
        ax.set_xlabel("Observed-value bins")
        ax.set_ylabel("RMSE")
        out_rmse = rmse_vals

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'rmse': out_rmse,
        'handles': handles
    }

import matplotlib.pyplot as plt
db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']
sim_df = pd.DataFrame()
for w in [1, 2, 5, 10]:
    ml_model_temp = WaterTempLSTMModel(
        model1=pn.models.get() / f"TempLSTM_w{w}/TempLSTM1.yml",
        model2=pn.models.get() / f"TempLSTM_w{w}/TempLSTM2.yml",
        Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
        debug=True,
        thermal_mitigation_bank_size=1620 * 3,  # mgd
        )
    ml_model_temp.load_data(db_TempLSTM)
    ml_model_temp.update_until(date=pd.Timestamp('2024-01-01'))

    sim = ml_model_temp.records["T_L_mu"]
    
    obs = db_TempLSTM.copy()
    obs.loc[obs['tmmx_water_src'] != "obs", 'QbcTmax_T_L'] = np.nan
    obs = obs['QbcTmax_T_L'].values
    sim, obs = clt.dropna_any(sim, obs)
    
    sim_df[f"weight {w}"] = sim

fig, ax = plt.subplots()
rmse_over_obs_bins_in_barplot(ax, obs, sim_df, bins=(0, 30, 5), colors=["#4477AA", "#EE6677", "#228833", "#CCBB44"])
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.set_xlabel("$T_{max}$ (°C)")
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "TempLSTM_rmse_barplot_comparising_weights.jpg", dpi=500)












