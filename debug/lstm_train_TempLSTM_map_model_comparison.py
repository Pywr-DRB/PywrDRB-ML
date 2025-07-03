#%% Imports
import pathnavigator
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import clt

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair
# from src.prep_data import data_prep

import pandas as pd
from src.lstm_model import WaterTempLSTMModel

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_TempLSTM.copy()['1979-01-01': '2023-12-31']

df = database[["QbcTavg_T_L", "QbcTmax_T_L", "tavg_water_lordville_src", "tmmx_water_lordville_src"]].dropna()
df.loc[df["tavg_water_lordville_src"] != "obs", "QbcTavg_T_L"] = np.nan
df.loc[df["tmmx_water_lordville_src"] != "obs", "QbcTmax_T_L"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
x = df["QbcTavg_T_L"].values
y = df["QbcTmax_T_L"].values


# Linear mapping would be sufficient
Tavg2Tmax_coefs = WaterTempLSTMModel.fit_linear_regression(x, y)
WaterTempLSTMModel.tavg2tmax(x, Tavg2Tmax_coefs)
WaterTempLSTMModel.tavg2tmax(18, Tavg2Tmax_coefs)
clt.io.to_json(Tavg2Tmax_coefs, pn.models.get("TempLSTM") / "Tavg2Tmax_coefs.json")
#%%
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
    'learn_rate_pre': 0.05,
    'learn_rate_fine': 0.05,
    'n_epochs_pre': 50,
    'n_epochs_fine': 350,
    'early_stopping_patience': 50,
    'hidden_units': 16,
    'head_hidden_units': 16,
    'head_n_distr': 1,
    'weight_loss': True,
    'weight_threshold': 20,
    'weight_value': 2,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.1,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 4
    }

lstm1_settings = {

    }

TempLSTM_map = {
    "model_id": "TempLSTM_map",
    "x_vars": ["QbcTavg_T_L", "doy"],
    "y_vars": ["QbcTmax_T_L"],
    "y_vars_src": ["tavg_water_src"],
    }

TempLSTM_map_srad = {
    "model_id": "TempLSTM_map_srad",
    "x_vars": ["QbcTavg_T_L", "srad", "doy"],
    "y_vars": ["QbcTmax_T_L"],
    "y_vars_src": ["tavg_water_src"],
    }

TempLSTM_map_QiQc = {
    "model_id": "TempLSTM_map_QiQc",
    "x_vars": ["QbcTavg_T_C", "QbcTavg_T_i","QbcTavg_T_L", "doy"],
    "y_vars": ["QbcTmax_T_L"],
    "y_vars_src": ["tavg_water_src"],
    }

TempLSTM_map_srad_QiQc = {
    "model_id": "TempLSTM_map_srad_QiQc",
    "x_vars": ["QbcTavg_T_C", "QbcTavg_T_i","QbcTavg_T_L", "srad", "doy"],
    "y_vars": ["QbcTmax_T_L"],
    "y_vars_src": ["tavg_water_src"],
    }

#%%
subfolder = "TempLSTM_map_comparison"
lstm_settings_list = [TempLSTM_map] #, TempLSTM_map_srad, TempLSTM_map_QiQc, TempLSTM_map_srad_QiQc]
model_ids = []
for lstm_settings in lstm_settings_list:
    lstm_config = deepcopy(config_template)
    lstm_config.update(lstm_settings)
    lstm_config_file = make_lstm_model(subfolder=subfolder, **lstm_config)
    # df = data_prep(lstm_config_file, root_dir) already included in loop_to_train_lstm_models
    model_ids.append(lstm_settings["model_id"])

#%% LSTM
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=False)

#%% Post evaluation
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, mode="TempLSTM", overwrite=False)
df_metric_train = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="TempLSTM")
df_metric_train_above = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="TempLSTM", xbound=(20, None))
pairs = return_sim_obs_pair(model_ids, subfolder=subfolder, period="train", only_months=None, mode="TempLSTM")
#%% Hist barplot
pn.mkdir(pn.models.get(f"{subfolder}") / "figures")
fig, ax = plt.subplots()
for model_id in model_ids:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=40, alpha=0.5, label=model_id)
    ax.axvline(20, color="b", linestyle="--")
    ax.axvline(24, color="b", linestyle="-.")
ax.set_ylim([0, 5.5])
ax.set_xlim([0, 30])
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_histbarplot.png")
plt.show()

#%% error_boxplot
for model_id in model_ids:
    fig, ax = plt.subplots()
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    ax.axhline(0, ls="--", lw=1, c="grey", zorder=-1)
    clt.plots.error_over_obs_bins_in_boxplot(ax, obs, sim, bins=30, label=model_id)
    ax.axvline(20, color="b", linestyle="--")
    ax.axvline(24, color="b", linestyle="-.")
    ax.set_ylim([-15, 25])
    ax.legend()
    plt.tight_layout()
    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"error_boxplot_{model_id}.png")
    plt.show()

#%% scatter plot
fig, ax = plt.subplots()
for model_id in [model_ids[0]]:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.scatter(ax, x=obs, y=sim, s=2, alpha=0.3, linear_regr_line=False, label=model_id)
clt.ax.add_45_degree_ref_line(ax)
ax.set_xlabel("Observed (degC)")
ax.set_ylabel("Predicted (degC)")
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "scatter.png")
plt.show()

#%% rmse_barplot
fig, ax = plt.subplots()

# Plot for both dataframes
x = range(len(model_ids))
bar_width = 0.35

ax.bar([i - bar_width/2 for i in x], df_metric_train.loc[model_ids, 'rmse'], width=bar_width, label='all')
ax.bar([i + bar_width/2 for i in x], df_metric_train_above.loc[model_ids, 'rmse'], width=bar_width, label='above 20 degC')

# X-axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(df_metric_train.loc[model_ids, 'rmse'].index, rotation=45, ha='right')
ax.set_ylabel('RMSE')
ax.legend()
plt.tight_layout()
ax.set_ylim([0, 4])
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_barplot.png")
plt.show()

#%% Time series
for y in range(1979, 2024):
    fig, ax = plt.subplots()
    sim, obs = pairs[model_ids[0]]
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    for model_id in model_ids:
        sim, _ = pairs[model_id]
        ax.plot(sim[f"{y}":f"{y}"], alpha=0.8, label=model_id)
    ax.set_ylim([0, 30])
    ax.set_xlabel("Date")
    ax.set_ylabel("Tavg_C (degC)")
    ax.legend()
    plt.tight_layout()

    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"lstm_map_{y}.png")

    plt.show()

#%%
for y in range(1979, 2024):
    fig, ax = plt.subplots()
    sim, obs = pairs[model_ids[0]]
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    for model_id in [model_ids[0]]:
        sim, _ = pairs[model_id]
        ax.plot(sim[f"{y}":f"{y}"], alpha=0.8, label=model_id)
    ax.set_ylim([0, 30])
    ax.set_xlabel("Date")
    ax.set_ylabel("Tavg_C (degC)")
    ax.legend()
    plt.tight_layout()

    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"lstm_map_{y}.png")

    plt.show()

#%%
