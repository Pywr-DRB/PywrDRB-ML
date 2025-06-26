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

config_template = {
    'input_data_file': "data/database/SalinityLSTM_database.csv",
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
    'weight_threshold': 80,
    'weight_value': 5,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.1,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 4
    }

lstm_settings_none = {
    "model_id": "SalinityLSTM_none",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

lstm_settings_lag1 = {
    "model_id": "SalinityLSTM_lag1",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "doy", "saltfront"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

lstm_settings_7davg = {
    "model_id": "SalinityLSTM_7d_avg",
    "x_vars": ["Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

lstm_settings_lag1_7davg = {
    "model_id": "SalinityLSTM_lag1_7d_avg",
    "x_vars": ["Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

lstm_settings_1d_7davg = {
    "model_id": "SalinityLSTM_1d_7d_avg",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

lstm_settings_lag1_1d_7davg = {
    "model_id": "SalinityLSTM_lag1_1d_7d_avg",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }
#%%
subfolder = "SalinityLSTM_comparison"
lstm_settings_list = [lstm_settings_none, lstm_settings_lag1, lstm_settings_7davg, lstm_settings_lag1_7davg, lstm_settings_1d_7davg, lstm_settings_lag1_1d_7davg]
model_ids = []
for lstm_settings in lstm_settings_list:
    lstm_config = deepcopy(config_template)
    lstm_config.update(lstm_settings)
    lstm_config_file = make_lstm_model(subfolder=subfolder, **lstm_config)
    # df = data_prep(lstm_config_file, root_dir) already included in loop_to_train_lstm_models
    model_ids.append(lstm_settings["model_id"])

#%% LSTM
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=False)

#%% Random forest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_SalinityLSTM.copy()['1979-01-01': '2023-12-31']

df = database[["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront", "saltfront_src"]].dropna()
df.loc[df["saltfront_src"] != "obs", "saltfront"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
X = df[["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"]].values
obs = df["saltfront"].values
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, obs)
sim = rf_model.predict(X)

obs = pd.Series(obs, index=df.index)
sim = pd.Series(sim, index=df.index)
df_metric_rf = clt.metrics.error_metrics(sim, obs, to_frame=True, name="rf_model_1d_7d_avg")
df_metric_rf_above = clt.metrics.error_metrics(sim[obs>=80], obs[obs>=80], to_frame=True, name="rf_model_1d_7d_avg")

df_simple = pd.DataFrame()
df_simple["mu_ft"] = sim
df_simple["sd_ft"] = np.nan
df_simple["obs"] = obs
df_simple = df_simple.reindex(database.index)
df_simple.index.name = "date"
df_simple.to_csv(pn.models.get(f"{subfolder}") / "simple_run_rf_model_1d_7d_avg.csv")

clt.io.to_joblib(rf_model, pn.models.get(f"{subfolder}") / "simple_run_rf_model_1d_7d_avg.gz")
#%% Post evaluation
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, mode="SalinityLSTM", overwrite=False)
df_metric_train = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM")
df_metric_train = pd.concat([df_metric_train, df_metric_rf])
df_metric_train_above = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM", xbound=(80, None))
df_metric_train_above = pd.concat([df_metric_train_above, df_metric_rf_above])
pairs = return_sim_obs_pair(model_ids, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM")
pairs["rf_model_1d_7d_avg"] = (sim, obs)
#%% Hist barplot
pn.mkdir(pn.models.get(f"{subfolder}") / "figures")
model_ids_none = ['SalinityLSTM_none', 'SalinityLSTM_7d_avg', 'SalinityLSTM_1d_7d_avg', "rf_model_1d_7d_avg"]
model_ids_lag1 = ['SalinityLSTM_lag1', 'SalinityLSTM_lag1_7d_avg', 'SalinityLSTM_lag1_1d_7d_avg', "rf_model_1d_7d_avg"]

fig, ax = plt.subplots()
for model_id in model_ids_none:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=40, alpha=0.5, label=model_id)
    ax.axvline(82.9, color="b", linestyle="--")
    ax.axvline(87, color="b", linestyle="-.")
    ax.axvline(92.5, color="b", linestyle=":")
ax.set_ylim([0, 5.5])
ax.set_xlim([37, 95])
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_histbarplot_no_lag.png")
plt.show()

fig, ax = plt.subplots()
for model_id in model_ids_lag1:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=40, alpha=0.5, label=model_id)
    ax.axvline(82.9, color="b", linestyle="--")
    ax.axvline(87, color="b", linestyle="-.")
    ax.axvline(92.5, color="b", linestyle=":")
ax.set_ylim([0, 5.5])
ax.set_xlim([37, 95])
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_histbarplot_lag1.png")
plt.show()

#%% error_boxplot
for model_id in model_ids + ["rf_model_1d_7d_avg"]:
    fig, ax = plt.subplots()
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    ax.axhline(0, ls="--", lw=1, c="grey", zorder=-1)
    clt.plots.error_over_obs_bins_in_boxplot(ax, obs, sim, bins=30, label=model_id)
    ax.axvline(82.9, color="b", linestyle="--")
    ax.axvline(87, color="b", linestyle="-.")
    ax.axvline(92.5, color="b", linestyle=":")
    ax.set_ylim([-15, 25])
    ax.legend()
    plt.tight_layout()
    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"error_boxplot_{model_id}.png")
    plt.show()

#%% scatter plot
fig, ax = plt.subplots()
for model_id in model_ids_none:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.scatter(ax, x=obs, y=sim, s=2, alpha=0.1, linear_regr_line=False, label=model_id)
clt.ax.add_45_degree_ref_line(ax)
ax.set_xlabel("Observed (RM)")
ax.set_ylabel("Predicted (RM)")
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "scatter_no_lag.png")
plt.show()

fig, ax = plt.subplots()
for model_id in model_ids_lag1:
    sim, obs = pairs[model_id]
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.scatter(ax, x=obs, y=sim, s=2, alpha=0.1, linear_regr_line=False, label=model_id)
clt.ax.add_45_degree_ref_line(ax)
ax.set_xlabel("Observed (RM)")
ax.set_ylabel("Predicted (RM)")
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "scatter_lag1.png")
plt.show()

#%% rmse_barplot
fig, ax = plt.subplots()

# Plot for both dataframes
x = range(len(model_ids_none))
bar_width = 0.35

ax.bar([i - bar_width/2 for i in x], df_metric_train.loc[model_ids_none, 'rmse'], width=bar_width, label='all')
ax.bar([i + bar_width/2 for i in x], df_metric_train_above.loc[model_ids_none, 'rmse'], width=bar_width, label='above 80 RM')

# X-axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(df_metric_train.loc[model_ids_none, 'rmse'].index, rotation=45, ha='right')
ax.set_ylabel('RMSE')
ax.legend()
plt.tight_layout()
ax.set_ylim([0, 4])
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_barplot_none.png")
plt.show()

fig, ax = plt.subplots()

# Plot for both dataframes
x = range(len(model_ids_lag1))
bar_width = 0.35

ax.bar([i - bar_width/2 for i in x], df_metric_train.loc[model_ids_lag1, 'rmse'], width=bar_width, label='all')
ax.bar([i + bar_width/2 for i in x], df_metric_train_above.loc[model_ids_lag1, 'rmse'], width=bar_width, label='above 80 RM')

# X-axis labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(df_metric_train.loc[model_ids_lag1, 'rmse'].index, rotation=45, ha='right')
ax.set_ylabel('RMSE')
ax.legend()
plt.tight_layout()
ax.set_ylim([0, 4])
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_barplot_lag1.png")
plt.show()
#%% Time series
for y in range(1979, 2024):
    fig, ax = plt.subplots()
    sim, obs = pairs[model_ids_none[0]]
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    for model_id in model_ids_none:
        sim, _ = pairs[model_id]
        ax.plot(sim[f"{y}":f"{y}"], alpha=0.8, label=model_id)
    ax.set_ylim([40, 100])
    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.legend()
    plt.tight_layout()
    
    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"no_lag_{y}.png")
    
    plt.show()

for y in range(1979, 2024):
    fig, ax = plt.subplots()
    sim, obs = pairs[model_ids_lag1[0]]
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    for model_id in model_ids_none:
        sim, _ = pairs[model_id]
        ax.plot(sim[f"{y}":f"{y}"], alpha=0.8, label=model_id)
    ax.set_ylim([40, 100])
    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.legend()
    plt.tight_layout()
    
    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"lag1_{y}.png")
    
    plt.show()


















