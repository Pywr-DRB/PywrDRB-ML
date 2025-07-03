import pathnavigator
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import clt

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair, get_rf_model
# from src.prep_data import data_prep

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)

def return_T_L(T_C, T_i, prefix="QbcTavg", map_to_Tmax=True, src_col="tavg_water_src"):
    T_C = T_C.to_frame("T_C")
    T_i = T_i.to_frame("T_i")
    
    global db_TempLSTM
    database = db_TempLSTM.copy()
    
    Q_C = database.loc[T_C.index, prefix + "_Q_C"].to_frame("Q_C")
    Q_i = database.loc[T_i.index, prefix + "_Q_i"].to_frame("Q_i")
    Q_L = database.loc[T_C.index, prefix + "_Q_L"].to_frame("Q_L")
    
    Tavg = (T_C["T_C"]*Q_C["Q_C"] + T_i["T_i"]*Q_i["Q_i"])/Q_L["Q_L"]
    Tavg = Tavg.to_frame("Tavg")
    
    T_L_src = database.loc[Tavg.index, [src_col]]
    
    if map_to_Tmax:
        rf_model = get_rf_model()
        Tmax = rf_model.predict(Tavg.values.reshape(-1, 1))
        Tavg["T_L"] = Tmax
        Tavg.loc[T_L_src[src_col] != "obs", "T_L"] = np.nan
    df = pd.concat([T_C, T_i, Tavg], axis=1)
    return df

subfolder1 = "TempLSTM1_comparison"
model1_ids = ['TempLSTM1_none', 'TempLSTM1_Qc', 'TempLSTM1_lag1', 'TempLSTM1_lag1_Qc', 'rf_model_Qc', 'rf_model']
simple_runs1 = {}
for model_id in model1_ids:
    print(model_id)
    df = pd.read_csv(pn.models.get(f"{subfolder1}/simple_run_{model_id}.csv"), index_col=0, parse_dates=True)
    simple_runs1[model_id] = df

subfolder2 = "TempLSTM2_comparison"
model2_ids = ['TempLSTM2_none', 'TempLSTM2_Qc', 'TempLSTM2_lag1', 'TempLSTM2_lag1_Qc', 'rf_model', 'rf_model_noQ']

simple_runs2 = {}
for model_id in model2_ids:
    df = pd.read_csv(pn.models.get(f"{subfolder2}/simple_run_{model_id}.csv"), index_col=0, parse_dates=True)
    simple_runs2[model_id] = df

model_pairs = {
    "TempLSTM_none": ("TempLSTM1_none", "TempLSTM2_none"),
    "TempLSTM_Qc": ("TempLSTM1_Qc", "TempLSTM2_Qc"),
    "TempLSTM_lag1": ("TempLSTM1_lag1", "TempLSTM2_lag1"),
    "TempLSTM_lag1_Qc": ("TempLSTM1_lag1_Qc", "TempLSTM2_lag1_Qc"),
    "rf_model": ("rf_model_Qc", "rf_model"),
    "rf_model_noQ": ("rf_model", "rf_model_noQ")
    }

dfs = {}
for model_id, (model_id1, model_id2) in model_pairs.items():
    df1 = simple_runs1[model_id1]
    df2 = simple_runs2[model_id2]
    df = return_T_L(df1["mu_ft"], df2["mu_ft"], prefix="QbcTavg", map_to_Tmax=True, src_col="tavg_water_src")
    dfs[model_id] = df

df_obs = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
df_obs.loc[df_obs["tavg_water_cannonsville_src"] != "obs", "QbcTavg_T_C"] = np.nan
df_obs.loc[df_obs["tavg_water_src"] != "obs", "QbcTavg_T_i"] = np.nan
df_obs.loc[df_obs["tavg_water_src"] != "obs", "QbcTavg_T_L"] = np.nan
df_obs.loc[df_obs["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan
df_obs = df_obs.loc[df.index, ["QbcTavg_T_i", "QbcTavg_T_C", "QbcTavg_T_L", "QbcTmax_T_L"]]
df_obs.columns = ["T_i", "T_C", "Tavg", "T_L"]

def cal_metrics(df, df_obs):
    df = df.copy()
    df_obs = df_obs.copy().loc[df.index, :]
    
    df_met = []
    df_met_above = []
    for col in ["T_i", "T_C", "Tavg", "T_L"]:
        obs = df_obs[col]
        sim = df[col]
        obs, sim = clt.utils.dropna_any(obs, sim)
        df_metric = clt.metrics.error_metrics(sim, obs, to_frame=True, name=col)
        df_metric_above = clt.metrics.error_metrics(sim[obs>=20], obs[obs>=20], to_frame=True, name=col)
        df_met.append(df_metric)
        df_met_above.append(df_metric_above)
    df_met = pd.concat(df_met, axis=0)
    df_met_above = pd.concat(df_met_above, axis=0)
    return df_met, df_met_above
    
df_metric = pd.DataFrame()
df_metric_above = pd.DataFrame()
for model_id, (model_id1, model_id2) in model_pairs.items():
    df = dfs[model_id]
    df_met, df_met_above = cal_metrics(df, df_obs)
    df_metric[model_id] = df_met["rmse"]
    df_metric_above[model_id] = df_met_above["rmse"]



#%%
subfolder = "TempLSTM_comparison"
pn.mkdir(pn.models.get() / subfolder / "figures")
# Plot grouped bar plot
fig, ax = plt.subplots(figsize=(6, 4))
df_metric.plot(kind='bar', ax=ax)
ax.set_title("1979-2023")
ax.set_ylabel("RMSE (degC)")
ax.legend(title="Model")
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
plt.xticks(rotation=0)
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_barplot_all_period.png")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4))
ax = df_metric_above.plot(kind='bar', ax=ax)
ax.set_title("1979-2023 when observed temp. >= 20 degC")
ax.set_ylabel("RMSE (degC)")
ax.legend(title="Model")
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
plt.xticks(rotation=0)
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_barplot_critical.png")
plt.show()

#%% Hist barplot
pn.mkdir(pn.models.get(f"{subfolder}") / "figures")
model_ids = ['rf_model', 'rf_model_noQ'] # 'TempLSTM_none', 'TempLSTM_Qc', 'TempLSTM_lag1', 'TempLSTM_lag1_Qc', 

fig, ax = plt.subplots()
for model_id in model_ids:
    df = dfs[model_id]
    sim = df["T_L"].copy()
    obs = df_obs["T_L"].copy()
    sim, obs = clt.dropna_any(sim, obs)
    clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=40, alpha=0.5, label=model_id)
ax.axvline(20, color="b", linestyle="--")
ax.axvline(24, color="b", linestyle="-.")
ax.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines
ax.set_ylim([0, 2])
ax.set_xlim([0, 33])
ax.legend(loc="upper left")
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / "rmse_histbarplot.png")
plt.show()

#%% Time series
for y in range(1979, 2024):
    fig, ax = plt.subplots()
    obs = df_obs["T_L"]
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    for model_id in model_ids:
        df = dfs[model_id]
        sim = df["T_L"]
        ax.plot(sim[f"{y}":f"{y}"], alpha=0.8, label=model_id)
    ax.set_ylim([-30, 50])
    ax.set_xlabel("Date")
    ax.set_ylabel("Tavg_C (degC)")
    ax.legend()
    plt.tight_layout()
    
    clt.fig.savefig(fig, filename=pn.models.get(f"{subfolder}/figures") / f"T_L_{y}.png")
    
    plt.show()







