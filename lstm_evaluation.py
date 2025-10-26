#%% Run TempLSTM and evaluate
import pandas as pd
import numpy as np
import pathnavigator
from copy import deepcopy
import clt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

from src.lstm_model import WaterTempLSTMModel, SalinityLSTMModel

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']

#%%
ml_model_temp = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )


ml_model_salt = SalinityLSTMModel(
    model_salinity=pn.models.get() / r"SalinityLSTM/SalinityLSTM.yml",
    start_date='1979-01-01', end_date='2023-12-31',
    Q_Trenton_lstm_var_name="Q_Trenton_bc",
    Q_Schuylkill_lstm_var_name="Q_Schuylkill_bc",
    debug=True,
    disable_tqdm=True
    )

ml_model_temp.lstm1.calc_feature_importance()
ml_model_temp.lstm2.calc_feature_importance()
ml_model_salt.lstm.calc_feature_importance()

df_fi1 = ml_model_temp.lstm1.feat_importance
df_fi2 = ml_model_temp.lstm2.feat_importance
df_fi_salt = ml_model_salt.lstm.feat_importance

ml_model_temp.load_data(db_TempLSTM)
ml_model_temp.update_until(date=pd.Timestamp('2024-01-01'))

ml_model_salt.load_data()
ml_model_salt.update_until(date=pd.Timestamp('2024-01-01'))

#%% Plot LSTM feature importance in three subplots
# Create feature name mapping dictionary
feature_name_mapping = {
    # Temperature features
    'tmmn': '$T_{min}^{air}$',
    'tmmx': '$T_{max}^{air}$',
    'pr': '$Pr$',
    'srad': '$Srad$',
    'doy': '$doy$',

    # Reservoir/Storage features
    'bc_cannonsville_storage_pct': '$S_C$',

    # Flow features
    'QbcTavg_Q_C': '$Q_C$',
    'QbcTavg_Q_i': '$Q_i$',

    # Salinity features
    'Q_Schuylkill_bc': '$Q_S$',
    'Q_Trenton_bc': '$Q_T$',
    'Q_Schuylkill_bc_7d_avg': '$\overline{Q}_S^{7d}$',
    'Q_Trenton_bc_7d_avg': '$\overline{Q}_T^{7d}$',

    # Add more mappings as needed
}

def map_feature_names(df, mapping_dict):
    """Map feature names using the provided dictionary"""
    df_mapped = df.copy()
    df_mapped['x_var'] = df_mapped['x_var'].map(mapping_dict).fillna(df_mapped['x_var'])
    return df_mapped

fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

# Set consistent bar width for all subplots
bar_width = 0.6

# Plot df_fi1 (TempLSTM1) - sorted by delta_nll
df_fi1_sorted = df_fi1.sort_values('delta_nll', ascending=False)
df_fi1_mapped = map_feature_names(df_fi1_sorted, feature_name_mapping)
# Normalize to [0,1] range
df_fi1_norm = df_fi1_mapped.copy()
min_val, max_val = df_fi1_norm['delta_nll'].min(), df_fi1_norm['delta_nll'].max()
df_fi1_norm['delta_nll'] = (df_fi1_norm['delta_nll'] - min_val) / (max_val - min_val)
ax = axes[0]
ax.bar(df_fi1_norm['x_var'], df_fi1_norm['delta_nll'], width=bar_width, color='#0070C0')
ax.set_title('TempLSTM1')
ax.set_xlabel('Features')
ax.set_ylabel('Normalized Delta NLL')
ax.tick_params(axis='x', rotation=0)
ax.text(-0.1, 1.1, "a)", transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')


# Plot df_fi2 (TempLSTM2) - sorted by delta_nll
df_fi2_sorted = df_fi2.sort_values('delta_nll', ascending=False)
df_fi2_mapped = map_feature_names(df_fi2_sorted, feature_name_mapping)
# Normalize to [0,1] range
df_fi2_norm = df_fi2_mapped.copy()
min_val, max_val = df_fi2_norm['delta_nll'].min(), df_fi2_norm['delta_nll'].max()
df_fi2_norm['delta_nll'] = (df_fi2_norm['delta_nll'] - min_val) / (max_val - min_val)
ax = axes[1]
ax.bar(df_fi2_norm['x_var'], df_fi2_norm['delta_nll'], width=bar_width, color='#B31B1B')
ax.set_title('TempLSTM2')
ax.set_xlabel('Features')
ax.tick_params(axis='x', rotation=0)
ax.text(-0.1, 1.1, "b)", transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')


# Plot df_fi_salt (SalinityLSTM) - sorted by delta_nll
df_fi_salt_sorted = df_fi_salt.sort_values('delta_nll', ascending=False)
df_fi_salt_mapped = map_feature_names(df_fi_salt_sorted, feature_name_mapping)
# Normalize to [0,1] range
df_fi_salt_norm = df_fi_salt_mapped.copy()
min_val, max_val = df_fi_salt_norm['delta_nll'].min(), df_fi_salt_norm['delta_nll'].max()
df_fi_salt_norm['delta_nll'] = (df_fi_salt_norm['delta_nll'] - min_val) / (max_val - min_val)
ax = axes[2]
ax.bar(df_fi_salt_norm['x_var'], df_fi_salt_norm['delta_nll'], width=bar_width, color='slateblue')
ax.set_title('SalinityLSTM')
ax.set_xlabel('Features')
ax.tick_params(axis='x', rotation=0)
ax.text(-0.1, 1.1, "c)", transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')

# Adjust layout to prevent overlap
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "feature_importance.jpg")
plt.show()

#%% RSME barplot and timeseries
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5,5))

ax = axes[0, 0]
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

ax = axes[1, 0]
sim = ml_model_temp.records["T_L_mu"]
obs = db_TempLSTM.copy()
obs.loc[obs['tmmx_water_src'] != "obs", 'QbcTmax_T_L'] = np.nan
obs = obs['QbcTmax_T_L'].values
df = pd.DataFrame({"obs": obs, "sim": sim}, index=ml_model_temp.dates)
year = 2019
df = df.loc[f"{year}-01-01":f"{year}-12-31", :]
ax.plot(df["obs"], ls='None', marker='o', color="k", alpha=0.2, ms=3, label="obs")
ax.plot(df["sim"], color='salmon', lw=1, label="sim")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.legend(frameon=False)
custom_ticks = pd.to_datetime([f"{year}-01-01", f"{year}-04-01", f"{year}-07-01", f"{year}-10-01", f"{year}-12-31"])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])  # show only month/day
ax.set_xlim([df.index[0], df.index[-1]])
ax.set_ylabel("$T_{max}$ (°C)")
ax.set_xlabel(f"Date ({year})")

ax = axes[0, 1]
sim = ml_model_salt.records["sf_mu"]
obs = db_SalinityLSTM.copy()
obs.loc[obs['saltfront_src'] != "obs", 'saltfront'] = np.nan
obs = obs['saltfront'].values
sim, obs = clt.dropna_any(sim, obs)
clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=(60, 90, 5), color="mediumpurple")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.set_ylabel(f"RMSE\n(Overall: {round(clt.metrics.rmse(sim, obs), 2)})")
ax.set_xlabel("$Saltfront$ (RM)")

# ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper right')
# clt.plots.rmse_over_obs_bins_in_barplot(ax_inset, obs, sim, bins=(80, 90, 5), color="mediumpurple")
# ax_inset.grid(True, axis='y', lw=0.3, ls="--")
# ax_inset.set_ylim(0, 2.5)
# ax_inset.set_ylabel("")
# ax_inset.set_xlabel("")

ax = axes[1, 1]
sim = ml_model_salt.records["sf_mu"]
obs = db_SalinityLSTM.copy()
obs.loc[obs['saltfront_src'] != "obs", 'saltfront'] = np.nan
obs = obs['saltfront'].values
df = pd.DataFrame({"obs": obs, "sim": sim}, index=ml_model_salt.dates)
year = 2007
df = df.loc[f"{year}-01-01":f"{year}-12-31", :]
ax.plot(df["obs"], ls='None', marker='o', color="k", alpha=0.2, ms=3, label="obs")
ax.plot(df["sim"], color='mediumpurple', lw=1, label="sim")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.legend(frameon=False)
custom_ticks = pd.to_datetime([f"{year}-01-01", f"{year}-04-01", f"{year}-07-01", f"{year}-10-01", f"{year}-12-31"])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])  # show only month/day

ax.set_xlim([df.index[0], df.index[-1]])
ax.set_ylim([60, 88])
ax.set_ylabel("$Saltfront$ (RM)")
ax.set_xlabel(f"Date ({year})")

# Add subplot labels
axes[0, 0].text(-0.3, 1, "a)", transform=axes[0, 0].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.3, 1, "b)", transform=axes[1, 0].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.3, 1, "c)", transform=axes[0, 1].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.3, 1, "d)", transform=axes[1, 1].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')

plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "rmse_barplot_and_ts.jpg", dpi=500)
plt.show()


#%% Plot with historical thermal release

# Simulate with historical releases
ml_model_hist = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620,  # mgd
    )
ml_model_hist.load_data(db_TempLSTM)

thermal_releases = db_TempLSTM[['rel_thermal']].fillna(0).values.flatten()

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for date in dates:
    # Update data in the ml_model_hist for the next step(s) model update.
    t = ml_model_hist.t
    thermal_release = thermal_releases[t]
    Q_C = ml_model_hist.Q_C[t] + thermal_releases[t]

    # Record
    ml_model_hist.remained_bank_amount -= thermal_release
    ml_model_hist.records["thermal_releases"][ml_model_hist.t] = thermal_release
    ml_model_hist.records["remained_bank_amounts"][ml_model_hist.t] = ml_model_hist.remained_bank_amount
    ml_model_hist.update(t=t, Q_C=Q_C)
# Update the model until the end of the simulation period
ml_model_hist.update_until(date="2024-01-01")


#%%
sim_no_tr = ml_model_temp.records["T_L_mu"]
sim_tr = ml_model_hist.records["T_L_mu"]
obs_no_tr = db_TempLSTM.copy()
obs_no_tr.loc[obs_no_tr['tmmx_water_src'] != "obs", 'QbcTmax_T_L'] = np.nan
obs_no_tr = obs_no_tr['QbcTmax_T_L'].values
obs_tr = db_TempLSTM.copy()
obs_tr.loc[obs_tr['tmmx_water_src'] != "obs", 'QobsTmax_T_L'] = np.nan
obs_tr = obs_tr['QobsTmax_T_L'].values

df = pd.DataFrame({"obs_no_tr": obs_no_tr, "obs_tr": obs_tr, "sim_no_tr": sim_no_tr, "sim_tr": sim_tr, "thermal_release": thermal_releases}, index=ml_model_temp.dates)


yr = 2019
for yr in range(2006, 2023):
    df_ = df.loc[f"{yr}-6-1":f"{yr}-8-31", :]

    fig, ax = plt.subplots()
    ax.plot(df_["obs_no_tr"], ls='--', marker='o', color="r", alpha=1, ms=3, label="obs_no_tr")
    ax.plot(df_["obs_tr"], ls=':', marker='x', color="b", alpha=1, ms=3, label="obs_tr")
    ax.plot(df_["sim_no_tr"], color='salmon', lw=2, label="sim_no_tr")
    ax.plot(df_["sim_tr"], color='dodgerblue', lw=2, label="sim_tr")
    ax.grid(True, axis='y', lw=0.3, ls="--")
    ax.legend(frameon=False)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks[::3])
    ax.set_xlim([df_.index[0], df_.index[-1]])
    ax.set_ylabel("$T_{max}$ (°C)")
    ax.set_xlabel("Date")

    ax2 = ax.twinx()
    ax2.bar(df_.index, df_["thermal_release"], width=1.0, color='blue', label="Thermal Release")
    ax2.set_ylabel("Thermal release (mgd)")
    ax2.set_ylim([0, max(df_["thermal_release"])*3])

    plt.tight_layout()
    clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "compare_with_hist_tr.jpg")
    plt.show()