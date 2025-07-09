import pandas as pd
import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.mkdir("outputs/coupled_pywrdrb")
pn.sc.add("wd", pn.get("outputs/coupled_pywrdrb"), overwrite=True)

import pywrdrb

#%% Create coupled Pywr-DRB model
inflow_type = 'pub_nhmv10_BC_withObsScaled'
model_filename = str(pn.sc.wd / f"{inflow_type}.json")
output_filename = str(pn.sc.wd / f"{inflow_type}.hdf5")

temp_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": str(pn.get()),
    "model1": str(pn.models.get() / r"TempLSTM1_comparison\TempLSTM1_Qc.yml"),
    "model2": str(pn.models.get() / r"TempLSTM2_comparison\TempLSTM2_Qc.yml"),
    "Tavg2Tmax_coefs": str(pn.get() / "models/TempLSTM/Tavg2Tmax_coefs.json"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "activate_thermal_control": False,
    "Q_C_lstm_var_name": "QbcTavg_Q_C",
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "thermal_mitigation_bank_size": 1620,  # mgd
    "asycronized_update": False,
    "debug": True
    }

salinity_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": pn.get_str(),
    "model_salinity": str(pn.models.get() / r"SalinityLSTM_comparison\SalinityLSTM_1d_7d_avg.yml"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "asycronized_update": False,
    "debug": True
    }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type,
    start_date="1979-01-01",
    end_date="2023-12-31",
    options={
        "temperature_model": temp_options,
        "salinity_model": salinity_options,
        }
    )

mb.make_model()
mb.write_model(model_filename)

#%% Load the model and run it
model = pywrdrb.Model.load(str(model_filename))
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)
stats = model.run()

#%% Load the output data
data = pywrdrb.Data()
results_sets = [
    'temperature',
    'salinity',
    'mrf_targets',
    'major_flow',
    "ffmp_level_boundaries",
    "res_level"
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

df_temperature = data.temperature[inflow_type][0]
df_salinity = data.salinity[inflow_type][0]

df_mrf_targets = data.mrf_targets[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]
df_ffmp = data.ffmp_level_boundaries[inflow_type][0]

df_res_level = data.res_level[inflow_type][0]

#%% Load obs for plotting
import matplotlib.pyplot as plt
import numpy as np
df_obs_temp = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_temp.loc[df_obs_temp["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan

df_obs_salinity = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_salinity.loc[df_obs_salinity["saltfront_src"] != "obs", "saltfront"] = np.nan

#%% Plot the results
for year in range(2006, 2024):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    fig, axs = plt.subplots(nrows=2, figsize=(8, 5), sharex=True)

    ax = axs[0]
    # ax.fill_between(
    #     df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"].index,
    #     df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"] - df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_sd"],
    #     df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"] + df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_sd"],
    #     color="royalblue", alpha=0.15, label="±1 sd")

    ax.plot(df_obs_temp.loc[start_date:end_date,"QbcTmax_T_L"], label="obs", color="k")
    ax.plot(df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"], label="coupled", color="royalblue", alpha=0.8)

    ax.set_ylabel("Maximum water temp.\nat Lordville (°C)")
    ax.set_ylim([-5, 30])
    ax.legend(loc="upper left", frameon=False)

    ax = axs[1]
    ax.plot(df_obs_salinity.loc[start_date:end_date,"saltfront"], label="obs", color="k")
    ax.plot(df_salinity.loc[start_date:end_date,"salt_front_location_mu"], label="coupled", color="r", alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.set_ylim([40, 100])
    ax.legend(loc="upper left", frameon=False)
    plt.show()



#%%
import clt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

df = pd.DataFrame()
df["Water Tmax (degC)"] = df_temperature.loc["2007":"2023", "temperature_after_thermal_release_mu"]
df["Salt front (RM)"] = df_salinity.loc["2007":"2023", "salt_front_location_mu"]
df["obs_T_C"] = df_obs_temp["QbcTmax_T_L"]
df["obs_saltfront"] = df_obs_salinity["saltfront"]
df["nyc_zone"] = df_res_level["nyc"]
df["zone"] = "flood"
df.loc[df["nyc_zone"] == 5, "zone"] = "drought emergency"
df.loc[df["nyc_zone"] == 4, "zone"] = "drought watch"
df.loc[df["nyc_zone"] == 3, "zone"] = "drought warning"
df.loc[df["nyc_zone"] == 2, "zone"] = "normal"

df["zone"] = pd.Categorical(df["zone"], categories=[
    "drought emergency", "drought watch", "drought warning", "normal", "flood"
])

# Define custom color map
zone_color_map = {
    "drought emergency": "#8B0000",  # dark red
    "drought watch": "#FF0000",      # red
    "drought warning": "mistyrose", #"#FFA07A",    # light red (light salmon)
    "normal": "#D3D3D3",             # light gray
    "flood": "royalblue"             # royal blue
}

# Reorder DataFrame to control z-order
zorder_order = ["normal","drought warning","flood","drought watch","drought emergency"]
df = df.set_index("zone").loc[zorder_order].reset_index()

# Map zone to colors
zone_colors = df["zone"].map(zone_color_map)

# Set seaborn style
sns.set(style="white")

# Create figure and grid spec
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(10, 10, hspace=0.0, wspace=0.0)

# Define axes
ax_joint = fig.add_subplot(gs[1:-1, 0:-1])
ax_marg_x = fig.add_subplot(gs[0, 0:-1], sharex=ax_joint)
ax_marg_y = fig.add_subplot(gs[1:-1, -1], sharey=ax_joint)

# Scatter plot
sc = ax_joint.scatter(
    df["Salt front (RM)"],
    df["Water Tmax (degC)"],
    c=zone_colors,
    s=10,
    alpha=0.6
)

# KDE plots
sns.kdeplot(df["Salt front (RM)"], ax=ax_marg_x, fill=True, color="mediumpurple")
sns.kdeplot(df["Water Tmax (degC)"], ax=ax_marg_y, fill=True, color="salmon", vertical=True)

# Remove spines and ticks from KDE plots
for ax in [ax_marg_x, ax_marg_y]:
    ax.axis("off")

# Label main plot
ax_joint.set_xlabel("$Saltfront$ (RM)", fontsize=14)
ax_joint.set_ylabel("$T_{max}$ (°C)", fontsize=14)

# Optional: add legend manually
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=label) for label, color in zone_color_map.items()]
ax_joint.legend(handles=legend_elements, title="Zone", frameon=False, loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=12)

ax_joint.axhline(24, color="grey", linestyle="-")
ax_joint.axvline(82.9, color="grey", linestyle=":")
ax_joint.axvline(87, color="grey", linestyle=":")
ax_joint.axvline(92.5, color="grey", linestyle=":")

ax_joint.set_xlim([55, 90])
ax_joint.set_ylim([-0.5, 30])
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "tmax_and_saltfront_dynamics.jpg")
plt.show()



#%%
# Pairplot for all columns in df
sns.pairplot(df)
plt.suptitle("2007-2023", y=1.02)
plt.show()

#%%
df_ = df[df["Salt front (RM)"] >= 80]
df_ = df[df.index.month.isin([6, 7, 8])]

# Jointplot between saltfront and Tmax
g = sns.jointplot(
    data=df_,
    x="Salt front (RM)", y="Water Tmax (degC)", kind="hex", color="#4CB391")

g.ax_joint.axhline(24, color="r", linestyle="-")
g.ax_joint.axvline(82.9, color="b", linestyle="--")
g.ax_joint.axvline(87, color="b", linestyle="-.")
g.ax_joint.axvline(92.5, color="b", linestyle=":")
plt.show()

#%%

fig, ax = plt.subplots()
ax.scatter(x=df["Salt front (RM)"], y=df["Water Tmax (degC)"], label="sim", s=1, alpha=0.5)

ax.scatter(x=df["obs_saltfront"], y=df["obs_T_C"], label="obs", s=1, alpha=0.5)

ax.axvline(82.9, c="grey", lw=1, ls="--")
ax.axvline(87, c="grey", lw=1, ls="--")
ax.axvline(92.5, c="grey", lw=1, ls="--")
ax.axhline(24, c="grey", lw=1, ls="--")
ax.set_xlabel("Salt front (RM)")
ax.set_ylabel("Water Tmax (degC)")
ax.legend()
plt.show()

#%% Drought zone analysis
zone = 4
if zone == 1:
    df_ = df[df["nyc_zone"].isin([0,1,2])]
else:
    df_ = df[df["nyc_zone"]==zone]
df_.index = df_.index.strftime("%Y/%m/%d")

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(10, 6))
axes = axes.flatten()

ax = axes[0]
ax.bar(df_.index, df_["Water Tmax (degC)"], color="royalblue")
ax.bar(df_.index, df_["obs_T_C"], color="yellow", alpha=0.3)
ax.set_ylabel("Water Tmax (°C)")
ax.axhline(24, color="grey", linewidth=1, linestyle="--")

ax.set_title("Flood")
if zone == 3:
    ax.set_title("Normal")
if zone == 4:
    ax.set_title("Drought watch")
if zone == 5:
    ax.set_title("Drought warning")
if zone == 6:
    ax.set_title("Drought emergency")

ax = axes[1]
ax.bar(df_.index, df_["Salt front (RM)"], color="red")
ax.set_ylabel("Salt front (RM)")
ax.axhline(82.9, c="grey", lw=1, ls="--")
ax.axhline(87, c="grey", lw=1, ls="--")
ax.axhline(92.5, c="grey", lw=1, ls="--")

ax.legend()
plt.xticks(rotation=90)
plt.show()

#%%
# No need to do this. It has been internalized in the Pywr-DRB
# for c in df_temperature:
#     if c != "thermal_release_requirement":
#         df_temperature[c] = df_temperature[c].shift(-1)

# for c in df_salinity:
#     df_salinity[c] = df_salinity[c].shift(-1)