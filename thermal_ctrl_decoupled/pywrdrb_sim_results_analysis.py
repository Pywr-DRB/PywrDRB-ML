import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt
import pandas as pd

output_folder = "dps_GaussianRBFPolicy_143990"
salinity_dict = {
    "Standard bank,\nmax Jrel": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF1_28.csv", parse_dates=True, index_col=[0]),
    "Standard bank,\nmin Jadd": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF2_151.csv", parse_dates=True, index_col=[0]),
    "Enlarged bank": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF3_114.csv", parse_dates=True, index_col=[0]),
    "No control": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_no_ctrl.csv", parse_dates=True, index_col=[0]),
    "Ficed-release\n(baseline)": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_rulebased.csv", parse_dates=True, index_col=[0]),
    }

temp_dict = {
    "Standard bank,\nmax Jrel": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_temp_RBF1_28.csv", parse_dates=True, index_col=[0]),
    "Standard bank,\nmin Jadd": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_temp_RBF2_151.csv", parse_dates=True, index_col=[0]),
    "Enlarged bank": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_temp_RBF3_114.csv", parse_dates=True, index_col=[0]),
    "No control": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_temp_no_ctrl.csv", parse_dates=True, index_col=[0]),
    "Ficed-release\n(baseline)": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_temp_rulebased.csv", parse_dates=True, index_col=[0]),
    }


#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5, 4))

styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
lws = [3.5, 3, 2.5, 2, 1.5, 1]
for i, (label, df) in enumerate(salinity_dict.items()):
    data = df["sf_mu"].dropna().values
    data = data[data >= 60]
    sorted_data = np.sort(data)
    cdf = np.linspace(0, 1, len(sorted_data))

    #ax.plot(sorted_data, cdf, label=label)
    ax.plot(sorted_data, cdf, label=label, linestyle=styles[i % len(styles)], lw=lws[i], alpha=0.4)

ax.set_xlim([60, 87])
ax.set_ylim([0, 1])
ax.set_xlabel("Salt front location (RM)")
ax.set_ylabel("Empirical CDF")
ax.grid(True)
ax.legend()
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.attemp1.get()/"coupled_pywrdrb_sols.jpg")
plt.show()

#%% Load obs for plotting
import matplotlib.pyplot as plt
import numpy as np
df_obs_temp = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_temp.loc[df_obs_temp["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan

df_obs_salinity = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_salinity.loc[df_obs_salinity["saltfront_src"] != "obs", "saltfront"] = np.nan

#%% Plot the results

import clt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

df_temperature = temp_dict["Standard bank,\nmax Jrel"]
df_salinity = salinity_dict["Standard bank,\nmax Jrel"]
df_res_level = pd.read_csv(pn.outputs.get(output_folder) / "df_res_level_RBF3_114.csv", parse_dates=True, index_col=[0])



df = pd.DataFrame()
df["Water Tmax (degC)"] = df_temperature.loc["1979":"2023", "T_L_mu"]
df["Salt front (RM)"] = df_salinity.loc["1979":"2023", "sf_mu"]
df["obs_T_C"] = df_obs_temp["QbcTmax_T_L"]
df["obs_saltfront"] = df_obs_salinity["saltfront"]
df["nyc_zone"] = df_res_level["nyc"]
df["zone"] = "flood"

df.loc[df["nyc_zone"] == 6, "zone"] = "drought emergency"
df.loc[df["nyc_zone"] == 5, "zone"] = "drought watch"
df.loc[df["nyc_zone"] == 4, "zone"] = "drought warning"
df.loc[df["nyc_zone"] == 3, "zone"] = "normal"

mask = (df["zone"] == "flood") & (df.index.month.isin([6, 7, 8]))
df.loc[mask, "zone"] = "flood (Jun, Jul, Aug)"
df.loc[~mask & (df["zone"] == "flood"), "zone"] = "flood (other months)"

df["zone"] = pd.Categorical(df["zone"], categories=[
    "drought emergency", "drought watch", "drought warning", "normal", "flood (other months)", "flood (Jun, Jul, Aug)"
])
df["date"] = df.index

# Define custom color map
zone_color_map = {
    "drought emergency": "#8B0000",  # dark red
    "drought watch": "#FF0000",      # red
    "drought warning": "mistyrose", #"#FFA07A",    # light red (light salmon) "salmon", #
    "normal": "#D3D3D3",             # light gray "darkgrey", #
    #"flood": "royalblue",             # royal blue
    "flood (other months)": "royalblue",
    "flood (Jun, Jul, Aug)": "blue",
}

# Reorder DataFrame to control z-order
#zorder_order = ["normal","drought warning","flood","drought watch","drought emergency"]
zorder_order = ["drought warning","normal","flood (other months)", "flood (Jun, Jul, Aug)", "drought watch","drought emergency"]
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

# Observed
# sc_obs = ax_joint.scatter(
#     df["obs_saltfront"],
#     df["obs_T_C"],
#     c=zone_colors,
#     marker='+',
#     s=10,
#     alpha=0.6,
#     linewidths=0.5,
#     label='Observed'
# )

# Simulated
sc_sim = ax_joint.scatter(
    df["Salt front (RM)"],
    df["Water Tmax (degC)"],
    c=zone_colors,
    s=10,
    alpha=0.6,
    marker='o',
    label='Simulated'
)


# KDE plots
df_kde = df[df["zone"] == "normal"]
sns.kdeplot(df_kde["Salt front (RM)"], ax=ax_marg_x, fill=True, color="darkgrey")
sns.kdeplot(y=df_kde.loc[df_kde["date"].dt.month.isin([6, 7, 8]), "Water Tmax (degC)"], ax=ax_marg_y, fill=True, color="darkgrey")

df_kde = df[df["zone"] == "drought warning"]
sns.kdeplot(df_kde["Salt front (RM)"], ax=ax_marg_x, fill=True, color="salmon")
sns.kdeplot(y=df_kde.loc[df_kde["date"].dt.month.isin([6, 7, 8]), "Water Tmax (degC)"], ax=ax_marg_y, fill=True, color="salmon")

# Add arrow and annotation for thermal control period
ax_marg_y.annotate(
    "Thermal control\nperiod only\n(Jun, Jul, Aug)",
    xy=(0.18, 23.5),  # Point on the KDE plot (x is KDE density, y is temperature)
    xytext=(0.6, 27),  # Text position (further right and lower)
    xycoords='data',
    fontsize=11,
    ha='center',
    va='center',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=None),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2, connectionstyle="arc3,rad=0.1")
)

#sns.kdeplot(df_kde["Salt front (RM)"], ax=ax_marg_x, fill=True, color="mediumpurple")
#sns.kdeplot(df_kde["Water Tmax (degC)"], ax=ax_marg_y, fill=True, color="salmon", vertical=True)

# Clean up marginal plots while keeping ticks
ax_marg_x.spines['top'].set_visible(False)
ax_marg_x.spines['right'].set_visible(False)
ax_marg_x.spines['left'].set_visible(False)
ax_marg_x.set_ylabel('')
ax_marg_x.set_yticks([])
ax_marg_x.tick_params(labeltop=False)
ax_marg_x.axis("off")

ax_marg_y.spines['top'].set_visible(False)
ax_marg_y.spines['right'].set_visible(False)
ax_marg_y.spines['bottom'].set_visible(False)
ax_marg_y.set_xlabel('')
ax_marg_y.set_xticks([])
ax_marg_y.tick_params(labelright=False)
ax_marg_y.axis("off")

# Label main plot
ax_joint.set_xlabel("$Saltfront$ (RM)", fontsize=14)
ax_joint.set_ylabel("$T_{max}$ (°C)", fontsize=14)

# (not working) Set ticks to point inward for the main plot with enhanced visibility
ax_joint.tick_params(direction='in', length=6, width=1, colors='black')

# Optional: add legend manually
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=label) for label, color in zone_color_map.items()]
zone_legend = ax_joint.legend(handles=legend_elements, title="Reservoir operation zone", 
                              frameon=False, loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=12)
# Add marker legend
from matplotlib.lines import Line2D
marker_legend_elements = [
    Line2D([0], [0], marker='+', color='w', markeredgecolor='black', markeredgewidth=1.5, markersize=8, label='Observed'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='Simulated                  ')
]
marker_legend = ax_joint.legend(handles=marker_legend_elements, title="Data type", frameon=False, loc="center left", bbox_to_anchor=(1.15, 0.15), fontsize=12)

from matplotlib.patches import Rectangle

# Example: add a white rectangle in data coordinates
fig.patches.append(Rectangle(
    xy=(0.95, 0.22),      # figure fraction coordinates (0–1)
    width=0.18, height=0.15,
    transform=fig.transFigure,
    facecolor='white',
    edgecolor=None,
    alpha=1.0,
    zorder=5
))

# Add the zone legend back since the marker legend overwrites it
ax_joint.add_artist(zone_legend)

ax_joint.axhline(24, color="grey", linestyle="-")
ax_joint.axvline(82.9, color="grey", linestyle=":")
ax_joint.axvline(87, color="grey", linestyle=":")
ax_joint.axvline(92.5, color="grey", linestyle=":")

# Add rotated text labels for vertical lines
ax_joint.text(82.3, 27, "82.9", rotation=90, ha='center', va='bottom', fontsize=11, color='grey')
ax_joint.text(86.4, 27, "87.0", rotation=90, ha='center', va='bottom', fontsize=11, color='grey')
ax_joint.text(91.9, 27, "92.5", rotation=90, ha='center', va='bottom', fontsize=11, color='grey')
ax_joint.text(57, 24.1, "24.0", rotation=0, ha='center', va='bottom', fontsize=11, color='grey')

ax_joint.set_xlim([55, 93])
ax_joint.set_ylim([-0.5, 30])
#plt.tight_layout()
#clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "tmax_and_saltfront_dynamics_RBF.jpg")
plt.show()


#%%
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio

bank_size = 1620*3

temp_dict = {
    "RBF-1": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_RBF1_159.csv", parse_dates=True, index_col=[0]),
    "RBF-2": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_RBF2_108.csv", parse_dates=True, index_col=[0]),
    "RBF-3": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_RBF3_106.csv", parse_dates=True, index_col=[0]),
    "RBF-4": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_RBF4_11.csv", parse_dates=True, index_col=[0]),
    "no_ctrl": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_no_ctrl.csv", parse_dates=True, index_col=[0]),
    "rule_based": pd.read_csv(pn.outputs.get("stage1_nowcast_GaussianRBFPolicy_135322") / "df_pywrdrb_temp_rulebased.csv", parse_dates=True, index_col=[0]),
    }

Jrel_dict = {
    "RBF-1": compute_reliability(temp_dict["RBF-1"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    "RBF-2": compute_reliability(temp_dict["RBF-2"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    "RBF-3": compute_reliability(temp_dict["RBF-3"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    "RBF-4": compute_reliability(temp_dict["RBF-4"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    "no_ctrl": compute_reliability(temp_dict["no_ctrl"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    "rule_based": compute_reliability(temp_dict["rule_based"], col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True),
    }

Jadd_dict = {
    "RBF-1": compute_max_annual_accumulated_degree_days(temp_dict["RBF-1"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    "RBF-2": compute_max_annual_accumulated_degree_days(temp_dict["RBF-2"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    "RBF-3": compute_max_annual_accumulated_degree_days(temp_dict["RBF-3"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    "RBF-4": compute_max_annual_accumulated_degree_days(temp_dict["RBF-4"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    "no_ctrl": compute_max_annual_accumulated_degree_days(temp_dict["no_ctrl"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    "rule_based": compute_max_annual_accumulated_degree_days(temp_dict["rule_based"], col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True),
    }

Jtubr_dict = {
    "RBF-1": compute_max_thermal_bank_usage_ratio(temp_dict["RBF-1"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    "RBF-2": compute_max_thermal_bank_usage_ratio(temp_dict["RBF-2"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    "RBF-3": compute_max_thermal_bank_usage_ratio(temp_dict["RBF-3"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    "RBF-4": compute_max_thermal_bank_usage_ratio(temp_dict["RBF-4"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    "no_ctrl": compute_max_thermal_bank_usage_ratio(temp_dict["no_ctrl"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    "rule_based": compute_max_thermal_bank_usage_ratio(temp_dict["rule_based"], col='remained_bank_amounts', bank_size=bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))*3,
    }

#%%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5, 4))

styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
lws = [3.5, 3, 2.5, 2, 1.5, 1]
for i, (label, arr) in enumerate(Jrel_dict.items()):
    arr = arr[arr<1]
    sorted_data = np.sort(arr)
    cdf = np.linspace(0, 1, len(sorted_data))

    #ax.plot(sorted_data, cdf, label=label)
    ax.plot(sorted_data, cdf, label=label, linestyle=styles[i % len(styles)], lw=lws[i], alpha=0.7)

#ax.set_xlim([60, 87])
ax.set_ylim([0, 1])
ax.set_xlabel("Jrel")
ax.set_ylabel("Empirical CDF")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(figsize=(5, 4))

styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
lws = [3.5, 3, 2.5, 2, 1.5, 1]
for i, (label, arr) in enumerate(Jadd_dict.items()):
    sorted_data = np.sort(arr)
    cdf = np.linspace(0, 1, len(sorted_data))

    #ax.plot(sorted_data, cdf, label=label)
    ax.plot(sorted_data, cdf, label=label, linestyle=styles[i % len(styles)], lw=lws[i], alpha=0.7)

#ax.set_xlim([60, 87])
ax.set_ylim([0, 1])
ax.set_xlabel("Jadd")
ax.set_ylabel("Empirical CDF")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(figsize=(5, 4))

styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
lws = [3.5, 3, 2.5, 2, 1.5, 1]
for i, (label, arr) in enumerate(Jtubr_dict.items()):
    sorted_data = np.sort(arr)
    cdf = np.linspace(0, 1, len(sorted_data))

    #ax.plot(sorted_data, cdf, label=label)
    ax.plot(sorted_data, cdf, label=label, linestyle=styles[i % len(styles)], lw=lws[i], alpha=0.7)

#ax.set_xlim([60, 87])
ax.set_ylim([0, 1])
ax.set_xlabel("Jtubr")
ax.set_ylabel("Empirical CDF")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()




