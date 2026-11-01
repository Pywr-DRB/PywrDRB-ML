# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:19:22 2026

@author: CL
"""

import numpy as np
import pandas as pd
import pathnavigator
import pywrdrb

import warnings
# Suppress the NumPy deprecation warning about array to scalar conversion
warnings.filterwarnings('ignore', message='.*Conversion of an array with ndim > 0 to a scalar is deprecated.*', category=DeprecationWarning)

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

# Output folder for simulation results
output_folder = "coupled_pywrdrb_RBF_sols2"
pn.outputs.mkdir(output_folder)

total_volume_mg_dict = {}
shortage_days_dict = {}
df_demand = pd.DataFrame()
df_delivery = pd.DataFrame()
df_res_level = pd.DataFrame()
#%% 
policy = "GaussianRBFPolicy"
job_id = "143990"

# No control
df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_demand_nyc_no_ctrl.csv", index_col=[0], parse_dates=True)
df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_delivery_nyc_no_ctrl.csv", index_col=[0], parse_dates=True)
df_res_level_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_res_level_no_ctrl.csv", index_col=[0], parse_dates=True)
df_demand["No control"] = df_demand_nyc["demand_nyc"]        # MGD
df_delivery["No control"] = df_delivery_nyc["delivery_nyc"]  # MGD
df_res_level["No control"] = df_res_level_nyc['nyc']

# Rule-based
df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_demand_nyc_rulebased.csv", index_col=[0], parse_dates=True)
df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_delivery_nyc_rulebased.csv", index_col=[0], parse_dates=True)
df_res_level_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_res_level_rulebased.csv", index_col=[0], parse_dates=True)
df_demand["Fixed-release\n(baseline)"] = df_demand_nyc["demand_nyc"]        # MGD
df_delivery["Fixed-release\n(baseline)"] = df_delivery_nyc["delivery_nyc"]  # MGD
df_res_level["Fixed-release\n(baseline)"] = df_res_level_nyc['nyc']

# RBF control
name = "coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl"
model_filename = str(pn.models.get(name) / f"{name}.json")

names = ["Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", 
         "Enlarged bank"]
for i, sol_idx in enumerate([28, 151, 114]):
    df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_demand_nyc_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)
    df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_delivery_nyc_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)
    df_demand[names[i]] = df_demand_nyc["demand_nyc"]        # MGD
    df_delivery[names[i]] = df_delivery_nyc["delivery_nyc"]  # MGD
    df_res_level[names[i]] = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_res_level_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)['nyc']


#%% plot shortage and days
df_shortage = df_demand - df_delivery
df_shortage[df_shortage<1] = 0

min_duration = 3
def filter_short_events(series):
    positive = series > 0
    event_groups = positive.ne(positive.shift()).cumsum()
    event_lengths = positive.groupby(event_groups).transform("sum")
    return series.mask(positive & (event_lengths < min_duration), 0.0)

if min_duration > 0:
    df_shortage = df_shortage.apply(filter_short_events)

#df_shortage.plot(ylabel="Shortage Volumn (MG)")

total_volume_mg = df_shortage.sum()        # total shortage
shortage_days = (df_shortage > 0).sum()    # total shortage days

# total_volume_mg.plot(ylabel="Shortage Volumn (MG)")
# shortage_days.plot(ylabel="Shortage Days")

import numpy as np
import matplotlib.pyplot as plt

days = shortage_days.reindex(total_volume_mg.index)
x = np.arange(len(total_volume_mg))
width = 0.36

fig, ax1 = plt.subplots(figsize=(6, 3), constrained_layout=True)
ax2 = ax1.twinx()

bars1 = ax1.bar(
    x - width / 2,
    total_volume_mg,
    width=width,
    color="#4C78A8",  # muted blue
    label="Shortage volume",
)

bars2 = ax2.bar(
    x + width / 2,
    days,
    width=width,
    color="#F2A65A",  # soft orange
    label="Shortage days",
)

ax1.set_ylabel("Shortage volume (MG)", color="#4C78A8")
ax2.set_ylabel("Shortage days", color="#D17B29")
ax1.tick_params(axis="y", colors="#4C78A8")
ax2.tick_params(axis="y", colors="#D17B29")

ax1.set_xticks(x)
ax1.set_xticklabels(total_volume_mg.index, fontsize=8)
ax1.set_xlabel("Thermal release policy")

ax1.grid(axis="y", alpha=0.2)
ax1.set_axisbelow(True)

ax1.legend(
    handles=[bars1, bars2],
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    ncol=2,
    frameon=False,
    fontsize=8,
)

plt.show()
#%% Compare each scenario with no control
dff_shortage = pd.DataFrame()

for i in ['Fixed-release\n(baseline)','Standard bank,\nmax Jrel', 'Standard bank,\nmin Jadd', 'Enlarged bank']:
    dff_shortage[i] = df_shortage[i] - df_shortage['No control']


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

n = len(dff_shortage.columns)
colors = plt.get_cmap("tab10").colors

fig, axes = plt.subplots(
    n, 1,
    figsize=(8, 1.2 * n),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# Make the single-column case iterable too
axes = [axes] if n == 1 else axes

for i, (ax, column) in enumerate(zip(axes, dff_shortage.columns)):
    ax.plot(
        dff_shortage.index,
        dff_shortage[column],
        color=colors[i % len(colors)],
        linewidth=1.5,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(str(column), fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    #ax.spines[["top", "right"]].set_visible(False)
    ax.text(
    0.02, 0.85, f"({chr(97 + i)})",
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold",
    va="top",
)

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.supylabel("Shortage volume difference from no control (MG)")

plt.show()
#%% drought event analysis

df_counts = df_res_level.apply(lambda col: col.value_counts())

zone_names = {
    6: "drought emergency",
    5: "drought watch",
    4: "drought warning",
    3: "normal",
    2: "flood",
    1: "flood",
    0: "flood",
}
df_counts = (
    df_res_level.apply(lambda col: col.value_counts())
      .reindex(range(6, -1, -1), fill_value=0)
      .fillna(0)
      .astype(int)
      .rename(index=zone_names)
)

df_counts = df_counts.groupby(level=0, sort=False).sum()


dff_counts = pd.DataFrame()
for i in ['Fixed-release\n(baseline)','Standard bank,\nmax Jrel', 'Standard bank,\nmin Jadd', 'Enlarged bank']:
    dff_counts[i] = df_counts[i] - df_counts['No control']

#dff_counts

colors = ["#4C78A8", "#F2A65A", "#59A89C", "#B07AA1"]

ax = dff_counts.plot.bar(
    figsize=(8, 4),
    color=colors,
    width=0.8,
    edgecolor="white",
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Zone")
ax.set_ylabel("Frequency difference")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    ncol=2,
    frameon=False,
    fontsize=8,
)

plt.tight_layout()
plt.show()





