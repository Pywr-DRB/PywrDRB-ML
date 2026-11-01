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

df_Qt = pd.DataFrame()
df_Qs = pd.DataFrame()
#%% 
policy = "GaussianRBFPolicy"
job_id = "143990"

# No control
df_major_flow = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_major_flow_no_ctrl.csv", index_col=[0], parse_dates=True)
df_Qt['No Control'] = df_major_flow['delTrenton']
df_Qs['No Control'] = df_major_flow['outletSchuylkill']

# Rule-based
df_major_flow = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_major_flow_rulebased.csv", index_col=[0], parse_dates=True)
df_Qt['Fixed-release\n(baseline)'] = df_major_flow['delTrenton']
df_Qs['Fixed-release\n(baseline)'] = df_major_flow['outletSchuylkill']

# RBF control
name = "coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl"
model_filename = str(pn.models.get(name) / f"{name}.json")

names = ["Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", 
         "Enlarged bank"]
for i, sol_idx in enumerate([28, 151, 114]):
    df_major_flow = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_major_flow_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)
    df_Qt[names[i]] = df_major_flow['delTrenton']
    df_Qs[names[i]] = df_major_flow['outletSchuylkill']

df_Qt.index = pd.date_range(start="1978-01-01", end="2023-12-31", freq='D')
df_Qs.index = pd.date_range(start="1978-01-01", end="2023-12-31", freq='D')

policy = "GaussianRBFPolicy"
job_id = "143990"
df_Qt.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_Qt.csv")
df_Qs.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_Qs.csv")
#%% df_Qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change this if your column has a different name or capitalization
reference = "No Control"

df = df_Qt/1000

train = df[reference].dropna()
scenarios = [col for col in df.columns if col != reference]

# Use all observations to set common bin edges
all_values = df.to_numpy().ravel()
all_values = all_values[np.isfinite(all_values)]
bins = np.histogram_bin_edges(all_values, bins=50)

train_min, train_max = train.min(), train.max()
colors = plt.get_cmap("tab10").colors

fig, axes = plt.subplots(
    len(scenarios), 1,
    figsize=(6, 1.2 * len(scenarios)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.atleast_1d(axes)

summary = []

for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
    values = df[scenario].dropna()

    # Normalize each histogram independently to proportions
    ax.hist(
        train, bins=bins,
        weights=np.ones(len(train)) / len(train),
        color="0.75", label="Training input of $Q_{T}$",
    )
    ax.hist(
        values, bins=bins,
        weights=np.ones(len(values)) / len(values),
        histtype="step",
        linewidth=1.8,
        color=colors[i % len(colors)],
        label=scenario,
    )

    outside = (values < train_min) | (values > train_max)
    summary.append({
        "Scenario": scenario,
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Outside training range": outside.sum(),
        "Outside training range (%)": 100 * outside.mean(),
    })

    ax.axvline(train_min, color="0.35", linestyle="--", linewidth=1)
    ax.axvline(train_max, color="0.35", linestyle="--", linewidth=1)
    ax.set_ylabel("Proportion")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.2)
    
    ax.text(
        0.02, 0.95, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.set_xlim([-10, 90])

axes[-1].set_xlabel("Trenton river discharge ($10^3$ MGD)")
plt.show()

coverage = pd.DataFrame(summary).set_index("Scenario")
print(f"No control training range: [{train_min:.2f}, {train_max:.2f}]")
print(coverage)

#%% df_Qs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change this if your column has a different name or capitalization
reference = "No Control"

df = df_Qs/1000

train = df[reference].dropna()
scenarios = [col for col in df.columns if col != reference]

# Use all observations to set common bin edges
all_values = df.to_numpy().ravel()
all_values = all_values[np.isfinite(all_values)]
bins = np.histogram_bin_edges(all_values, bins=50)

train_min, train_max = train.min(), train.max()
colors = plt.get_cmap("tab10").colors

fig, axes = plt.subplots(
    len(scenarios), 1,
    figsize=(6, 1.2 * len(scenarios)),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.atleast_1d(axes)

summary = []

for i, (ax, scenario) in enumerate(zip(axes, scenarios)):
    values = df[scenario].dropna()

    # Normalize each histogram independently to proportions
    ax.hist(
        train, bins=bins,
        weights=np.ones(len(train)) / len(train),
        color="0.75", label="Training input of $Q_{S}$",
    )
    ax.hist(
        values, bins=bins,
        weights=np.ones(len(values)) / len(values),
        histtype="step",
        linewidth=1.8,
        color=colors[i % len(colors)],
        label=scenario,
    )

    outside = (values < train_min) | (values > train_max)
    summary.append({
        "Scenario": scenario,
        "Minimum": values.min(),
        "Maximum": values.max(),
        "Outside training range": outside.sum(),
        "Outside training range (%)": 100 * outside.mean(),
    })

    ax.axvline(train_min, color="0.35", linestyle="--", linewidth=1)
    ax.axvline(train_max, color="0.35", linestyle="--", linewidth=1)
    ax.set_ylabel("Proportion")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.2)
    
    ax.text(
        0.02, 0.95, f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.set_xlim([-3, 30])

axes[-1].set_xlabel("Schuylkill river discharge ($10^3$ MGD)")
plt.show()

coverage = pd.DataFrame(summary).set_index("Scenario")
print(f"No control training range: [{train_min:.2f}, {train_max:.2f}]")
print(coverage)