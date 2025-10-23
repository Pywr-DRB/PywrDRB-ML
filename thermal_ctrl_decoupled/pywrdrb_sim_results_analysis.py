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

output_folder = "dps_GaussianRBFPolicy_139181"
salinity_dict = {
    "RBF-better Jrel": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF1_28.csv", parse_dates=True, index_col=[0]),
    "RBF-better Jadd": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF2_68.csv", parse_dates=True, index_col=[0]),
    "RRBF-best Jrel": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF3_20.csv", parse_dates=True, index_col=[0]),
    "RBF-best Jadd": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_RBF4_51.csv", parse_dates=True, index_col=[0]),
    "No control": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_no_ctrl.csv", parse_dates=True, index_col=[0]),
    "Rule-based": pd.read_csv(pn.outputs.get(output_folder) / "df_pywrdrb_salinity_rulebased.csv", parse_dates=True, index_col=[0]),
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




