#%%
import pandas as pd
import pathnavigator
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio

disable = False  # Set to True to disable tqdm progress bar
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']

#%% No Control
ml_model_noCtrl = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model_noCtrl.load_data(database)
ml_model_noCtrl.update_until(date=pd.Timestamp('2024-01-01'))  # Update until the end of 2023
df_noCtrl = pd.DataFrame(ml_model_noCtrl.records, index=ml_model_noCtrl.dates)
df_noCtrl.to_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv")

Jrel_noCtrl = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_noCtrl = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)

Jrel_noCtrl_arr = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_noCtrl_arr = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)

# Jrel_noCtrl
# Out[4]: 0.2018

# Jadd_noCtrl
# Out[5]: 1.0
#%% Rule-Based Control
from src.policies import RuleBasedPolicy
def return_dps_func(*params):
    # Initialize the thermal control policy with specific parameters
    policy = RuleBasedPolicy(threshold=24, thermal_release_amount=65)
    # Define the function that will be used for the control algorithm
    def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        # Retrieve the ml_model from the model
        ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
        # Reset the bank amount at the beginning of June
        if current_date.day == 1 and current_date.month == 6:
            ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
        ml_model.update_until(date=current_date)
        # Complete the preparation for thermal control
        cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t-1]  
        ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=cannonsville_storage_pct, lead_time=0)
        # Make thermal release decision and record the thermal release
        thermal_release = policy.run(X=ml_model.forecast_T_L_mu_arr)
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        return thermal_release
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

ml_model = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model.load_data(database)

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for t, date in tqdm(enumerate(dates), desc="Running thermal control policy", disable=disable):
    Q_C = None  # Placeholder for controlled release
    Q_i = None  # Placeholder for inflow
    cannonsville_storage_pct = None  # Placeholder for storage percentage

    if date.month in [6, 7, 8]:
        thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
    else:
        thermal_release = 0

    # Update data in the ml_model for the next step(s) model update.
    #t = ml_model.t
    ml_model.Q_C[t] += thermal_release
    acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
    ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  

    # Record
    ml_model.remained_bank_amount -= thermal_release
    ml_model.records["thermal_releases"][ml_model.t] = thermal_release
    ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

# Update the model until the end of the simulation period
ml_model.update_until(date="2024-01-01")

df = pd.DataFrame(ml_model.records, index=ml_model.dates)
df.to_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv")

Jrel = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

Jrel_arr = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_arr = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
Jtubr_arr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))

# Jrel
# Out[3]: 0.516

# Jadd
# Out[4]: 0.7794

# Jtubr
# Out[5]: 0.4681

#%% Ruled based with 1x bank size
def return_dps_func(*params):
    # Initialize the thermal control policy with specific parameters
    policy = RuleBasedPolicy(threshold=24, thermal_release_amount=65)
    # Define the function that will be used for the control algorithm
    def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        # Retrieve the ml_model from the model
        ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
        # Reset the bank amount at the beginning of June
        if current_date.day == 1 and current_date.month == 6:
            ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
        ml_model.update_until(date=current_date)
        # Complete the preparation for thermal control
        cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t-1]  
        ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=cannonsville_storage_pct, lead_time=0)
        # Make thermal release decision and record the thermal release
        thermal_release = policy.run(X=ml_model.forecast_T_L_mu_arr)
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        return thermal_release
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

ml_model = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620,  # mgd
    )
ml_model.load_data(database)

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for t, date in tqdm(enumerate(dates), desc="Running thermal control policy", disable=disable):
    Q_C = None  # Placeholder for controlled release
    Q_i = None  # Placeholder for inflow
    cannonsville_storage_pct = None  # Placeholder for storage percentage

    if date.month in [6, 7, 8]:
        thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
    else:
        thermal_release = 0

    # Update data in the ml_model for the next step(s) model update.
    #t = ml_model.t
    ml_model.Q_C[t] += thermal_release
    acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
    ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  

    # Record
    ml_model.remained_bank_amount -= thermal_release
    ml_model.records["thermal_releases"][ml_model.t] = thermal_release
    ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

# Update the model until the end of the simulation period
ml_model.update_until(date="2024-01-01")

df = pd.DataFrame(ml_model.records, index=ml_model.dates)
df.to_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased_1xbanksize.csv")

Jrel = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

Jrel_arr = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_arr = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
Jtubr_arr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))

# Jrel
# Out[22]: 0.5323

# Jadd
# Out[23]: 0.839

# Jtubr
# Out[24]: 1.0

#%% Historical thermal releases
database["rel_thermal"] = database["rel_thermal"].fillna(0)
def return_dps_func(*params):
    # Define the function that will be used for the control algorithm
    def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        # Retrieve the ml_model from the model
        ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
        # Reset the bank amount at the beginning of June
        if current_date.day == 1 and current_date.month == 6:
            ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
        ml_model.update_until(date=current_date)
        # Complete the preparation for thermal control
        cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t-1]  
        thermal_release = database.loc[current_date, "rel_thermal"]
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        return thermal_release
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

ml_model = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model.load_data(database)

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for t, date in tqdm(enumerate(dates), desc="Running historic thermal release", disable=disable):
    Q_C = None  # Placeholder for controlled release
    Q_i = None  # Placeholder for inflow
    cannonsville_storage_pct = None  # Placeholder for storage percentage

    if date.month in [6, 7, 8]:
        thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
    else:
        thermal_release = 0

    # Update data in the ml_model for the next step(s) model update.
    #t = ml_model.t
    ml_model.Q_C[t] += thermal_release
    acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
    ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  

    # Record
    ml_model.remained_bank_amount -= thermal_release
    ml_model.records["thermal_releases"][ml_model.t] = thermal_release
    ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

# Update the model until the end of the simulation period
ml_model.update_until(date="2024-01-01")

df_hist = pd.DataFrame(ml_model.records, index=ml_model.dates).loc["2010": ,:]
df_hist.to_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv")

Jrel_hist = compute_reliability(df_hist, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_hist = compute_max_annual_accumulated_degree_days(df_hist, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr_hist = compute_max_thermal_bank_usage_ratio(df_hist, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

Jrel_hist_arr = compute_reliability(df_hist, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_hist_arr = compute_max_annual_accumulated_degree_days(df_hist, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
Jtubr_hist_arr = compute_max_thermal_bank_usage_ratio(df_hist, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))


Jrel_obs = compute_reliability(database , col="QobsTmax_T_L", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_obs = compute_max_annual_accumulated_degree_days(database , col='QobsTmax_T_L', threshold=20, only_summer_period=True, return_distribution=False)

# Jrel_hist
# Out[3]: 0.3769

# Jadd_hist
# Out[4]: 0.5614

# Jtubr_hist
# Out[5]: 0.4994


# If consider all periods
# Jrel_hist
# Out[11]: 0.2018

# Jadd_hist
# Out[12]: 1.0

# Jtubr_hist
# Out[13]: 0.4994


# If calculate from hist data
# Jrel_obs
# Out[18]: 0.4353

# Jadd_obs
# Out[19]: 1.2668

r"""
fig, ax = plt.subplots()
database.groupby(database.index.year).sum().loc[2005:2024, ["rel_thermal"]].plot(kind='bar', legend=False, ax=ax)
ax.axhline(1620, c="k")
ax.set_xlabel("Year")
ax.set_ylabel("Thermal release (MG)")
plt.tight_layout()
plt.show()
#%% Plot Histograms
# Plot histogram for Jrel
fig, ax = plt.subplots()
ax.hist(Jrel_noCtrl_arr, bins=30, alpha=0.7, edgecolor='black', label='No ctrl')
ax.hist(Jrel_arr, bins=30, alpha=0.7, edgecolor='black', label='Rule-based ctrl')
ax.hist(Jrel_hist_arr, bins=30, alpha=0.7, edgecolor='black', label='Historical ctrl')
ax.set_xlabel('JRel Values')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)
ax.axvline(Jrel_noCtrl, color='red', linestyle='--', label=f'Jrel (no ctrl): {Jrel_noCtrl:.4f}')
ax.axvline(Jrel, color='red', linestyle=':', label=f'Jrel (rule-based): {Jrel:.4f}     ')
ax.axvline(Jrel_hist, color='red', linestyle='-.', label=f'Jrel (historical): {Jrel_hist:.4f}')
ax.legend(loc = "upper right", frameon=False)
ax.set_xlim([0, 1])
plt.tight_layout()
plt.show()
#%%
# Plot histogram for Jadd_noCtrl_arr
fig, ax = plt.subplots()
ax.hist(Jadd_noCtrl_arr, bins=30, alpha=0.7, edgecolor='black', label='No ctrl')
ax.hist(Jadd_arr, bins=30, alpha=0.7, edgecolor='black', label='Rule-based ctrl')
ax.hist(Jadd_hist_arr, bins=30, alpha=0.7, edgecolor='black', label='Historical ctrl')
ax.set_xlabel('JADD Values')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)
ax.axvline(Jadd_noCtrl*132.4373, color='red', linestyle='--', label=f'Jadd (no ctrl): {Jadd_noCtrl*132.4373:.1f}')
ax.axvline(Jadd*132.4373, color='red', linestyle=':', label=f'Jadd (rule-based): {Jadd*132.4373:.1f}     ')
ax.axvline(Jadd_hist*132.4373, color='red', linestyle='-.', label=f'Jadd (historical): {Jadd_hist*132.4373:.1f}')
ax.legend()
ax.set_xlim([0, 270])
plt.tight_layout()
plt.show()
"""

