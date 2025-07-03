#%%
import pandas as pd
import pathnavigator
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_mean_thermal_bank_usage_ratio


disable = False  # Set to True to disable tqdm progress bar
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']

#%% No Control
ml_model_noCtrl = WaterTempLSTMModel(
    model1=pn.models.get() / r"TempLSTM1_comparison\TempLSTM1_Qc.yml",
    model2=pn.models.get() / r"TempLSTM2_comparison\TempLSTM2_Qc.yml",
    Tavg2Tmax_coefs=pn.get() / "models/TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620,  # mgd-day
    )
ml_model_noCtrl.load_data(database)
ml_model_noCtrl.update_until(date=pd.Timestamp('2024-01-01'))  # Update until the end of 2023
df_noCtrl = pd.DataFrame(ml_model_noCtrl.records, index=ml_model_noCtrl.dates)

Jrel_noCtrl = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_noCtrl = compute_max_annual_accumulated_degree_days(df_noCtrl, col='T_L_mu', threshold=20, return_distribution=False)

Jrel_noCtrl_arr = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_noCtrl_arr = compute_max_annual_accumulated_degree_days(df_noCtrl, col='T_L_mu', threshold=20, return_distribution=True)

#%% Plot Histograms
# Plot histogram for Jrel_noCtrl_arr
fig, ax = plt.subplots()
ax.hist(Jrel_noCtrl_arr, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('JRel Values')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of JRel (No Control)')
ax.grid(True, alpha=0.3)
ax.axvline(Jrel_noCtrl, color='red', linestyle='--',
           label=f'Jrel: {Jrel_noCtrl:.4f}')
ax.legend(loc = "upper left")
ax.set_xlim([0.5, 1])
plt.tight_layout()
plt.show()

# Plot histogram for Jadd_noCtrl_arr
fig, ax = plt.subplots()
ax.hist(Jadd_noCtrl_arr, bins=30, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('JADD Values')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of JADD (No Control)')
ax.grid(True, alpha=0.3)
ax.axvline(Jadd_noCtrl, color='red', linestyle='--',
           label=f'Jadd: {Jadd_noCtrl:.4f}')
ax.legend()
ax.set_xlim([0, 230])
plt.tight_layout()
plt.show()


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
        # Make a forecast for the current date
        ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0)
        # Make thermal release decision and record the thermal release
        thermal_release = policy.run(X=ml_model.forecast_T_L_mu_arr)
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        return thermal_release
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

ml_model = WaterTempLSTMModel(
    model1=pn.models.get() / r"TempLSTM1_comparison\TempLSTM1_Qc.yml",
    model2=pn.models.get() / r"TempLSTM2_comparison\TempLSTM2_Qc.yml",
    Tavg2Tmax_coefs=pn.get() / "models/TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620,  # mgd
    )
ml_model.load_data(database)

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for date in tqdm(dates, desc="Running thermal control policy", disable=disable):
    Q_C = None  # Placeholder for controlled release
    Q_i = None  # Placeholder for inflow
    cannonsville_storage_pct = None  # Placeholder for storage percentage

    if date.month in [6, 7, 8]:
        thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
    else:
        thermal_release = 0

    # Update data in the ml_model for the next step(s) model update.
    t = ml_model.t
    ml_model.Q_C[t] += thermal_release
    Q_C = ml_model.Q_C[t]
    try:
        ml_model.X_1[t, ml_model.x_vars_1.index(ml_model.Q_C_lstm_var_name)] = Q_C
    except ValueError:
        if ml_model.debug:
            print(f"Warning: '{ml_model.Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
    try:
        ml_model.X_2[t, ml_model.x_vars_2.index(ml_model.Q_C_lstm_var_name)] = Q_C
    except ValueError:
        if ml_model.debug:
            print(f"Warning: '{ml_model.Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")

    # Record
    ml_model.remained_bank_amount -= thermal_release
    ml_model.records["thermal_releases"][ml_model.t] = thermal_release
    ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

# Update the model until the end of the simulation period
ml_model.update_until(date="2024-01-01")

#%%
ml_model.load_data(database)
df = pd.DataFrame(ml_model.records, index=ml_model.dates)

Jrel = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd = compute_max_annual_accumulated_degree_days(df, col='T_L_mu', threshold=20, return_distribution=False)

Jrel_arr = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_arr = compute_max_annual_accumulated_degree_days(df, col='T_L_mu', threshold=20, return_distribution=True)

#%% Plot Histograms
# Plot histogram for Jrel_noCtrl_arr
fig, ax = plt.subplots()
ax.hist(Jrel_arr, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('JRel Values')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of JRel (No Control)')
ax.grid(True, alpha=0.3)
ax.axvline(Jrel, color='red', linestyle='--',
           label=f'Jrel: {Jrel:.4f}')
ax.legend(loc = "upper left")
ax.set_xlim([0.5, 1])
plt.tight_layout()
plt.show()

# Plot histogram for Jadd_noCtrl_arr
fig, ax = plt.subplots()
ax.hist(Jadd_arr, bins=30, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('JADD Values')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of JADD (No Control)')
ax.grid(True, alpha=0.3)
ax.axvline(Jadd, color='red', linestyle='--',
           label=f'Jadd: {Jadd:.4f}')
ax.legend()
ax.set_xlim([0, 230])
plt.tight_layout()
plt.show()

r"""
{'T_C': np.float64(1.612883998379305),
 'T_i': np.float64(6.684021409680599),
 'Tavg': np.float64(1.2969361086676146),
 'T_L': np.float64(1.4062992125475333)}
"""
