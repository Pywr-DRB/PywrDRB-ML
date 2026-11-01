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

df_Sc = pd.DataFrame()
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

df_Sc['No Control'] = ml_model_noCtrl.cannonsville_storage_pct

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

df_Sc['Fixed-release\n(baseline)'] = ml_model.cannonsville_storage_pct


#%% RBF Control
import clt
import joblib
from src.policies import GaussianRBFPolicy
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
n_dim = 3  # Number of dimensions for the policy
n_basis = n_dim + 1 #2  # Number of basis functions for the Gaussian RBF policy
disable = True  # Set to True to disable tqdm progress bar
df_ref = clt.borg.read_ref(pn.outputs.get("dps_GaussianRBFPolicy_143990/borg.ref"))

def eval_func(*params):
    database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
    policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
    minmaxscalers = joblib.load(pn.thermal_ctrl_decoupled.get() / "minmaxscalers.gz")
    policy.set_params(*params)  # Generate random parameters for the policy
    
    def return_dps_func():#*params):
        # Define the function that will be used for the control algorithm
        def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
            # Retrieve the ml_model from the model
            ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
            # Reset the bank amount at the beginning of June
            if current_date.day == 1 and current_date.month == 6:
                ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
            ml_model.update_until(date=current_date)

            # Prepare the inputs
            # Have to retrieve storage info after update until such that t have been moved forward
            cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t-1]  # Placeholder for storage percentage
            ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=cannonsville_storage_pct, lead_time=0)
            forecast_T_L_mu = ml_model.forecast_T_L_mu_arr[-1]
            forecast_T_C_mu = ml_model.forecast_T_C_mu_arr[-1]

            remained_bank_ratio = ml_model.remained_bank_amount/ml_model.thermal_mitigation_bank_size

            X = np.array([
                minmaxscalers["T_L"].transform(pd.DataFrame([[forecast_T_L_mu]], columns=["T_L"]))[0][0],
                minmaxscalers["T_C"].transform(pd.DataFrame([[forecast_T_C_mu]], columns=["T_C"]))[0][0],
                remained_bank_ratio,
                ])

            # Make thermal release decision and record the thermal release
            thermal_release = policy.run(X=X) * 300 # assuming the maximum thermal release is 300 MGD per day
            # Ensure thermal release does not exceed the bank size
            thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
            return thermal_release
        return dps_func

    # Prepare the decision-making function with parameters
    dm_func = return_dps_func() #*params

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
        cannonsville_storage_pct = None        

        if date.month in [6, 7, 8]:
            thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
        else:
            thermal_release = 0

        # Update data in the ml_model for the next step(s) model update.
        #t = ml_model.t # 
        ml_model.Q_C[t] += thermal_release
        acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
        ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  # Update the storage percentage based on the thermal release

        # Record
        ml_model.remained_bank_amount -= thermal_release
        ml_model.records["thermal_releases"][ml_model.t] = thermal_release
        ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

    # Update the model until the end of the simulation period
    ml_model.update_until(date="2024-01-01")
    
   
    return ml_model.cannonsville_storage_pct
sol_idx = 28
params = df_ref.iloc[sol_idx, :-3].values
df_Sc['Standard bank,\nmax Jrel'] = eval_func(*params)

sol_idx = 151
params = df_ref.iloc[sol_idx, :-3].values
df_Sc['Standard bank,\nmin Jadd'] = eval_func(*params)

sol_idx = 114
params = df_ref.iloc[sol_idx, :-3].values
df_Sc['Enlarged bank'] = eval_func(*params)


#%%
df_Sc.index = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')

policy = "GaussianRBFPolicy"
job_id = "143990"
df_Sc.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_Sc.csv")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change this if your column has a different name or capitalization
reference = "No Control"

train = df_Sc[reference].dropna()
scenarios = [col for col in df_Sc.columns if col != reference]

# Use all observations to set common bin edges
all_values = df_Sc.to_numpy().ravel()
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
    values = df_Sc[scenario].dropna()

    # Normalize each histogram independently to proportions
    ax.hist(
        train, bins=bins,
        weights=np.ones(len(train)) / len(train),
        color="0.75", label="Training input of Sc",
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
    ax.set_xlim([0, 101])

axes[-1].set_xlabel("Cannonsville reservoir storage (%)")
plt.show()

coverage = pd.DataFrame(summary).set_index("Scenario")
print(f"No control training range: [{train_min:.2f}, {train_max:.2f}]")
print(coverage)
#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

sns.ecdfplot(
    data=df_Sc,
    palette="colorblind",
    linewidth=1.8,
    ax=ax,
)

ax.set_xlabel("Sc")  # Replace with the variable name and units
ax.set_ylabel("Cumulative proportion")
ax.grid(alpha=0.2)

ax.legend(
    title="Scenario",
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    ncol=3,
    frameon=False,
    fontsize=8,
)

plt.show()
