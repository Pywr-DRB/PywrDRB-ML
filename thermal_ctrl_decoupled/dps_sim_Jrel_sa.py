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

Jrel_noCtrl_1 = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jrel_noCtrl_5 = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.05, only_summer_period=True, return_distribution=False)
Jrel_noCtrl_10 = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.1, only_summer_period=True, return_distribution=False)
Jadd_noCtrl = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)

# Jrel_noCtrl_1
# Out[10]: 0.303

# Jrel_noCtrl_5
# Out[11]: 0.4762

# Jrel_noCtrl_10
# Out[12]: 0.6464

# Jadd_noCtrl
# Out[6]: 1

# Jrel_noCtrl_arr = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
# Jadd_noCtrl_arr = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)



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

Jrel_1 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jrel_5 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.05, only_summer_period=True, return_distribution=False)
Jrel_10 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.1, only_summer_period=True, return_distribution=False)
Jadd = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

# Jrel_1
# Out[20]: 0.5913

# Jrel_5
# Out[21]: 0.8026

# Jrel_10
# Out[22]: 0.9877

# Jadd
# Out[23]: 0.7798

# Jtubr
# Out[24]: 0.2675

# Jrel_arr = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
# Jadd_arr = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
# Jtubr_arr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))

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

    df = pd.DataFrame(ml_model.records, index=ml_model.dates)

    Jrel_1 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
    Jrel_5 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.05, only_summer_period=True, return_distribution=False)
    Jrel_10 = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.1, only_summer_period=True, return_distribution=False)
    Jadd = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
    Jtubr = compute_max_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

    objs = [Jrel_1, Jrel_5, Jrel_10, Jadd, Jtubr]
    return (objs, )

sol_idx = 28
params = df_ref.iloc[sol_idx, :-3].values
sol_28 = eval_func(*params)
# [0.9981, 0.9981, 0.9981, 0.7351, 0.3255]

sol_idx = 151
params = df_ref.iloc[sol_idx, :-3].values
sol_151 = eval_func(*params)
# [0.5485, 0.6957, 0.9788, 0.7123, 0.3282]

sol_idx = 114
params = df_ref.iloc[sol_idx, :-3].values
sol_114 = eval_func(*params)
# [0.9989, 0.9989, 0.9989, 0.5006, 0.6635]

#%% Plot
# names = ["No control", "Fixed-release\n(baseline)", "historic", #\n(2010-2023)
#          "Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", 
#          "Enlarged bank"]
# cmap_highlights={
#     names[0]: 'k', 
#     names[1]: '#E41A1C',
#     names[2]: "blue",
#     names[3]: "limegreen",
#     names[4]: "cyan", #"aquamarine",
#     names[5]: "saddlebrown",
#     names[6]: "peru"
#     }
toC = 103.16
# "No control" = [Jrel_1, Jrel_5, Jrel_10, Jadd, Jtubr]
# "Fixed-release\n(baseline)" = [-0.3030, -0.4762, -0.6464, 1*toC,  0]
# "Standard bank,\nmax Jrel" = [-0.5913, -0.8026, -0.9877, 0.7798*toC, 0.2675]
# "Standard bank,\nmin Jadd" = [-0.9981, -0.9981, -0.9981, 0.7351*toC, 0.3255]
# "Enlarged bank" = [-0.5485, -0.6957, -0.9788, 0.7123*toC, 0.3282]
# [-0.9989, -0.9989, -0.9989, 0.5006*toC, 0.6635]
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

toC = 103.16

# Replace these placeholders with your actual values if available.
no_control = [np.nan, np.nan, np.nan, np.nan, np.nan]

data = {
    "No control": no_control,
    "Fixed-release\n(baseline)": [-0.3030, -0.4762, -0.6464, 1.0000 * toC, 0],
    "Standard bank,\nmax Jrel": [-0.5913, -0.8026, -0.9877, 0.7798 * toC, 0.2675],
    "Standard bank,\nmin Jadd": [-0.9981, -0.9981, -0.9981, 0.7351 * toC, 0.3255],
    "Enlarged bank": [-0.5485, -0.6957, -0.9788, 0.7123 * toC, 0.3282],
    # "Additional scenario": [-0.9989, -0.9989, -0.9989, 0.5006 * toC, 0.6635],
}

colors = {
    "No control": "black",
    "Fixed-release\n(baseline)": "#E41A1C",
    "Standard bank,\nmax Jrel": "limegreen",
    "Standard bank,\nmin Jadd": "cyan",
    "Enlarged bank": "saddlebrown",
    # "Additional scenario": "peru",
}

axis_labels = [
    r"$J_{\mathrm{rel},1}$",
    r"$J_{\mathrm{rel},5}$",
    r"$J_{\mathrm{rel},10}$",
    r"$J_{\mathrm{add}}$",
    r"$J_{\mathrm{tubr}}$",
]

# Choose meaningful fixed limits for each variable.
limits = [
    (-1.0, 0.0),
    (-1.0, 0.0),
    (-1.0, 0.0),
    (0.0, toC),
    (0.0, 0.7),
]

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(axis_labels))

for name, values in data.items():
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        continue

    normalized = np.array([
        (value - low) / (high - low)
        for value, (low, high) in zip(values, limits)
    ])

    ax.plot(
        x, normalized,
        color=colors[name],
        linewidth=2.3,
        marker="o",
        markersize=5,
        label=name,
    )

# Draw vertical axes and labels in each variable's original units.
for i, (low, high) in enumerate(limits):
    ax.axvline(i, color="0.35", linewidth=0.9, zorder=0)

    for tick in np.linspace(0, 1, 5):
        value = low + tick * (high - low)
        ax.text(
            i - 0.04, tick, f"{value:.2f}",
            ha="right", va="center",
            fontsize=9, color="0.3",
        )

ax.set_xticks(x)
ax.set_xticklabels(axis_labels, fontsize=12)
ax.set_yticks([])
ax.set_xlim(-0.25, len(axis_labels) - 0.75)
ax.set_ylim(-0.05, 1.05)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
)

fig.tight_layout()
plt.show()