#%%
import joblib
import pandas as pd
import pathnavigator
from tqdm import tqdm
import numpy as np

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_mean_thermal_bank_usage_ratio
from src.policies import GeneralizedPiecewiseLinearPolicy

# Global configuration variables for the piecewise linear policy
# Policy parameters
n_dim = 5  # Number of dimensions for the policy
n_steps = 4  # Number of steps for the piecewise linear policy
disable = True  # Set to True to disable tqdm progress bar
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']

def eval_func(*params):
    # Initialize the thermal control policy with specific parameters
    policy = GeneralizedPiecewiseLinearPolicy(n_dim=n_dim, n_steps=n_steps)
    #params = policy.gen_params(seed=42)[0]
    minmaxscalers = joblib.load(pn.stage1_thermal_ctrl_decoupled.get() / "minmaxscalers.gz")
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
            # Nowcast/forecast
            #ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0)
            #forecast_T_L_mu = ml_model.forecast_T_L_mu_arr[-1]
            #forecast_T_L_sd = ml_model.forecast_T_L_sd_arr[-1]

            T_L_prev = ml_model.T_L_mu
            Q_C = ml_model.Q_C[ml_model.t]
            Q_i = ml_model.Q_i[ml_model.t]
            cannonsville_storage_pct = ml_model.cannonsville_storage_pct[ml_model.t]
            doc = ml_model.doc[ml_model.t]


            X = np.array([
                minmaxscalers["doc"].transform(pd.DataFrame([[doc]], columns=["doc"]))[0][0],
                minmaxscalers["Q_C"].transform(pd.DataFrame([[Q_C]], columns=["Q_C"]))[0][0],
                minmaxscalers["Q_i"].transform(pd.DataFrame([[Q_i]], columns=["Q_i"]))[0][0],
                minmaxscalers["cannonsville_storage_pct"].transform(pd.DataFrame([[cannonsville_storage_pct]], columns=["cannonsville_storage_pct"]))[0][0],
                minmaxscalers["T_L"].transform(pd.DataFrame([[T_L_prev]], columns=["T_L"]))[0][0],
                ])

            # Make thermal release decision and record the thermal release
            thermal_release = policy.run(X=X) * 300 # assuming the maximum thermal release is 300 MGD per day
            # Ensure thermal release does not exceed the bank size
            thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
            return thermal_release
        return dps_func

    # Prepare the decision-making function with parameters
    #params = []
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
    df = pd.DataFrame(ml_model.records, index=ml_model.dates)

    Jrel = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
    Jadd = compute_max_annual_accumulated_degree_days(df, col='T_L_mu', threshold=20, return_distribution=False)
    Jtubr = compute_mean_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

    objs = [-Jrel, Jadd, Jtubr]
    return (objs, )

# Jrel_arr = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
# Jadd_arr = compute_max_annual_accumulated_degree_days(df, col='T_L_mu', threshold=20, return_distribution=True)
# Jtubr_arr = compute_mean_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))
