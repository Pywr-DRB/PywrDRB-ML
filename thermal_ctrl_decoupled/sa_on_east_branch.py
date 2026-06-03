import numpy as np
import pandas as pd
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
#pn.add_to_sys_path()
pn.chdir()
#import os, sys
#print("cwd =", os.getcwd())
#print("sys.path[0] =", sys.path[0])
import clt

import joblib
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio
from tqdm import tqdm
from src.policies import GaussianRBFPolicy
# print("before")

# m = WaterTempLSTMModel(
#     model1="models/TempLSTM/TempLSTM1.yml",
#     model2="models/TempLSTM/TempLSTM2.yml",
#     Tavg2Tmax_coefs="models/TempLSTM/Tavg2Tmax_coefs.json",
#     debug=True,
# )

# print("after")


policy ="GaussianRBFPolicy"
job_id = "143990"

pn.outputs.mkdir(f"dps_{policy}_{job_id}_east_dt")

df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))

database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])
df_hist = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historic"] = database["rel_thermal"]
df_res["Rule-based"] = df_rulebased["thermal_releases"]
df_res["Tmax (No control)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (Rule-based)"] = df_rulebased["T_L_mu"]
df_res["Tmax (hist)"] = df_hist["T_L_mu"]

df_objs = pd.DataFrame()


east_dt = 1
sol_idx = 28
for east_dt in [0, 1, 2, 3, 4, 5]:
    for sol_idx in [28, 151, 114]:
    #for sol_idx in tqdm(list(df_ref.index)):
        params = df_ref.iloc[sol_idx, :-3]
        n_dim = 3  # Number of dimensions for the policy
        n_basis = n_dim + 1  # Number of basis functions for the Gaussian RBF policy
    
        def eval_func(*params):
            database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
            # Initialize the thermal control policy with specific parameters
            policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
            #params = policy.gen_params(seed=42)[0]
            minmaxscalers = joblib.load(pn.get("thermal_ctrl_decoupled") / "minmaxscalers.gz")
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
                east_tributary_temperature_perturbation=east_dt,
                )
            ml_model.load_data(database)
    
            dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
            for t, date in tqdm(enumerate(dates), desc="Running thermal control policy", disable=True):
                Q_C = None  # Placeholder for controlled release
                Q_i = None  # Placeholder for inflow
                cannonsville_storage_pct = None        
    
                if date.month in [6, 7, 8]:
                    thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
                else:
                    thermal_release = 0
    
                # Update data in the ml_model for the next step(s) model update.
                #t = ml_model.t # 
                
                acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
                ml_model.Q_C[t] += thermal_release
                ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  # Update the storage percentage based on the thermal release
    
                # Record
                ml_model.remained_bank_amount -= thermal_release
                ml_model.records["thermal_releases"][ml_model.t] = thermal_release
                ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount
                
            # Update the model until the end of the simulation period
            ml_model.update_until(date="2024-01-01")
            df = pd.DataFrame(ml_model.records, index=ml_model.dates)
            return ml_model, df
        ml_model, df_rbf = eval_func(*params)
        #df_rbf['thermal_releases'] = df_rbf['thermal_releases'].fillna(0)
        Jrel = compute_reliability(df_rbf, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
        Jadd = compute_max_annual_accumulated_degree_days(df_rbf, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
        Jtubr = compute_max_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
        Jtubr_avg = compute_mean_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
    
        objs = [Jtubr*3, -Jrel, Jadd, Jtubr_avg*3]
        
        df_res[f"{sol_idx}_{east_dt}"] = df_rbf["thermal_releases"]
        df_res[f"Tmax {sol_idx}_{east_dt}"] = df_rbf["T_L_mu"]
        df_objs[f"{sol_idx}_{east_dt}"] = objs
    
        df_rbf.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}_east_dt") / f"df_{sol_idx}_{east_dt}.csv")

df_res.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}_east_dt") / "df_res.csv")
df_objs.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}_east_dt") / "df_objs.csv")
