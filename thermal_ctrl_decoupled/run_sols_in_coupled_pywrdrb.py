import joblib
import numpy as np
import pandas as pd
import pathnavigator
import pywrdrb
import clt

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
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio
from src.policies import GaussianRBFPolicy, RuleBasedPolicy


# Output folder for simulation results
output_folder = "coupled_pywrdrb_RBF_sols"
pn.outputs.mkdir(output_folder)
#%% Simulation with control
# General configuration
name = f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl"
pn.models.mkdir(name)
model_filename = str(pn.models.get(name) / f"{name}.json")

# Load dps solutions
policy = "GaussianRBFPolicy"
job_id = "143990"
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))

for i, sol_idx in enumerate([28, 151, 63, 106]):
    output_filename=pn.outputs.get(output_folder) / f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_RBF{i+1}_{sol_idx}.hdf5"
    model = pywrdrb.Model.load(str(model_filename))
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_filename,
        parameters=[p for p in model.parameters if p.name]
    )

    # Thermal control
    plist = [p.name for p in model.parameters]
    temperature_model = model.parameters["temperature_model"]

    # This is copied and revised from lstm_thermal_ctrl_gaussian_rbf.py
    def return_dps_func(*params):
        n_dim = 3  # Number of dimensions for the policy
        n_basis = 4  # Number of basis functions for the Gaussian RBF policy
        policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
        minmaxscalers = joblib.load(pn.thermal_ctrl_decoupled.get() / "minmaxscalers.gz")
        policy.set_params(*params)

        # Define the function that will be used for the control algorithm
        def dps_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, current_date):
            if current_date.day == 1 and current_date.month == 6:
                ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
            ml_model.update_until(date=current_date)
            
            if current_date.month in [6, 7, 8]:
                # Prepare the inputs
                # Have to retrieve storage info after update until such that t have been moved forward
                ml_model.forecast(t=ml_model.t, Q_C=Q_C, Q_i=Q_i, cannonsville_storage_pct=cannonsville_storage_pct, lead_time=0)
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
            else:
                thermal_release = 0
            
            # Manually add to records for post-processing
            ml_model.remained_bank_amount -= thermal_release
            ml_model.records["thermal_releases"][ml_model.t] = thermal_release
            ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

            return thermal_release
        return dps_func


    params = df_ref.iloc[sol_idx, :-3].values
    dps_func = return_dps_func(*params)

    temperature_model.set_control_algorithm(dps_func)
    stats = model.run()

    # Asycronized update
    ml_model = model.parameters["salinity_model"].ml_model
    ml_model.update_until(date="2024-01-01")
    df_salinity = pd.DataFrame(ml_model.records, index=ml_model.dates)
    df_salinity.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_pywrdrb_salinity_RBF{i+1}_{sol_idx}.csv")

    ml_model = model.parameters["temperature_model"].ml_model
    ml_model.update_until(date="2024-01-01")
    df_temp = pd.DataFrame(ml_model.records, index=ml_model.dates)
    df_temp.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_pywrdrb_temp_RBF{i+1}_{sol_idx}.csv")

#%% No ctrl
# General configuration
name = f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_without_ctrl"
pn.models.mkdir(name)
model_filename = str(pn.models.get(name) / f"{name}.json")

output_filename=pn.outputs.get(output_folder) / f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_without_ctrl.hdf5"
model = pywrdrb.Model.load(str(model_filename))
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)

stats = model.run()

# Asycronized update
ml_model = model.parameters["salinity_model"].ml_model
ml_model.update_until(date="2024-01-01")
df_salinity_noCtrl = pd.DataFrame(ml_model.records, index=ml_model.dates)
df_salinity_noCtrl.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_pywrdrb_salinity_no_ctrl.csv")

ml_model = model.parameters["temperature_model"].ml_model
ml_model.update_until(date="2024-01-01")
df_temp_noCtrl = pd.DataFrame(ml_model.records, index=ml_model.dates)
df_temp_noCtrl.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_pywrdrb_temp_no_ctrl.csv")

#%% Rule-based
name = f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_without_ctrl"
pn.models.mkdir(name)
model_filename = str(pn.models.get(name) / f"{name}.json")

output_filename=pn.outputs.get(output_folder) / f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_rulebased.hdf5"
model = pywrdrb.Model.load(str(model_filename))
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)

# Thermal control
plist = [p.name for p in model.parameters]
temperature_model = model.parameters["temperature_model"]

# This is copied and revised from lstm_thermal_ctrl_gaussian_rbf.py
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


params = []
dps_func = return_dps_func(*params)

temperature_model.set_control_algorithm(dps_func)
stats = model.run()

# Asycronized update
ml_model = model.parameters["salinity_model"].ml_model
ml_model.update_until(date="2024-01-01")
df_salinity = pd.DataFrame(ml_model.records, index=ml_model.dates)
df_salinity.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_pywrdrb_salinity_rulebased.csv")

ml_model = model.parameters["temperature_model"].ml_model
ml_model.update_until(date="2024-01-01")
df_temp = pd.DataFrame(ml_model.records, index=ml_model.dates)
df_temp.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_pywrdrb_temp_rulebased.csv")
#%%
Jrel = compute_reliability(df_temp, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd = compute_max_annual_accumulated_degree_days(df_temp, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr = compute_max_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
Jtubr_avg = compute_mean_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
objs = [Jtubr*3, -Jrel, Jadd, Jtubr_avg*3]
#     [0.9999, -0.2277, 0.8615, 0.9662999999999999]
# 159 [0.9999, -0.3778, 0.742, 0.9665999999999999] (dps)

Jrel_arr = compute_reliability(df_temp, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_arr = compute_max_annual_accumulated_degree_days(df_temp, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
Jtubr_arr = compute_max_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))


Jrel_noCtrl = compute_reliability(df_temp, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_noCtrl = compute_max_annual_accumulated_degree_days(df_temp, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
Jtubr_noCtrl = compute_max_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
Jtubr_avg_noCtrl = compute_mean_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
objs = [Jtubr_noCtrl*3, -Jrel_noCtrl, Jadd_noCtrl, Jtubr_avg_noCtrl*3]
# [0.9990000000000001, -0.2277, 0.8615, 0.9665999999999999]
# 159 [0.9999, -0.3778, 0.742, 0.9665999999999999] (dps)

Jrel_arr_noCtrl = compute_reliability(df_temp, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_arr_noCtrl = compute_max_annual_accumulated_degree_days(df_temp, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)
Jtubr_arr_noCtrl = compute_max_thermal_bank_usage_ratio(df_temp, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=True, last_date_of_ctrl=(8, 31))




#%%
output_filename=pn.outputs.get(output_folder) / f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_without_ctrl.hdf5"
data_dict = {k: np.ravel(v) for k, v in pywrdrb.hdf5_to_dict(output_filename).items()}
df_dict = pd.DataFrame(data_dict)
df_dict["time"] = pd.to_datetime(df_dict["time"].str.decode("utf-8"))
df_dict.set_index("time", inplace=True)
df_dict_withoutCtrl = df_dict[
    ['link_01425000', 'flow_01425000', 'max_flow_catchmentConsumption_01425000',
     "spill_cannonsville",
     'outflow_cannonsville', 'thermal_release_requirement', 'downstream_release_target_cannonsville',
     'reservoir_cannonsville', 'reservoir_pepacton', 'reservoir_neversink']
    ]
df_dict_withoutCtrl["Q_C_"] = df_dict['link_01425000'] - (df_dict['flow_01425000'] - df_dict['max_flow_catchmentConsumption_01425000'])
df_dict_withoutCtrl["reservoir_cannonsville"] = df_dict["reservoir_cannonsville"] / 95700 * 100
df_dict_withoutCtrl["reservoir_pepacton"] = df_dict["reservoir_pepacton"] / 140200 * 100
df_dict_withoutCtrl["reservoir_neversink"] = df_dict["reservoir_neversink"] / 34900 * 100

output_filename=pn.outputs.get(output_folder) / f"coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl.hdf5"
data_dict = {k: np.ravel(v) for k, v in pywrdrb.hdf5_to_dict(output_filename).items()}
df_dict = pd.DataFrame(data_dict)
df_dict["time"] = pd.to_datetime(df_dict["time"].str.decode("utf-8"))
df_dict.set_index("time", inplace=True)
df_dict_withCtrl = df_dict[
    ['link_01425000', 'flow_01425000', 'max_flow_catchmentConsumption_01425000',
     "spill_cannonsville",
     'outflow_cannonsville', 'thermal_release_requirement', 'downstream_release_target_cannonsville',
     'reservoir_cannonsville', 'reservoir_pepacton', 'reservoir_neversink']
    ]
df_dict_withCtrl["Q_C_"] = df_dict['link_01425000'] - (df_dict['flow_01425000'] - df_dict['max_flow_catchmentConsumption_01425000'])
df_dict_withCtrl["reservoir_cannonsville"] = df_dict["reservoir_cannonsville"] / 95700 * 100
df_dict_withCtrl["reservoir_pepacton"] = df_dict["reservoir_pepacton"] / 140200 * 100
df_dict_withCtrl["reservoir_neversink"] = df_dict["reservoir_neversink"] / 34900 * 100

#%%
data = pywrdrb.Data()
results_sets = [
    'temperature',
    'salinity',
    'major_flow', 'res_storage', 'res_release', 'inflow', 'max_flow_catchmentConsumption',
    "downstream_release_target"
    ]
data.load_output(output_filenames=[str(output_filename)], results_sets=results_sets)

df_temperature = data.temperature[output_filename.stem][0]
df_salinity = data.salinity[output_filename.stem][0]

df_major_flow = data.major_flow[output_filename.stem][0]
df_inflow = data.inflow[output_filename.stem][0]
df_consumption = data.max_flow_catchmentConsumption[output_filename.stem][0]
df_target = data.downstream_release_target[output_filename.stem][0]

df = pd.DataFrame()
# Q_C
df["flow_01425000"] = df_major_flow["01425000"]
# Q_i
df["flow_01417000"] = df_major_flow["01417000"]
df["inflow_delLordville"] = df_inflow["delLordville"]
df["consumption_delLordville"] = df_consumption["delLordville"]
# Q_L
df["flow_lordville"] = df_major_flow["delLordville"]

df["Q_C"] = df["flow_01425000"]
df["Q_i"] = df["flow_01417000"] + df["inflow_delLordville"] - df["consumption_delLordville"]
df["Q_L"] = df["flow_lordville"]
#%% You should not recreate the model file once it is created.
# Create coupled Pywr-DRB model with control
inflow_type = 'pub_nhmv10_BC_withObsScaled'
name = "coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl"
pn.models.mkdir(name)
model_filename = str(pn.models.get(name) / f"{name}.json")

temp_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": str(pn.get()),
    "model1": str(pn.models.get() / r"TempLSTM\TempLSTM1.yml"),
    "model2": str(pn.models.get() / r"TempLSTM\TempLSTM2.yml"),
    "Tavg2Tmax_coefs": str(pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "activate_thermal_control": True,
    "Q_C_lstm_var_name": "QbcTavg_Q_C",
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "thermal_mitigation_bank_size": 1620*3,  # mgd  3x to align with dps
    "asycronized_update": True,
    "debug": True
    }

salinity_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": pn.get_str(),
    "model_salinity": str(pn.models.get() / r"SalinityLSTM\SalinityLSTM.yml"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "asycronized_update": False,
    "debug": True
    }

# 1 year of warmup to avoid the influence from initial reservoir storage. Org: "1960-01-01"
mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type,
    start_date="1978-01-01",
    end_date="2023-12-31",
    options={
        "temperature_model": temp_options,
        "salinity_model": salinity_options,
        }
    )
mb.make_model()
mb.write_model(model_filename)

# Create coupled Pywr-DRB model without control
inflow_type = 'pub_nhmv10_BC_withObsScaled'
name = "coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_without_ctrl"
pn.models.mkdir(name)
model_filename = str(pn.models.get(name) / f"{name}.json")

temp_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": str(pn.get()),
    "model1": str(pn.models.get() / r"TempLSTM\TempLSTM1.yml"),
    "model2": str(pn.models.get() / r"TempLSTM\TempLSTM2.yml"),
    "Tavg2Tmax_coefs": str(pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "activate_thermal_control": False,
    "Q_C_lstm_var_name": "QbcTavg_Q_C",
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "thermal_mitigation_bank_size": 1620*3,  # mgd  3x to align with dps
    "asycronized_update": True,
    "debug": True
    }

salinity_options = {
    "ml_model_type": "lstm",
    "PywrDRB_ML_plugin_path": pn.get_str(),
    "model_salinity": str(pn.models.get() / r"SalinityLSTM\SalinityLSTM.yml"),
    "start_date": "1979-01-01",
    "end_date": "2023-12-31",
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc",
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "asycronized_update": False,
    "debug": True
    }

# 1 year of warmup to avoid the influence from initial reservoir storage. Org: "1960-01-01"
mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type,
    start_date="1978-01-01",
    end_date="2023-12-31",
    options={
        "temperature_model": temp_options,
        "salinity_model": salinity_options,
        }
    )
mb.make_model()
mb.write_model(model_filename)