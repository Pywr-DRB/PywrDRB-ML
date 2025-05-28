import pandas as pd
import pathnavigator

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.mkdir("outputs/coupled_pywrdrb")
pn.sc.add("wd", pn.get("outputs/coupled_pywrdrb"), overwrite=True)

import pywrdrb

r"""
Debugging conclusions
- If simulation is not start as the date we generated flow training data for 
LSTM, the simulated flow might not be identical. Thus, it may cause the 
differences in the output. However, it should be minor.
- For lstm, we need to update all input variables. Not just the variables that connect to the lstm.
"""
#%% Create coupled Pywr-DRB model
inflow_type = 'pub_nhmv10_BC_withObsScaled'
model_filename = str(pn.sc.wd / f"{inflow_type}.json")
output_filename = str(pn.sc.wd / f"{inflow_type}.hdf5")

temp_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "activate_thermal_control": False, 
    "Q_C_lstm_var_name": "QbcTavg_Q_C", 
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "disable_tqdm": False,
    "debug": True
    }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="2006-01-01",
    end_date="2023-12-31",
    options={
        "temperature_model": temp_options,
        }
    )

mb.make_model()
mb.write_model(model_filename)

#%% Load the model and run it
model = pywrdrb.Model.load(str(model_filename))
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)
stats = model.run()

#%% Run trained TempLSTMs
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair, return_sim_obs_pair_for_T_L

subfolder = None #"TempLSTM_lag1_connected"
model_ids = ["TempLSTM1", "TempLSTM2"]
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, disable=False)
df_obs, df_sim = return_sim_obs_pair_for_T_L(lstm1=lstms["TempLSTM1"], lstm2=lstms["TempLSTM2"])

# unscaled_data = lstm.x[0, :, :] * (lstm.input_std + 1e-10) + lstm.input_mean
# x_vars = lstm.x_vars
# df_lstm = pd.concat(pairs["SalinityLSTM"], axis=1)
# for i, var in enumerate(x_vars):
#     df_lstm[var] = unscaled_data[:, i]

#%% Extract temperature_model in the coupled pywrdrb (debugging mode)
parameter_names = [p.name for p in model.parameters if p.name]
model_records = model.parameters["temperature_model"].records
model_records = pd.DataFrame(model_records)
model_records = model_records.set_index("date")
#%% Load the output data
data = pywrdrb.Data()
results_sets = [
    'major_flow', 
    'temperature', 
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)
df_temperature = data.temperature[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]

# df_salinity["delTrenton"] = df_major_flow["delTrenton"]
# df_salinity["outletSchuylkill"] = df_major_flow["outletSchuylkill"]
#%% Plot the results
import matplotlib.pyplot as plt

for year in range(2008, 2024):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    fig, ax = plt.subplots()
    # use model_records so their is not lag 1 in datetime
    ax.fill_between(
        model_records.loc[start_date:end_date,"T_L_mu"].index, 
        model_records.loc[start_date:end_date,"T_L_mu"] - model_records.loc[start_date:end_date,"T_L_sd"], 
        model_records.loc[start_date:end_date,"T_L_mu"] + model_records.loc[start_date:end_date,"T_L_sd"], 
        color="b", alpha=0.2, label="±1 sd")
    
    ax.plot(df_obs.loc[start_date:end_date,"T_L"], label="obs", color="k")
    ax.plot(model_records.loc[start_date:end_date,"T_L_mu"], label="coupled", color="b")
    ax.plot(df_sim.loc[start_date:end_date,"T_L"], label="trained lstm", color="r", ls="--")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Maximum water temperature at Lordville (°C)")
    ax.legend()
    plt.show()

