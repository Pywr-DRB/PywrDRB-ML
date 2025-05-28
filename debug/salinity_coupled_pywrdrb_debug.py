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

salinity_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc", 
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "disable_tqdm": False,
    "debug": True
    }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="2017-01-01",
    end_date="2023-12-31",
    options={
        "salinity_model": salinity_options,
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

#%% Run trained SalinityLSTM
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair

subfolder = None #"SalinityLSTM"
model_ids = ["SalinityLSTM"] #, "LSTM_bc_lag1"]
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, disable=False)
lstm = lstms["SalinityLSTM"]
pairs = return_sim_obs_pair(lstms, period="train", only_months=None, mode="SalinityLSTM", disable=False)

unscaled_data = lstm.x[0, :, :] * (lstm.input_std + 1e-10) + lstm.input_mean
x_vars = lstm.x_vars
df_lstm = pd.concat(pairs["SalinityLSTM"], axis=1)
for i, var in enumerate(x_vars):
    df_lstm[var] = unscaled_data[:, i]

#%% Extract salinity_model in the coupled pywrdrb (debugging mode)
parameter_names = [p.name for p in model.parameters if p.name]
model_records = model.parameters["salinity_model"].records
model_records = pd.DataFrame(model_records)
model_records = model_records.set_index("date")
#%% Load the output data
data = pywrdrb.Data()
results_sets = [
    'major_flow', 
    'salinity', 
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)
df_salinity = data.salinity[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]

df_salinity["delTrenton"] = df_major_flow["delTrenton"]
df_salinity["outletSchuylkill"] = df_major_flow["outletSchuylkill"]
#%% Plot the results
# The SD is super large so not ploting them out here.
import matplotlib.pyplot as plt

for year in range(1963, 2024):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    fig, ax = plt.subplots()
    # use model_records so their is not lag 1 in datetime
    ax.plot(df_lstm.loc[start_date:end_date,"saltfront"], label="obs", color="k")
    ax.plot(model_records.loc[start_date:end_date,"mu"], label="coupled", color="b")
    ax.plot(df_lstm.loc[start_date:end_date,"mu_ft"], label="trained lstm", color="r", ls="--")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.set_ylim([40, 100])
    ax.legend()
    plt.show()

