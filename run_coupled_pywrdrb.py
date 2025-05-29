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
    "debug": False
    }

salinity_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc", 
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "disable_tqdm": False,
    "debug": False
    }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="1960-01-01",
    end_date="2023-12-31",
    options={
        "temperature_model": temp_options,
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

#%% Load the output data
data = pywrdrb.Data()
results_sets = [
    'temperature', 
    'salinity', 
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)
df_temperature = data.temperature[inflow_type][0]
for c in df_temperature:
    if c != "thermal_release_requirement":
        df_temperature[c] = df_temperature[c].shift(-1)
df_salinity = data.salinity[inflow_type][0]
for c in df_salinity:
    df_salinity[c] = df_salinity[c].shift(-1)
#%% Load obs for plotting
import matplotlib.pyplot as plt
import numpy as np
df_obs_temp = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_temp.loc[df_obs_temp["tmmx_water_src"] != "obs", "QbcTmax_T_L"] = np.nan

df_obs_salinity = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), parse_dates=True, index_col=[0])
df_obs_salinity.loc[df_obs_salinity["saltfront_src"] != "obs", "saltfront"] = np.nan
#%% Plot the results
for year in range(2006, 2024):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    fig, axs = plt.subplots(nrows=2, figsize=(8, 5), sharex=True)
    
    ax = axs[0]
    ax.fill_between(
        df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"].index, 
        df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"] - df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_sd"], 
        df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"] + df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_sd"], 
        color="royalblue", alpha=0.15, label="±1 sd")
    
    ax.plot(df_obs_temp.loc[start_date:end_date,"QbcTmax_T_L"], label="obs", color="k")
    ax.plot(df_temperature.loc[start_date:end_date,"temperature_after_thermal_release_mu"], label="coupled", color="royalblue", alpha=0.8)
    
    ax.set_ylabel("Maximum water temp.\nat Lordville (°C)")
    ax.set_ylim([-5, 30])
    ax.legend(loc="upper left", frameon=False)
    
    ax = axs[1]
    ax.plot(df_obs_salinity.loc[start_date:end_date,"saltfront"], label="obs", color="k")
    ax.plot(df_salinity.loc[start_date:end_date,"salt_front_location_mu"], label="coupled", color="r", alpha=0.8)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.set_ylim([40, 100])
    ax.legend(loc="upper left", frameon=False)
    plt.show()