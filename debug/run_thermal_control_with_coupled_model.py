import pandas as pd
import pathnavigator

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.mkdir("outputs/coupled_pywrdrb_with_control")
pn.sc.add("wd", pn.get("outputs/coupled_pywrdrb_with_control"), overwrite=True)

import pywrdrb

#%% Create coupled Pywr-DRB model
inflow_type = 'pub_nhmv10_BC_withObsScaled'
model_filename = str(pn.sc.wd / f"{inflow_type}.json")
output_filename = str(pn.sc.wd / f"{inflow_type}.hdf5")

temp_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "activate_thermal_control": True, 
    "activate_input_bias_correction": False, # Pywr start date need to be same as teh trained LSTM otherwise the flow dynamics will be significantly different leading large bias correct in the forecast.
    "Q_C_lstm_var_name": "QbcTavg_Q_C", 
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "disable_tqdm": False,
    "debug": True
    }

# salinity_options = {
#     "PywrDRB_ML_plugin_path": pn.get(), 
#     "start_date": None, 
#     "Q_Trenton_lstm_var_name": "Q_Trenton_bc", 
#     "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
#     "disable_tqdm": False,
#     "debug": False
#     }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="1960-01-01",
    end_date="2007-12-31",
    options={
        "temperature_model": temp_options,
        #"salinity_model": salinity_options,
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


#%% Thermal control (Future coding structure for MOEA during developing stage where control algorithm will be assigned externally after load model)
# plist = [p.name for p in model.parameters]

#model = pywrdrb.Model.load(str(model_filename))
temperature_model = model.parameters["temperature_model"]

def return_dps_func(*params):
    def dps_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        ml_model.forecast(Q_C, Q_i, cannonsville_storage_pct, lead_time=0)
        if current_date.month in [6, 7, 8]:
            return 1 # Thermal release MGD
        else:
            return 0
    
    return dps_func

params = [0, 0, 0]
dps_func = return_dps_func(*params)

temperature_model.control_algorithm = dps_func

#%% Run the simulation
stats = model.run()

#%% Load the output data
data = pywrdrb.Data()
results_sets = [
    'temperature', 
    #'salinity', 
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

df_temperature = data.temperature[inflow_type][0]
df_temperature["reduce degC"] = df_temperature["temperature_after_thermal_release_mu"] - df_temperature["forecasted_temperature_before_thermal_release_mu"]
#df_salinity = data.salinity[inflow_type][0]

temperature_model = model.parameters["temperature_model"]
records = temperature_model.records

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df_temperature["temperature_after_thermal_release_mu"], label="no ctrl")
ax.plot(df_temperature_ctrl["temperature_after_thermal_release_mu"], label="ctrl")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(df_temperature["temperature_after_thermal_release_mu"]-df_temperature_ctrl["temperature_after_thermal_release_mu"], label="diff")
ax.legend()
plt.show()
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
    
#%%
# No need to do this. It has been internalized in the Pywr-DRB
# for c in df_temperature:
#     if c != "thermal_release_requirement":
#         df_temperature[c] = df_temperature[c].shift(-1)

# for c in df_salinity:
#     df_salinity[c] = df_salinity[c].shift(-1)