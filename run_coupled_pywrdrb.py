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

#%%
inflow_type = 'pub_nhmv10_BC_withObsScaled'
model_filename = str(pn.sc.wd / f"{inflow_type}.json")
output_filename = str(pn.sc.wd / f"{inflow_type}.hdf5")


temp_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "activate_thermal_control": False, 
    "Q_C_lstm_var_name": "QbcTavg_Q_C", 
    "Q_i_lstm_var_name": "QbcTavg_T_C",
    "disable_tqdm": False
    }

salinity_options = {
    "PywrDRB_ML_plugin_path": pn.get(), 
    "start_date": None, 
    "Q_Trenton_lstm_var_name": "Q_Trenton_bc", 
    "Q_Schuylkill_lstm_var_name": "Q_Schuylkill_bc",
    "disable_tqdm": False
    }

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="1960-01-01",
    end_date="1970-12-31",
    options={
        "temperature_model": temp_options,
        "salinity_model": salinity_options,
        }
    )
#%%
mb.make_model()
mb.write_model(model_filename)
model = pywrdrb.Model.load(str(model_filename))
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)
stats = model.run()
#%%
data = pywrdrb.Data()
results_sets = [
    'major_flow', 
    'res_storage', 
    'res_release', 
    'inflow', 
    'catchment_consumption', 
    'downstream_release_target', 
    'reservoir_downstream_gage',
    'temperature', 
    'salinity', 
    ]
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

df_res_release = data.res_release[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]
df_res_storage = data.res_storage[inflow_type][0]
df_catchment_consumption = data.catchment_consumption[inflow_type][0]
df_downstream_release_target = data.downstream_release_target[inflow_type][0]
df_reservoir_downstream_gage = data.reservoir_downstream_gage[inflow_type][0]
df_temperature = data.temperature[inflow_type][0]
df_salinity = data.salinity[inflow_type][0]
df_hdf5 = pywrdrb.hdf5_to_dict(output_filename)
#%% Check Estimated_Q_C
df = pd.DataFrame()
df["cannonsville_direct_release"] = df_res_release["cannonsville"]
df["downstream_release_target_cannonsville"] = df_downstream_release_target['cannonsville']
df["outflow_cannonsville"] = df_hdf5["outflow_cannonsville"]
df["spill_cannonsville"] = df_hdf5["spill_cannonsville"]
df["storage_cannonsville"] = df_res_storage["cannonsville"]
df["01425000"] = df_major_flow["01425000"]
df["max_flow_catchmentConsumption_01425000"] = df_hdf5["max_flow_catchmentConsumption_01425000"]
df["estimated_Q_C"] = df_temperature["estimated_Q_C"]
df["diff"] = df["estimated_Q_C"] - df["01425000"]

#%% Check Estimated_Q_i
df = pd.DataFrame()
df["pepacton_direct_release"] = df_res_release["pepacton"]
df["downstream_release_target_pepacton"] = df_downstream_release_target['pepacton']
df["outflow_pepacton"] = df_hdf5["outflow_pepacton"]
df["spill_pepacton"] = df_hdf5["spill_pepacton"]
df["storage_pepacton"] = df_res_storage["pepacton"]
df["01417000"] = df_major_flow["01417000"] # pepacton
df["max_flow_catchmentConsumption_01417000"] = df_hdf5["max_flow_catchmentConsumption_01417000"] # pepacton
df["flow_delLordville"] = df_hdf5["flow_delLordville"] 
df["max_flow_catchmentConsumption_delLordville"] = df_hdf5["max_flow_catchmentConsumption_delLordville"] 
df["estimated_Q_i"] = df_temperature["estimated_Q_i"]
df["diff"] = df["estimated_Q_i"] - (df["01417000"] + df["flow_delLordville"]-df["max_flow_catchmentConsumption_delLordville"])

#%%
# comfirm estimated_Q_C = calculated_estimated_Q_C
all(df["estimated_Q_C"] == df["calculated_estimated_Q_C"])
all(df["01425000"] == df["res_downstream_gauge"])
all(df["downstream_release_target_cannonsville"] == df["outflow_cannonsville"])

dff = df.loc[df["downstream_release_target_cannonsville"] != df["outflow_cannonsville"], :]

"reservoir_downstream_gage": "Streamflow at downstream gage below reservoirs (MGD).",
"major_flow": "Streamflow at major flow points of interest (MGD).",
"res_release": "Reservoir releases (MGD).",
"downstream_release_target": "Downstream release targets at Montague & Trenton (MGD).",
# res_downstream_gauge = direct_release + flow_01425000
#%%
parameter_names = [p.name for p in model.parameters if p.name]
import pywr
link_nodes = [n for n in model.nodes if isinstance(n, pywr.Link)]
"delTrenton" in parameter_names
model.nodes["link_delTrenton"].flow
model.nodes["link_delTrenton"].index

dir(model.nodes["reservoir_cannonsville"])
model.nodes["reservoir_cannonsville"].volume[0]
model.nodes["reservoir_cannonsville"].max_volume.get_value("0")
#%%
data = pywrdrb.Data()
results_sets = ['major_flow', 'res_storage', 'res_release', 'inflow']
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

df_res_release = data.res_release[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]
df_res_storage = data.res_storage[inflow_type][0]

df_hdf5 = pywrdrb.hdf5_to_dict(output_filename)
df_hdf5["temperature_after_thermal_release_mu"]
aaa = df_hdf5["estimated_Q_i"]
aaa = df_hdf5["estimated_Q_C"]

#%%
df = pd.DataFrame()
df["rel_cannonsville"] = df_res_release["cannonsville"]
df["01425000"] = df_major_flow["01425000"]
df["flow_lordville"] = df_major_flow["delLordville"]
df["01463500"] = df_major_flow["delTrenton"]
df["01474500"] = df_major_flow["outletSchuylkill"]

df["cannonsville_storage_pct"] = df_res_storage["cannonsville"] / 95700 * 100
df["pepacton_storage_pct"] = df_res_storage["pepacton"] / 140200 * 100
df["cannonsville_storage"] = df_res_storage["cannonsville"] 
df["pepacton_storage"] = df_res_storage["pepacton"] 
df.index.name = "date"
#df.to_csv(pn.data.raw.get() / f"pywrdrb_{inflow_type}_flow_and_storage.csv") # mgd / mg

#%%
r"""
import clt # My own private toolbox
import matplotlib.pyplot as plt
df_obs = pd.read_csv(pn.data.raw.get("drb_reservoir_storage_mg_2000_2024.csv"), parse_dates=True, index_col=[0])

fig, ax = plt.subplots()
ax.plot(df["cannonsville_storage_pct"], label="wrfaorc_withObsScaled")
ax.plot(df_obs["cannonsville_pct"], label="obs")
ax.legend()
ax.set_ylabel("cannonsville_storage_pct")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
ax.plot(df["pepacton_storage_pct"], label="wrfaorc_withObsScaled")
ax.plot(df_obs["pepacton_pct"], label="obs")
ax.legend()
ax.set_ylabel("pepacton_storage_pct")
plt.tight_layout()
plt.show()

"""