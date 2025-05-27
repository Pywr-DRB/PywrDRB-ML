import pandas as pd
import pathnavigator

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("pywrdrb")

import pywrdrb

r"""
Need to recheck wrf bias
For now I still use obs storage 
We can consider using hybrid nwm
"""
#%%
inflow_type = 'pub_nhmv10_BC_withObsScaled'

mb = pywrdrb.ModelBuilder(
    inflow_type=inflow_type, 
    start_date="1945-01-01",
    end_date="2023-12-31"
    )

# Make a model
mb.make_model()

model_filename = str(pn.data.pywrdrb.get() / f"{inflow_type}.json")
mb.write_model(model_filename)

model = pywrdrb.Model.load(str(model_filename))
output_filename = str(pn.data.pywrdrb.get() / f"{inflow_type}.hdf5")
recorder = pywrdrb.OutputRecorder(
    model=model,
    output_filename=output_filename,
    parameters=[p for p in model.parameters if p.name]
)
stats = model.run()

#%%
data = pywrdrb.Data()
results_sets = ['major_flow', 'res_storage', 'res_release', 'inflow', 'max_flow_catchmentConsumption']
data.load_output(output_filenames=[output_filename], results_sets=results_sets)

df_res_release = data.res_release[inflow_type][0]
df_major_flow = data.major_flow[inflow_type][0]
df_res_storage = data.res_storage[inflow_type][0]
df_inflow = data.inflow[inflow_type][0]
df_consumption = data.max_flow_catchmentConsumption[inflow_type][0]

#%%
df = pd.DataFrame()
# Q_C
df["flow_01425000"] = df_major_flow["01425000"] 
# Q_i
df["flow_01417000"] = df_major_flow["01417000"] 
df["inflow_delLordville"] = df_inflow["delLordville"]
df["consumption_delLordville"] = df_consumption["delLordville"]
# Q_L
df["flow_lordville"] = df_major_flow["delLordville"]
# Salinity
df["flow_delTrenton"] = df_major_flow["delTrenton"]
df["flow_outletSchuylkill"] = df_major_flow["outletSchuylkill"]

# Storage
df["cannonsville_storage"] = df_res_storage["cannonsville"] 
df["pepacton_storage"] = df_res_storage["pepacton"] 
df["cannonsville_storage_pct"] = df_res_storage["cannonsville"] / 95700 * 100
df["pepacton_storage_pct"] = df_res_storage["pepacton"] / 140200 * 100

df.index.name = "date"
df.to_csv(pn.data.raw.get() / f"pywrdrb_{inflow_type}_flow_and_storage.csv") # mgd / mg

#%% Check water balance that Q_L = Q_C + Q_i
dff = pd.DataFrame()
dff["Q_C"] = df["flow_01425000"]
dff["Q_i"] = df["flow_01417000"] + df["inflow_delLordville"] - df["consumption_delLordville"]
dff["Q_L"] = df["flow_lordville"]
dff["Q_C + Q_i"] = dff["Q_C"] + dff["Q_i"]
dff["Diff"] = dff["Q_C + Q_i"] - dff["Q_L"]

assert all(dff["Diff"].abs() < 0.2), "Water inbalance"


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