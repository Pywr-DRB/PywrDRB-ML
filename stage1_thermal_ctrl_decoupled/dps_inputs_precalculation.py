#%%
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()

db = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
db = db[["QbcTavg_Q_C", "QbcTavg_Q_i", "bc_cannonsville_storage_pct", "QbcTavg_T_C", "QbcTavg_T_i", "QbcTavg_T_L", "QbcTmax_T_L", "doc"]]
db = db.rename(columns={
    "QbcTavg_Q_C": "Q_C",
    "QbcTavg_Q_i": "Q_i",
    "bc_cannonsville_storage_pct": "cannonsville_storage_pct",
    "QbcTavg_T_C": "T_C",
    "QbcTavg_T_i": "T_i",
    "QbcTavg_T_L": "Tavg_L",
    "QbcTmax_T_L": "T_L"
})

minmaxscalers = {}
for v in db.columns:
    db[v] = db[v].astype(float)
    scaler = MinMaxScaler()
    db[v] = scaler.fit_transform(db[[v]])
    minmaxscalers[v] = scaler

joblib.dump(minmaxscalers, pn.stage1_thermal_ctrl_decoupled.get() / "minmaxscalers.gz")