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

from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days

database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']

ml_model_noCtrl = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model_noCtrl.load_data(database)
ml_model_noCtrl.update_until(date=pd.Timestamp('2024-01-01'))  # Update until the end of 2023
df_noCtrl = pd.DataFrame(ml_model_noCtrl.records, index=ml_model_noCtrl.dates)

Jrel_noCtrl = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
Jadd_noCtrl = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)

Jrel_noCtrl_arr = compute_reliability(df_noCtrl, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=True)
Jadd_noCtrl_arr = compute_max_annual_accumulated_degree_days(df_noCtrl, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=True)


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

scaler = MinMaxScaler()
scaler.fit_transform(Jadd_noCtrl_arr.reshape(-1, 1))
minmaxscalers["Jadd"] = scaler

scaler = MinMaxScaler()
scaler.fit_transform(Jrel_noCtrl_arr.reshape(-1, 1))
minmaxscalers["Jrel"] = scaler

joblib.dump(minmaxscalers, pn.get("thermal_ctrl_decoupled") / "minmaxscalers.gz")