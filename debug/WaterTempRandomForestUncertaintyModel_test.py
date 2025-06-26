import pandas as pd
import pathnavigator
from tqdm import tqdm
if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.rf_model import WaterTempRandomForestUncertaintyModel

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_TempLSTM.copy()['1979-01-01': '2023-12-31']
folder = "RFModels"

temp_rf = WaterTempRandomForestUncertaintyModel(
    rf_model1=pn.models.get(folder) / "rf_model1.gz",
    rf_model2=pn.models.get(folder) / "rf_model2.gz",
    rf_model_map=pn.models.get(folder) / "rf_model_map.gz",
    debug=True
    )
temp_rf.load_data(database)

temp_rf.update_until(t=3)

temp_rf.forecast(t=temp_rf.t, quantile=0.95)
temp_rf.forecast(t=temp_rf.t)

temp_rf.forecast_T_L_arr
temp_rf.forecast_T_L_lb_arr
temp_rf.forecast_T_L_ub_arr
# for _ in tqdm(range(500)):
#     temp_rf.update(t=temp_rf.t, quantile=0.95)

for _ in tqdm(range(500)):
    temp_rf.update(t=temp_rf.t, quantile=None)


r"""
Compute quantile use much more time
with q (500 steps): 6:27
no q (500 steps): 1.17

Try to do batch when it is not within the control period
for loop update is vary slow.
"""