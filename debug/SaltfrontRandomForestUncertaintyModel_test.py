import pandas as pd
import pathnavigator
from tqdm import tqdm
if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.rf_model import SaltfrontRandomForestUncertaintyModel

db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_SalinityLSTM.copy()['1979-01-01': '2023-12-31']
folder = "RFModels"

saltfront_rf = SaltfrontRandomForestUncertaintyModel(
    rf_model_saltfront=pn.models.get(folder) / "rf_model_saltfront.gz",
    debug=True
    )
saltfront_rf.load_data(database)

saltfront_rf.update_until(t=3)

saltfront_rf.forecast(t=saltfront_rf.t, quantile=0.95)
saltfront_rf.forecast(t=saltfront_rf.t)


for _ in tqdm(range(20)):
    saltfront_rf.update(t=saltfront_rf.t, quantile=0.95)

for _ in tqdm(range(30)):
    saltfront_rf.update(t=saltfront_rf.t, quantile=None)


r"""
Compute quantile use much more time
with q (500 steps): 6:27
no q (500 steps): 1.17

Try to do batch when it is not within the control period
for loop update is vary slow.
"""