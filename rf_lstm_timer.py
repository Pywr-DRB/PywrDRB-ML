import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()

import time
import pandas as pd
import numpy as np
from src.torch_bmi import bmi_lstm
from src.rf_model import RandomForestUncertaintyModel


subfolder = "SalinityLSTM"
model_id = "SalinityLSTM"

config_file = pn.models.get(f"{subfolder}/{model_id}.yml")
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())

start = time.time()
#sim = lstm.simple_run()
lstm.update_until_loop(timestep=100)
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
# Execution time: 0.0406 seconds


db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)
database = db_SalinityLSTM.copy()['1979-01-01': '2023-12-31']

df_res = pd.DataFrame(index=database.index)

rf_settings = {
    "n_estimators": 20,#35,
    "max_features": "sqrt",
    "random_state": 4,
    "n_jobs": -2,
    "max_depth": 10,#14,
    }
n_bootstrap = 100

df = database[["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy", "saltfront", "saltfront_src"]].dropna()
df.loc[df["saltfront_src"] != "obs", "saltfront"] = np.nan
df.dropna(axis=0, how='any', inplace=True)
x_vars = ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"]
y_var = "saltfront"
X = df[x_vars].values
obs = df[y_var].values
rf_model_saltfront = RandomForestUncertaintyModel(x_vars, y_var, **rf_settings)
rf_model_saltfront.fit(X, obs)

start = time.time()
#rf_model_saltfront.predict(X)
for i in range(100):
    rf_model_saltfront.predict(np.reshape(X[i], (1,-1)))
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
# Execution time: 1.4969 seconds

#%%
import platform
import sys
import multiprocessing
import psutil
import torch
import numpy as np
import pandas as pd
import sklearn
import time

def print_env_info():
    print("=== Python Environment Info ===")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Processor: {platform.processor()}")
    print(f"CPU cores (logical): {multiprocessing.cpu_count()}")
    print(f"Memory (GB): {round(psutil.virtual_memory().total / (1024**3), 2)}")
    print()

    print("=== Library Versions ===")
    print(f"Numpy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"Torch: {torch.__version__ if torch else 'N/A'}")
    print(f"Scikit-learn: {sklearn.__version__ if sklearn else 'N/A'}")
    print()

    # Optional: check GPU
    if torch.cuda.is_available():
        print("=== CUDA / GPU Info ===")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA not available.")
    print("==============================\n")

print_env_info()