import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == "Darwin":
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair

# from src.model_builder import config_template

config_template = {
    'input_data_file': "data/database/SalinityLSTM_database.csv",
    'x_vars': [],
    'y_vars': [],
    'y_vars_src': [],
    'lag_days': 1,
    'min_date': '1979-01-01',
    'max_date': '2023-12-31',
    'start_date_train': '1979-01-01',
    'end_date_train': '2023-12-31',
    'start_date_val': '2017-01-01',
    'end_date_val': '2017-12-31',
    'start_date_test': '2017-01-01',
    'end_date_test': '2017-12-31',
    'pre_train': True,
    'fine_tune': True,
    'learn_rate_pre': 0.05,
    'learn_rate_fine': 0.05,
    'n_epochs_pre': 50,
    'n_epochs_fine': 350,
    'early_stopping_patience': 50,
    'hidden_units': 16,
    'head_hidden_units': 16,
    'head_n_distr': 1,
    'weight_loss': True,
    'weight_threshold': 80,
    'weight_value': 5,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.1,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 2
    }

lstm_settings = {
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

# lstm_lag1_settings = {
#     "x_vars": ["01463500_bc", "01474500_bc", "saltfront", "doy"],
#     "y_vars": ["saltfront"],
#     "y_vars_src": ["saltfront_src"],
#     }

#%%
subfolder = "SalinityLSTMGapFiller"
lstm_config = deepcopy(config_template)
lstm_config.update(lstm_settings)
lstm_config_file = make_lstm_model(model_id="SalinityLSTM", subfolder=subfolder, **lstm_config)

# Have gap in saltfront
# lstm_config = deepcopy(config_template)
# lstm_config.update(lstm_lag1_settings)
# lstm_lag1_config_file = make_lstm_model(model_id="LSTM_bc_lag1", subfolder=subfolder, **lstm_config)

#%%
from src.prep_data import data_prep
df = data_prep(lstm_config_file, root_dir)
#df = data_prep(lstm_lag1_config_file, root_dir)

#%%
#subfolder = None #"SalinityLSTM"
model_ids = ["SalinityLSTM"]
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=True)

#%%
#subfolder = None #"SalinityLSTM"
model_ids = ["SalinityLSTM"] #, "LSTM_bc_lag1"]
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, mode="SalinityLSTM", disable=False, overwrite=False)

#%%
df_metric_train = loop_to_eval_lstm_models(lstms, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM")

#%% Fill data
import pandas as pd
pairs = return_sim_obs_pair(lstms, subfolder=subfolder, period="all", only_months=None, mode="SalinityLSTM")
lstm_simed_saltfront = pairs["SalinityLSTM"][0].to_frame("saltfront")
lstm_simed_saltfront.index.name = "date"
lstm_simed_saltfront.to_csv(pn.data.raw.get() / "lstm_simed_saltfront.csv")
#%%
import matplotlib.pyplot as plt
import clt
sim, obs = pairs['SalinityLSTM']
sim, obs = clt.dropna_any(sim, obs)
fig, ax = plt.subplots()
clt.plots.scatter(ax, x=obs, y=sim, s=1, alpha=0.5,)
# ax.scatter(x=obs, y=sim, s=1, alpha=0.5)
# clt.ax.add_45_degree_ref_line(ax)
# clt.ax.add_linear_regr_line(ax, sim, obs,)
ax.set_xlabel("Observed (RM)")
ax.set_ylabel("Predicted (RM)")
ax.legend()
plt.tight_layout()
plt.show()

#%%
sim, obs = pairs['SalinityLSTM']
for y in range(1979, 2024):
    fig, ax = plt.subplots()
    ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
    ax.plot(sim[f"{y}":f"{y}"], c="b", alpha=0.8, label="Predicted")
    ax.set_ylim([40, 100])
    ax.set_xlabel("Date")
    ax.set_ylabel("Salt front location (RM)")
    ax.legend()
    plt.tight_layout()
    plt.show()




















