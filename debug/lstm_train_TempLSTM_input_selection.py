import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, eval_TempLSTM, return_sim_obs_pair_for_T_L

# from src.model_builder import config_template

config_template = {
    'input_data_file': "data/database/TempLSTM_database.csv",
    'x_vars': [],
    'y_vars': [],
    'y_vars_src': [],
    'lag_days': 1,
    'min_date': '1979-01-01',
    'max_date': '2023-12-31',
    'start_date_train': '1979-10-01',
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
    'weight_threshold': 20,
    'weight_value': 2,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.1,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 4
    }

lstm1_settings = {
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_C"],
    "y_vars_src": ["tavg_water_cannonsville_src"],
    'min_date': '1979-01-01',
    'max_date': '2023-12-31',
    'start_date_train': '1979-12-31', # Can add reservoir_io_sntemp.csv for pretrain
    'end_date_train': '2023-12-31',
    'start_date_val': '2017-01-01',
    'end_date_val': '2017-12-31',
    'start_date_test': '2017-01-01',
    'end_date_test': '2017-12-31',
    }

lstm2_settings = {
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "QbcTavg_Q_i", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_i"],
    "y_vars_src": ["tavg_water_src"],
    'min_date': '2006-07-31',
    'max_date': '2023-12-31',
    'start_date_train': '2006-12-31', #'1992-10-01', # Can add reservoir_io_sntemp.csv for pretrain T_i is constrainted by T_L & T_C
    'end_date_train': '2023-12-31',
    'start_date_val': '2017-01-01',
    'end_date_val': '2017-12-31',
    'start_date_test': '2017-01-01',
    'end_date_test': '2017-12-31',
    }

#%%
subfolder = "TempLSTMGapFiller"
lstm1_config = deepcopy(config_template)
lstm1_config.update(lstm1_settings)
lstm2_config = deepcopy(config_template)
lstm2_config.update(lstm2_settings)
lstm1_config_file = make_lstm_model(model_id="TempLSTM1", subfolder=subfolder, **lstm1_config)
lstm2_config_file = make_lstm_model(model_id="TempLSTM2", subfolder=subfolder, **lstm2_config)

#%%
r"""
from src.prep_data import data_prep
df = data_prep(lstm1_config_file, root_dir)
df = data_prep(lstm2_config_file, root_dir)
"""

#%%
#subfolder = "TempLSTM_noQc"
model_ids = ["TempLSTM1", "TempLSTM2"]
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False)

#%%
#subfolder = "TempLSTM_noQc"
model_ids = ["TempLSTM1", "TempLSTM2"]
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, disable=False)

#%%
df_metric_train = loop_to_eval_lstm_models(lstms, period="train", only_months=None, mode="TempLSTM", disable=False)

#%%
df_metric = eval_TempLSTM(
    lstm1=lstms["TempLSTM1"], 
    lstm2=lstms["TempLSTM2"],
    period="all", only_months=None, disable=False
    )


df_metric_summer = eval_TempLSTM(
    lstm1=lstms["TempLSTM1"], 
    lstm2=lstms["TempLSTM2"],
    period="all", only_months=[6,7,8], disable=False
    )

with open(pn.models.get(f"{subfolder}") / "eval_metrics.txt", 'w') as f:
    f.write(f"subfolder: {subfolder}\n")
    
    f.write("=== df_metric ===\n")
    df_metric.to_csv(f, sep='\t', index=True)
    
    f.write("\n\n=== df_metric_summer ===\n")
    df_metric_summer.to_csv(f, sep='\t', index=True)
    
    f.write("\n\n=== lstm1_config ===\n")
    for key, value in lstm1_config.items():
        f.write(f"{key}: {value}\n")
        
    f.write("\n\n=== lstm2_config ===\n")
    for key, value in lstm2_config.items():
        f.write(f"{key}: {value}\n")
#%%
# import clt
# df_obs, df_sim = return_sim_obs_pair_for_T_L(lstm1=lstms["TempLSTM1"], lstm2=lstms["TempLSTM2"])

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(4, 4))
# x = df_obs["T_L"]
# y = df_sim["T_L"]
# x, y = clt.utils.dropna_any(x, y)
# clt.plots.scatter(ax, x=x, y=y, s=2)
# ax.set_xlabel("Observed Tmax at Lordville")
# ax.set_ylabel("Predicted Tmax at Lordville")
# ax.legend()
# plt.tight_layout()
# plt.show()



#%% plot bar plot

rmse = {
    "TempLSTM (lag1 + Q_C)": 1.26,
    "TempLSTM (lag1)": 1.55,
    "TempLSTM (Q_C)": 1.26,
    "TempLSTM (None)": 1.1
    }












