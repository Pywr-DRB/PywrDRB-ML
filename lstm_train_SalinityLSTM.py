import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair
from src.prep_data import data_prep

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
    'weight_value': 2,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.30,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 2,
    }
lstm_settings = {
    "model_id": "SalinityLSTM",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

#%%
subfolder = "SalinityLSTM"
lstm_config = deepcopy(config_template)
lstm_config.update(lstm_settings)
lstm_config_file = make_lstm_model(subfolder=subfolder, **lstm_config)
_ = data_prep(lstm_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile

#%%
model_ids = ["SalinityLSTM"]
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=True)
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, mode="SalinityLSTM", disable=False, overwrite=True)

#%%
df_metric_train = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM")
r"""

Out[4]: '\n                 nrmse         r        r2      rmse\
nmodel_id\nSalinityLSTM  0.084535  0.884838  0.777871  3.014507\n'

                 nrmse         r        r2      rmse
model_id
SalinityLSTM  0.084535  0.884838  0.777871  3.014507
"""
# #%%
# import matplotlib.pyplot as plt
# import clt
# pairs = return_sim_obs_pair(lstms, period="train", only_months=None, mode="SalinityLSTM", disable=False)

# #%%
# sim, obs = pairs['SalinityLSTM']
# sim, obs = clt.dropna_any(sim, obs)
# fig, ax = plt.subplots()
# clt.plots.scatter(ax, x=obs, y=sim, s=1, alpha=0.5,)
# # ax.scatter(x=obs, y=sim, s=1, alpha=0.5)
# # clt.ax.add_45_degree_ref_line(ax)
# # clt.ax.add_linear_regr_line(ax, sim, obs,)
# ax.set_xlabel("Observed (RM)")
# ax.set_ylabel("Predicted (RM)")
# ax.legend()
# plt.tight_layout()
# plt.show()

# #%%
# sim, obs = pairs['SalinityLSTM']
# for y in range(1964, 2024):
#     fig, ax = plt.subplots()
#     ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
#     ax.plot(sim[f"{y}":f"{y}"], c="b", alpha=0.8, label="Predicted")
#     ax.set_ylim([40, 100])
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Salt front location (RM)")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()



#%% Check new old 
# import pandas as pd

# new = pd.read_csv(pn.data.raw.get() / "pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])["1979":"2024"][["flow_delTrenton", "flow_outletSchuylkill"]]
# new.columns = ["Q_Trenton_bc", "Q_Schuylkill_bc"]

# old = pd.read_csv("/Users/cl/Downloads/pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])["1979":"2024"][["flow_delTrenton", "flow_outletSchuylkill"]]
# old.columns = ["Q_Trenton_bc", "Q_Schuylkill_bc"]

# db = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), parse_dates=True, index_col=["date"])["1979":"2024"][["Q_Trenton_bc", "Q_Schuylkill_bc"]]


# gauge = "Q_Trenton_bc"
# gauge = "Q_Schuylkill_bc"

# df_compare = pd.concat([
#     new[gauge].rename("new"),
#     old[gauge].rename("old"),
#     db[gauge].rename("db")
# ], axis=1).dropna()

# # Correlation matrix
# corr = df_compare.corr()
# print(corr)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].scatter(df_compare["old"], df_compare["new"], s=5)
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[0].set_xlabel(f"Old {gauge}")
# ax[0].set_ylabel(f"New {gauge}")
# ax[0].set_title("New vs Old")

# # --- New vs DB ---
# ax[1].scatter(df_compare["db"], df_compare["new"], s=5)
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# ax[1].set_xlabel(f"DB {gauge}")
# ax[1].set_ylabel(f"New {gauge}")
# ax[1].set_title("New vs DB")

# plt.tight_layout()
# plt.show()







