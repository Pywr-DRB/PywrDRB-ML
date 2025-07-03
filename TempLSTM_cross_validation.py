#%% Importing necessary libraries and setting up the environment
# k-fold cross validation training.
# Summary / overview: this performs k-fold cross-validation for hyperparameter
# tuning and model evaluation of the bmi lstm. It first creates separate model
# configuration files for each k-fold iteration, then prepares the dataset for
# each iteration. The script then tries different hyperparameter combinations,
# trains the model, and evaluates its performance on the inner validation data.
# The best hyperparameters and model performance are tracked and updated across
# inner folds. After hyperparameter tuning, the script trains the best model and
# evaluates its performance on the outer test data. The overall performance metrics
# are aggregated across outer folds and stored in a DataFrame. This process allows
# for robust hyperparameter tuning and model evaluation using k-fold
# cross-validation, providing a reliable estimate of the model's performance on
# unseen data.
import pandas as pd
import yaml
from copy import deepcopy
from tqdm import tqdm
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.prep_data import data_prep
from src.crossval_utils import calc_crossval_splits
from src.model_builder import make_lstm_model
#db = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
#db = db[db['tmmx_water_src'] == "obs"]
#db = db[db['tavg_water_src'] == "obs"]

# It need 365 as a training seuence length
date_range = pd.date_range(start='1994-01-01', end='1995-12-31', freq='D').to_list() + \
             pd.date_range(start='2006-01-01', end='2023-12-31', freq='D').to_list()

# splitting up data based on dates
k_outer_loop = 5
k_inner_loop = 4
subfolder = "TempLSTM_CrossVali"
pn.models.mkdir(f"{subfolder}/cfg")

crossval_folds = calc_crossval_splits(
    date_range=date_range,
    k_outer=k_outer_loop,
    k_inner=k_inner_loop
)

#%% Basic model configuration
config_template = {
    'input_data_file': "data/database/TempLSTM_database.csv",
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
    'weight_threshold': 20,
    'weight_value': 2,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0.3,
    'dropout_rate': 0.1,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 4,
    }
lstm1_settings = {
    "model_id": "TempLSTM1",
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "bc_cannonsville_storage_pct", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_C"],
    "y_vars_src": ["tavg_water_cannonsville_src"],
    }
lstm2_settings = {
    "model_id": "TempLSTM2",
    "x_vars": ["tmmn", "tmmx", "pr", "srad", "QbcTavg_Q_i", "doy", "QbcTavg_Q_C"],
    "y_vars": ["QbcTavg_T_i"],
    "y_vars_src": ["tavg_water_src"],
    }

# Create model_config.yml for each k_fold crossval
for outer_fold in tqdm(crossval_folds):
    test_starts = outer_fold['outer_test_start_dates']
    test_ends = outer_fold['outer_test_end_dates']
    for inner_fold in outer_fold['inner_folds']:

        # LSTM1 model configuration
        lstm1_config = deepcopy(config_template)
        lstm1_config.update(lstm1_settings)
        model_id = lstm1_config["model_id"]
        model_id += f"_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}"
        lstm1_config["model_id"] = model_id

        lstm1_config['start_date_train'] = inner_fold['inner_train_start_dates']
        lstm1_config['end_date_train'] = inner_fold['inner_train_end_dates']
        lstm1_config['start_date_val'] = inner_fold['inner_val_start_dates']
        lstm1_config['end_date_val'] = inner_fold['inner_val_end_dates']
        lstm1_config['start_date_test'] = test_starts
        lstm1_config['end_date_test'] = test_ends

        lstm1_config_file = make_lstm_model(subfolder=subfolder, yml_subsubfolder="cfg", **lstm1_config)
        _ = data_prep(lstm1_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile

        # LSTM2 model configuration
        lstm2_config = deepcopy(config_template)
        lstm2_config.update(lstm2_settings)
        model_id = lstm2_config["model_id"]
        model_id += f"_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}"
        lstm2_config["model_id"] = model_id

        lstm2_config['start_date_train'] = inner_fold['inner_train_start_dates']
        lstm2_config['end_date_train'] = inner_fold['inner_train_end_dates']
        lstm2_config['start_date_val'] = inner_fold['inner_val_start_dates']
        lstm2_config['end_date_val'] = inner_fold['inner_val_end_dates']
        lstm2_config['start_date_test'] = test_starts
        lstm2_config['end_date_test'] = test_ends
        lstm2_config_file = make_lstm_model(subfolder=subfolder, yml_subsubfolder="cfg", **lstm2_config)
        _ = data_prep(lstm2_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile

#%%
def run_inner_loop(inner_folds, hyperparameter_df, TempLSTM):

    # Initialize best hyperparameters and best rmse
    best_hyperparameters = None
    best_rmse = 10000
    best_config = None
    best_cfg_out_file = None
    # Inner loop (4-fold crossval for hyperparameter tuning)
    for inner_fold in tqdm(inner_folds, desc="Inner"):
        # current model config
        config_file = pn.models.get(f"{subfolder}/cfg/{TempLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml")
        # Try different hyperparameter combinations
        for index, row in hyperparameter_df.iterrows():

            with open(config_file, 'r') as stream:
                cur_config = yaml.safe_load(stream)

            #set current hyperparamters
            cur_config['early_stopping_patience'] = int(row['early_stopping'])
            cur_config['learn_rate_fine'] = float(row['learning_rate'])
            cur_config['learn_rate_pre'] = float(row['learning_rate'])
            cur_config['dropout_rate'] = float(row['dropout_rate'])

            # write out temporary config file with current hyperparametrs
            cur_cfg_out_file = pn.models.get(f"{subfolder}/tmp") / f"{TempLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
            with open(cur_cfg_out_file, 'w') as f:
                yaml.dump(cur_config, f, default_flow_style=False, indent=4)

            # Start a model
            cur_model_train = bmi_lstm()
            # Initialize a model instance for training by setting train = True
            cur_model_train.initialize(config_file = cur_cfg_out_file, train = True, disable_tqdm = True)
            cur_model_train.train_model()

            # Evaluate model on inner validation data
            preds_val = pd.read_parquet(cur_model_train.val_preds_file, engine='pyarrow')
            preds_val = preds_val.rename(columns={"mean": "pred"})

            metrics = calc_metrics(preds_val)

            # Update best hyperparameters and best model if needed
            if metrics['rmse'] < best_rmse:
                best_hyperparameters = row
                best_rmse = metrics['rmse']
                best_cfg_out_file = pn.models.get(f"{subfolder}/best") / f"{TempLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
                best_config = deepcopy(cur_config)
    with open(best_cfg_out_file, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, indent=4)

    return best_cfg_out_file, best_hyperparameters
#%%
# source functions for training
import pandas as pd
import numpy as np
import itertools
from src.torch_bmi import bmi_lstm
from src.torch_eval_functions import calc_metrics
from src.lstm_model import WaterTempLSTMModel
import json


# Read in db
db = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
db.loc[db['tavg_water_src'] != "obs", "QbcTavg_T_L"] = np.nan
db.loc[db['tmmx_water_src'] != "obs", "QbcTmax_T_L"] = np.nan

with open(pn.get() / "models/TempLSTM/Tavg2Tmax_coefs.json", "r") as file:
    Tavg2Tmax_coefs = json.load(file)

# Hyperparameters to tune
learning_rate = [0.005, 0.05] #[0.005, 0.05]
early_stopping = [20, 50] #[20, 50]
dropout_rate = [0, 0.1, 0.3]
# Create a product of the hyperparameter sets
hyperparameter_combinations = list(itertools.product(learning_rate, early_stopping, dropout_rate))

# Create a df from the hyperparameter combos
hyperparameter_df = pd.DataFrame(hyperparameter_combinations, columns=['learning_rate', 'early_stopping', 'dropout_rate'])

# hyperparamter tuning and evaluation
overall_performance1 = []
overall_performance2 = []
overall_performance_Tavg = []
overall_performance_Tmax = []
overall_best_hyperparameters1 = []
overall_best_hyperparameters2 = []

# outer loop (5-fold crossval)
pn.models.mkdir(f"{subfolder}/tmp")
pn.models.mkdir(f"{subfolder}/best")

for outer_fold in tqdm(crossval_folds, desc="Outer"):

    # Initialize best hyperparameters and best model
    best_model1 = None
    best_model2 = None

    best_cfg_out_file1, best_hyperparameters1 = run_inner_loop(inner_folds=outer_fold['inner_folds'],
                                        hyperparameter_df=hyperparameter_df, TempLSTM="TempLSTM1")

    best_cfg_out_file2, best_hyperparameters2 = run_inner_loop(inner_folds=outer_fold['inner_folds'],
                                        hyperparameter_df=hyperparameter_df, TempLSTM="TempLSTM2")
    # Save best hyperparameters for each model
    overall_best_hyperparameters1.append(best_hyperparameters1)
    overall_best_hyperparameters2.append(best_hyperparameters2)

    # LSTM1
    best_model1 = bmi_lstm()
    best_model1.initialize(config_file = best_cfg_out_file1, train = True, disable_tqdm = True)
    best_model1.train_model()
    preds_test1 = pd.read_parquet(best_model1.test_preds_file, engine='pyarrow')
    preds_test1 = preds_test1.rename(columns={"mean": "pred"})
    best_metrics1 = calc_metrics(preds_test1)
    overall_performance1.append(best_metrics1.to_frame().T)

    # LSTM2
    best_model2 = bmi_lstm()
    best_model2.initialize(config_file = best_cfg_out_file2, train = True, disable_tqdm = True)
    best_model2.train_model()
    preds_test2 = pd.read_parquet(best_model2.test_preds_file, engine='pyarrow')
    preds_test2 = preds_test2.rename(columns={"mean": "pred"})
    best_metrics2 = calc_metrics(preds_test2)
    overall_performance2.append(best_metrics2.to_frame().T)

    # Tavg
    Tavg_L_mu, Tavg_L_sd = WaterTempLSTMModel.blend_hot_cold_water(
        T_C_mu=preds_test1["pred"],
        T_i_mu=preds_test2["pred"],
        T_C_sd=preds_test1["sd"],
        T_i_sd=preds_test2["sd"],
        Q_C=db.loc[preds_test1["date"], "QbcTavg_Q_C"].values,
        Q_i=db.loc[preds_test1["date"], "QbcTavg_Q_i"].values
        )
    preds_Tavg = preds_test2.copy()
    preds_Tavg["pred"] = Tavg_L_mu
    preds_Tavg["sd"] = Tavg_L_sd
    preds_Tavg["obs"] = db.loc[preds_test1["date"], "QbcTavg_T_L"].values
    best_metrics_Tavg = calc_metrics(preds_Tavg)
    overall_performance_Tavg.append(best_metrics_Tavg.to_frame().T)

    # Tmax
    T_L_mu, T_L_sd = WaterTempLSTMModel.tavg2tmax(Tavg_L_mu, Tavg2Tmax_coefs)
    T_L_sd += Tavg_L_sd
    preds_Tmax = preds_test2.copy()
    preds_Tmax["pred"] = T_L_mu
    preds_Tmax["sd"] = T_L_sd
    preds_Tmax["obs"] = db.loc[preds_test1["date"], "QbcTmax_T_L"].values
    best_metrics_Tmax = calc_metrics(preds_Tmax)
    overall_performance_Tmax.append(best_metrics_Tmax.to_frame().T)

overall_performance1_df = pd.concat(overall_performance1, ignore_index=True)
overall_performance2_df = pd.concat(overall_performance2, ignore_index=True)
overall_performance_Tavg_df = pd.concat(overall_performance_Tavg, ignore_index=True)
overall_performance_Tmax_df = pd.concat(overall_performance_Tmax, ignore_index=True)

overall_performance1_df.to_csv(pn.models.get() / f"{subfolder}/overall_performance1.csv", index=False)
overall_performance2_df.to_csv(pn.models.get() / f"{subfolder}/overall_performance2.csv", index=False)
overall_performance_Tavg_df.to_csv(pn.models.get() / f"{subfolder}/overall_performance_Tavg.csv", index=False)
overall_performance_Tmax_df.to_csv(pn.models.get() / f"{subfolder}/overall_performance_Tmax.csv", index=False)

mean_rmse = {}
mean_rmse["T_C"] = overall_performance1_df.mean()["rmse"]
mean_rmse["T_i"] = overall_performance2_df.mean()["rmse"]
mean_rmse["Tavg"] = overall_performance_Tavg_df.mean()["rmse"]
mean_rmse["T_L"] = overall_performance_Tmax_df.mean()["rmse"]

r"""
{'T_C': np.float64(1.612883998379305),
 'T_i': np.float64(6.684021409680599),
 'Tavg': np.float64(1.2969361086676146),
 'T_L': np.float64(1.4062992125475333)}
"""

overall_best_hyperparameters1_df = pd.concat(overall_best_hyperparameters1, ignore_index=True, axis=1)
overall_best_hyperparameters2_df = pd.concat(overall_best_hyperparameters2, ignore_index=True, axis=1)
overall_best_hyperparameters1_df.to_csv(pn.models.get() / f"{subfolder}/overall_best_hyperparameters1.csv", index=False)
overall_best_hyperparameters2_df.to_csv(pn.models.get() / f"{subfolder}/overall_best_hyperparameters2.csv", index=False)

r"""
                     0       1      2      3      4
learning_rate    0.005   0.005   0.05   0.05   0.05
early_stopping  50.000  50.000  50.00  50.00  50.00
dropout_rate     0.100   0.300   0.10   0.30   0.00

                    0      1      2      3      4
learning_rate    0.05   0.05   0.05   0.05   0.05
early_stopping  50.00  50.00  50.00  20.00  50.00
dropout_rate     0.00   0.00   0.00   0.00   0.00
"""




