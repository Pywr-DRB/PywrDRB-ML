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
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import itertools
import torch
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.torch_bmi import bmi_lstm
from src.prep_data import data_prep
from src.crossval_utils import calc_crossval_splits
from src.model_builder import make_lstm_model

# It need 365 as a training seuence length
date_range = pd.date_range(start='1980-01-01', end='2023-12-31', freq='D').to_list()

# splitting up data based on dates
k_outer_loop = 5
k_inner_loop = 4
subfolder = "TempLSTM_CrossVali_loop"
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

# Hyperparameters to tune
learning_rate = [0.005, 0.005, 0.05]
early_stopping = [20, 50]
dropout_rate = [0, 0.1, 0.3]
seed = [4, 2, 5]
# Create a product of the hyperparameter sets
hyperparameter_combinations = list(itertools.product(learning_rate, early_stopping, dropout_rate, seed))
# Create a df from the hyperparameter combos
hyperparameter_df = pd.DataFrame(hyperparameter_combinations, columns=['learning_rate', 'early_stopping', 'dropout_rate', 'seed'])

# Create model_config.yml for each k_fold crossval
model1_ids = []
for outer_fold in tqdm(crossval_folds):
    test_starts = outer_fold['outer_test_start_dates']
    test_ends = outer_fold['outer_test_end_dates']
    for inner_fold in outer_fold['inner_folds']:
        # LSTM1 model configuration
        lstm1_config = deepcopy(config_template)
        lstm1_config.update(lstm1_settings)
        lstm1_config['start_date_train'] = inner_fold['inner_train_start_dates']
        lstm1_config['end_date_train'] = inner_fold['inner_train_end_dates']
        lstm1_config['start_date_val'] = inner_fold['inner_val_start_dates']
        lstm1_config['end_date_val'] = inner_fold['inner_val_end_dates']
        lstm1_config['start_date_test'] = test_starts
        lstm1_config['end_date_test'] = test_ends
        for index, row in hyperparameter_df.iterrows():
            cur_config = deepcopy(lstm1_config)
            model_id = lstm1_config["model_id"]
            model_id += f"_O{outer_fold['outer_fold']}_I{inner_fold['inner_fold']}_lr_{row['learning_rate']:.3f}_es_{row['early_stopping']:.0f}_dr_{row['dropout_rate']:.1f}_s_{row['seed']:.0f}"
            cur_config["model_id"] = model_id

            #set current hyperparamters
            cur_config['early_stopping_patience'] = int(row['early_stopping'])
            cur_config['learn_rate_fine'] = float(row['learning_rate'])
            cur_config['learn_rate_pre'] = float(row['learning_rate'])
            cur_config['dropout_rate'] = float(row['dropout_rate'])
            cur_config['seed'] = float(row['seed'])

            lstm_config_file = make_lstm_model(subfolder=subfolder, yml_subsubfolder="cfg", **cur_config)
            _ = data_prep(lstm_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile
            model1_ids.append(model_id)
            
model2_ids = []
for outer_fold in tqdm(crossval_folds):
    test_starts = outer_fold['outer_test_start_dates']
    test_ends = outer_fold['outer_test_end_dates']
    for inner_fold in outer_fold['inner_folds']:
        # LSTM1 model configuration
        lstm2_config = deepcopy(config_template)
        lstm2_config.update(lstm2_settings)
        lstm2_config['start_date_train'] = inner_fold['inner_train_start_dates']
        lstm2_config['end_date_train'] = inner_fold['inner_train_end_dates']
        lstm2_config['start_date_val'] = inner_fold['inner_val_start_dates']
        lstm2_config['end_date_val'] = inner_fold['inner_val_end_dates']
        lstm2_config['start_date_test'] = test_starts
        lstm2_config['end_date_test'] = test_ends
        for index, row in hyperparameter_df.iterrows():
            cur_config = deepcopy(lstm2_config)
            model_id = lstm2_config["model_id"]
            model_id += f"_O{outer_fold['outer_fold']}_I{inner_fold['inner_fold']}_lr_{row['learning_rate']:.3f}_es_{row['early_stopping']:.0f}_dr_{row['dropout_rate']:.1f}_s_{row['seed']:.0f}"
            cur_config["model_id"] = model_id

            #set current hyperparamters
            cur_config['early_stopping_patience'] = int(row['early_stopping'])
            cur_config['learn_rate_fine'] = float(row['learning_rate'])
            cur_config['learn_rate_pre'] = float(row['learning_rate'])
            cur_config['dropout_rate'] = float(row['dropout_rate'])
            cur_config['seed'] = float(row['seed'])

            lstm_config_file = make_lstm_model(subfolder=subfolder, yml_subsubfolder="cfg", **cur_config)
            _ = data_prep(lstm_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile
            model2_ids.append(model_id)
#%%
for model_id in tqdm(model1_ids):
    config_file = pn.models.get(f'{subfolder}/cfg/{model_id}.yml')
    cur_model_train = bmi_lstm()
    cur_model_train.initialize(config_file=config_file, train=True, disable_tqdm=True, root_dir=pn.get())
    cur_model_train.train_model()

for model_id in tqdm(model2_ids):
    config_file = pn.models.get(f'{subfolder}/cfg/{model_id}.yml')
    cur_model_train = bmi_lstm()
    cur_model_train.initialize(config_file=config_file, train=True, disable_tqdm=True, root_dir=pn.get())
    cur_model_train.train_model()
#%% Test run
# config_file = pn.models.get(f'{subfolder}/cfg/{model_id}.yml')

# cur_model_train = bmi_lstm()
# # Initialize a model instance for training by setting train = True
# cur_model_train.initialize(config_file=config_file, train=True, disable_tqdm=True, root_dir=pn.get())
# cur_model_train.train_model()


# Evaluate model on inner validation data
#preds_val = pd.read_parquet(cur_model_train.val_preds_file, engine='pyarrow')
#preds_val = preds_val.rename(columns={"mean": "pred"})
