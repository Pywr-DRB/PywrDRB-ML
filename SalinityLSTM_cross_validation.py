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

# It need 365 as a training seuence length
date_range = pd.date_range(start='1980-01-01', end='2023-12-31', freq='D').to_list()

# splitting up data based on dates
k_outer_loop = 5
k_inner_loop = 4
subfolder = "SalinityLSTM_CrossVali"
pn.models.mkdir(f"{subfolder}/cfg")

crossval_folds = calc_crossval_splits(
    date_range=date_range,
    k_outer=k_outer_loop,
    k_inner=k_inner_loop
)

#%% Basic model configuration
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
    'seed': 4,
    }
lstm_settings = {
    "model_id": "SalinityLSTM",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

# Create model_config.yml for each k_fold crossval
for outer_fold in tqdm(crossval_folds):
    test_starts = outer_fold['outer_test_start_dates']
    test_ends = outer_fold['outer_test_end_dates']
    for inner_fold in outer_fold['inner_folds']:

        # LSTM model configuration
        lstm_config = deepcopy(config_template)
        lstm_config.update(lstm_settings)
        model_id = lstm_config["model_id"]
        model_id += f"_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}"
        lstm_config["model_id"] = model_id

        lstm_config['start_date_train'] = inner_fold['inner_train_start_dates']
        lstm_config['end_date_train'] = inner_fold['inner_train_end_dates']
        lstm_config['start_date_val'] = inner_fold['inner_val_start_dates']
        lstm_config['end_date_val'] = inner_fold['inner_val_end_dates']
        lstm_config['start_date_test'] = test_starts
        lstm_config['end_date_test'] = test_ends

        lstm_config_file = make_lstm_model(subfolder=subfolder, yml_subsubfolder="cfg", **lstm_config)
        _ = data_prep(lstm_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile

#%%
def run_inner_loop(inner_folds, hyperparameter_df, SalinityLSTM):

    # Initialize best hyperparameters and best rmse
    best_hyperparameters = None
    best_rmse = 10000
    best_config = None
    best_cfg_out_file = None
    # Inner loop (4-fold crossval for hyperparameter tuning)
    for inner_fold in tqdm(inner_folds, desc="Inner"):
        # current model config
        config_file = pn.models.get(f"{subfolder}/cfg/{SalinityLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml")
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
            cur_cfg_out_file = pn.models.get(f"{subfolder}/tmp") / f"{SalinityLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
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
                best_cfg_out_file = pn.models.get(f"{subfolder}/best") / f"{SalinityLSTM}_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
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


# Read in db
db = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
db.loc[db['saltfront_src'] != "obs", "saltfront"] = np.nan

# Hyperparameters to tune
learning_rate = [0.005, 0.05] #[0.005, 0.05]
early_stopping = [20, 50] #[20, 50]
dropout_rate = [0, 0.1, 0.3]
# Create a product of the hyperparameter sets
hyperparameter_combinations = list(itertools.product(learning_rate, early_stopping, dropout_rate))

# Create a df from the hyperparameter combos
hyperparameter_df = pd.DataFrame(hyperparameter_combinations, columns=['learning_rate', 'early_stopping', 'dropout_rate'])

# hyperparamter tuning and evaluation
overall_performance = []
overall_best_hyperparameters = []

# outer loop (5-fold crossval)
pn.models.mkdir(f"{subfolder}/tmp")
pn.models.mkdir(f"{subfolder}/best")

for outer_fold in tqdm(crossval_folds, desc="Outer"):

    # Initialize best hyperparameters and best model
    best_model = None

    best_cfg_out_file, best_hyperparameters = run_inner_loop(inner_folds=outer_fold['inner_folds'],
                                        hyperparameter_df=hyperparameter_df, SalinityLSTM="SalinityLSTM")
    overall_best_hyperparameters.append(best_hyperparameters)

    # LSTM
    best_model = bmi_lstm()
    best_model.initialize(config_file = best_cfg_out_file, train = True, disable_tqdm = True)
    best_model.train_model()
    preds_test = pd.read_parquet(best_model.test_preds_file, engine='pyarrow')
    preds_test = preds_test.rename(columns={"mean": "pred"})
    best_metrics = calc_metrics(preds_test)
    overall_performance.append(best_metrics.to_frame().T)

overall_performance_df = pd.concat(overall_performance, ignore_index=True)
overall_performance_df.to_csv(pn.models.get() / f"{subfolder}/overall_performance.csv", index=False)

mean_rmse = {}
mean_rmse["saltfront"] = overall_performance_df.mean()["rmse"]

r"""
{'saltfront': np.float64(5.855963998878859)}

"""

overall_best_hyperparameters_df = pd.concat(overall_best_hyperparameters, ignore_index=True, axis=1)

r"""
                    0      1      2      3      4
learning_rate    0.05   0.05   0.05   0.05   0.05
early_stopping  50.00  50.00  50.00  20.00  20.00
dropout_rate     0.30   0.30   0.10   0.30   0.30

"""

preds_test[["pred", "obs"]].plot()


