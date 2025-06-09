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
import yaml
import sys 

sys.path.insert(1, "src")
from prep_data import data_prep
from crossval_utils import calc_crossval_splits

# splitting up data based on dates 
k_outer_loop = 5
k_inner_loop = 4 
main_config_file = "model_config.yml" 

with open(main_config_file, 'r') as stream:
    main_config = yaml.safe_load(stream)

crossval_folds = calc_crossval_splits(
    min_date=main_config['min_date'],
    max_date=main_config['max_date'],
    k_outer=k_outer_loop,
    k_inner=k_inner_loop
)

# Create model_config.yml for each k_fold crossval 
for outer_fold in crossval_folds: 
    test_starts = outer_fold['outer_test_start_dates']
    test_ends = outer_fold['outer_test_end_dates'] 
    for inner_fold in outer_fold['inner_folds']: 
        out_file = f"cfg/model_config_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
        new_config = main_config.copy()

        new_config['start_date_train'] = inner_fold['inner_train_start_dates']
        new_config['end_date_train'] = inner_fold['inner_train_end_dates'] 
        new_config['start_date_val'] = inner_fold['inner_val_start_dates'] 
        new_config['end_date_val'] = inner_fold['inner_val_end_dates'] 
        new_config['start_date_test'] = test_starts
        new_config['end_date_test'] = test_ends

        new_config['data_file'] = f"data/data_lordville_2010_2024_log_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.npz"

        with open(out_file, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, indent=4)

        # prepare the dataset based on new splits; write to new datafile 
        prepped_data = data_prep(model_cfg_file=out_file)


# source functions for training 
import pandas as pd
import itertools

# Adding custom paths to the Python path
import sys
sys.path.insert(1, "src")
# Importing the BMI class for running the LSTM model
import torch_bmi
from prep_data import data_prep
from eval_functions import * # see bmi_stream_temp https://code.usgs.gov/wma/wp/bmi-stream-temp/-/blob/main/6_model_eval/src/eval_functions.py?ref_type=heads 

# Hyperparameters to tune 
learning_rate = [0.005, 0.05]
early_stopping = [20, 50]

# Create a product of the hyperparameter sets
hyperparameter_combinations = list(itertools.product(learning_rate, early_stopping))

# Create a df from the hyperparameter combos 
hyperparameter_df = pd.DataFrame(hyperparameter_combinations, columns=['learning_rate', 'early_stopping'])

# hyperparamter tuning and evaluation 
overall_performance = []

# outer loop (5-fold crossval) 
for outer_fold in crossval_folds: 

    # Initialize best hyperparameters and best model
    best_hyperparameters = None
    best_model = None
    best_rmse = 10000   
    
    # Inner loop (4-fold crossval for hyperparameter tuning)
    for inner_fold in outer_fold['inner_folds']:
        # current model config 
        config_file = f"cfg/model_config_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
        
        # Try different hyperparameter combinations
        for index, row in hyperparameter_df.iterrows():

            with open(config_file, 'r') as stream:
                cur_config = yaml.safe_load(stream)
            
            #set current hyperparamters 
            cur_config['early_stopping_patience'] = int(row['early_stopping'])
            cur_config['learn_rate_fine'] = float(row['learning_rate']) 
            cur_config['learn_rate_pre'] = float(row['learning_rate']) 

            # write out temporary config file with current hyperparametrs
            cur_cfg_out_file = f"cfg/tmp/model_config_outer_{outer_fold['outer_fold']}_inner_{inner_fold['inner_fold']}.yml"
            with open(cur_cfg_out_file, 'w') as f:
                yaml.dump(cur_config, f, default_flow_style=False, indent=4)

            # Start a model 
            cur_model_train = torch_bmi.bmi_lstm()
            # Initialize a model instance for training by setting train = True
            cur_model_train.initialize(config_file = cur_cfg_out_file, train = True)

            cur_model_train.train_model()

            # Evaluate model on inner validation data
            preds_val = pd.read_parquet(cur_model_train.val_preds_file, engine='pyarrow')
            preds_val = preds_val.rename(columns={"mean": "pred"})

            metrics = calc_metrics(preds_val)
            
            # Update best hyperparameters and best model if needed
            if metrics['rmse'] < best_rmse:
                best_hyperparameters = row
                best_rmse = metrics['rmse']
                best_cfg_out_file = f"cfg/best/model_config_outer_{outer_fold['outer_fold']}.yml"
                with open(best_cfg_out_file, 'w') as f:
                    yaml.dump(cur_config, f, default_flow_style=False, indent=4)

    best_model = torch_bmi.bmi_lstm()
    best_model.initialize(config_file = best_cfg_out_file, train = True)

    best_model.train_model()

    preds_test = pd.read_parquet(best_model.test_preds_file, engine='pyarrow')
    preds_test = preds_test.rename(columns={"mean": "pred"})
    
    # Evaluate best model on outer test data
    best_metrics = calc_metrics(preds_test)
    
    # Aggregate performance across outer folds
    overall_performance.append(best_metrics.to_frame().T)

overall_performance_df = pd.concat(overall_performance, ignore_index=True)
print(overall_performance_df) 
