# script for preparing input data for model training
import os
import numpy as np
import pandas as pd
import torch

from .prep_data_utils import (read_config, separate_trn_tst, scale, convert_batch_reshape, coord_as_reshaped_array)

def data_prep(model_cfg_file, root_dir=None):
    """
    Prepares the dataset for training, validation, and testing of a model by reading, merging,
    and processing water temperature, reservoir releases, and gridmet data.

    This function performs the following steps:
    1. Reads input CSV files for water temperature, reservoir releases, and gridmet data.
    2. Processes and merges these datasets based on a common date index.
    3. Creates lagged features for the specified target variable if applicable.
    4. Splits the dataset into training, validation, and test sets.
    5. Scales the features using z-score normalization.
    6. Reshapes the data for model input.
    7. Saves the processed data to a compressed NumPy file.

    Args:
        args (dict): A dictionary containing configuration parameters, which may include:
            - 'reservoir_data_file' (str): Path to the CSV file containing reservoir release data.
            - 'water_temp_data_file' (str): Path to the CSV file containing water temperature data.
            - 'gridmet_data_file' (str): Path to the CSV file containing GridMET data.
            - 'min_date' (str): Minimum date for the dataset (format: 'YYYY-MM-DD').
            - 'max_date' (str): Maximum date for the dataset (format: 'YYYY-MM-DD').
            - 'time_idx_name' (str): Name of the time index variable in the dataset.
            - 'spatial_idx_name' (str): Name of the spatial index variable in the dataset.
            - 'y_vars' (list): List of target variable names for the model.
            - 'y_vars_src' (list): List of source variable names for the model.
            - 'x_vars' (list): List of predictor variable names for the model.
            - 'lag_days' (int): Number of lag days to apply for the target variable.
            - 'data_file' (str): Path to save the processed data as a compressed NumPy file.
            - 'start_date_train' (str): Start date for the training set.
            - 'end_date_train' (str): End date for the training set.
            - 'start_date_val' (str): Start date for the validation set.
            - 'end_date_val' (str): End date for the validation set.
            - 'start_date_test' (str): Start date for the test set.
            - 'end_date_test' (str): End date for the test set.
            - 'hidden_units' (int): Number of hidden units for the model.

    Returns:
        dict: A dictionary containing processed data, including:
            - 'x_train': Training set features.
            - 'x_val': Validation set features.
            - 'x_test': Test set features.
            - 'x_all_dates': Features for all dates in the dataset.
            - 'x_mean': Mean values of the features.
            - 'x_std': Standard deviation of the features.
            - 'obs_train': Training set observations.
            - 'obs_val': Validation set observations.
            - 'obs_test': Test set observations.
            - 'obs_all_dates': Observations for all dates in the dataset.
            - 'pretrain_train': Pre-training set features.
            - 'pretrain_val': Pre-training set validation features.
            - 'pretrain_test': Pre-training set test features.
            - 'lag_var': Name of the lagged variable.
            - 'lag_var_source': Name of the lagged water temperature source variable.
            - 'lag_var_mean': Mean of the lagged variable.
            - 'lag_var_std': Standard deviation of the lagged variable.
            - Other relevant data structures for model training.

    """
    config = read_config(model_cfg_file)
    args = config

    if root_dir is None:
        root_dir = args.get('root_dir')
    if root_dir is None:
        root_dir = ""
    
    train_dir = os.path.join(root_dir, args['train_dir'])
    if not os.path.exists(train_dir): # Create subfolder in models/
            os.mkdir(train_dir)
            
    # Read in input files for water temperature, reservoir releases, and gridmet
    data_df = pd.read_csv(os.path.join(root_dir, args['input_data_file']), parse_dates=['date'], index_col=['date'])
    data_df = data_df[args['min_date']:args['max_date']]
    data_df = data_df.reset_index()

    # Flag for water temperature source, 1 if obs, 0 otherwise (includes dwallin predictions and a few rows of linear interpolation)
    data_df['y_src'] = np.where(data_df[args['y_vars_src']] == 'obs', 1, 0)

    # Convert the merged DataFrame into an xarray Dataset
    data_df.set_index([args['time_idx_name'], args['spatial_idx_name']], inplace=True)

    # lag the water temp and source input feature by n lag days if using as predictor variable
    if args['y_vars'][0] in args['x_vars']:
        lag_var = f"{args['y_vars'][0]}_lag_{args['lag_days']}"
        lag_var_source = f"y_src_lag_{args['lag_days']}"
        # shift the autoregressive variable by n lag days
        data_df[lag_var] = data_df[args['y_vars']].shift(args['lag_days'])
        data_df[lag_var_source] = data_df['y_src'].shift(args['lag_days'])
        # if we have another LSTM layer predicting the delta temperature, calculate delta temp
        if args['delta_temp_layer']:
            data_df['delta_temp'] = data_df[args['y_vars']].diff()
        # slice the dataset to get rid of new NA's created by lagging the autoregresive variable
        data_df = data_df.iloc[args['lag_days']:]

        # Remove y_vars from x_vars and add the new lagged variables
        new_x_vars = [var for var in args['x_vars'] if var not in args['y_vars'] + ['y_src']] + [lag_var, lag_var_source]
        args['x_vars'] = new_x_vars
        lag_var_pos = np.where(args['x_vars'] == np.atleast_1d(lag_var))[0]
        lag_var_source_pos = np.where(args['x_vars'] == np.atleast_1d(lag_var_source))[0]
    else:
        lag_var = None
        lag_var_source = None
        lag_var_pos = None
        lag_var_source_pos = None
        # if we have another LSTM layer predicting the delta temperature, calculate delta temp
        if args['delta_temp_layer']:
            data_df['delta_temp'] = data_df[args['y_vars']].diff()
            data_df = data_df.iloc[1:]
    
    if args['y_vars_src'][0] in args['x_vars']:
        data_df[args['y_vars_src']] = np.where(data_df[args['y_vars_src']] == 'obs', 1, 0)
        
    x_dataset = data_df.loc[:, args['x_vars']]
    obs_dataset = data_df.loc[:, args['y_vars'] + ['y_src']]
    pretrain_dataset = data_df.loc[:, args['y_vars']]
    # setting water temperature to NA if it isn't an observation
    obs_dataset.loc[obs_dataset['y_src'] != 1, args['y_vars']] = np.nan
    obs_dataset.drop(columns=['y_src'], inplace=True)
    if args['delta_temp_layer']:
        x_delta_temp_dataset = data_df.loc[:, args['delta_temp_vars']]

    # turn into xarray datasets
    x_xr = x_dataset.to_xarray()
    obs_xr = obs_dataset.to_xarray()
    pretrain_xr = pretrain_dataset.to_xarray()
    if args['delta_temp_layer']:
        x_delta_temp_xr = x_delta_temp_dataset.to_xarray()

    # scale, etc...
    x_train, x_val, x_test = separate_trn_tst(x_xr,
                                              args['time_idx_name'],
                                              args['start_date_train'],
                                              args['end_date_train'],
                                              args['start_date_val'],
                                              args['end_date_val'],
                                              args['start_date_test'],
                                              args['end_date_test'],
                                              args['spatial_idx_name'])

    if args['delta_temp_layer']:
        x_delta_train, x_delta_val, x_delta_test = separate_trn_tst(x_delta_temp_xr,
                                                                    args['time_idx_name'],
                                                                    args['start_date_train'],
                                                                    args['end_date_train'],
                                                                    args['start_date_val'],
                                                                    args['end_date_val'],
                                                                    args['start_date_test'],
                                                                    args['end_date_test'],
                                                                    args['spatial_idx_name'])
        x_delta_all_dates, _, _ = separate_trn_tst(x_delta_temp_xr,
                                                args['time_idx_name'],
                                                args['min_date'], # min and max of dataset
                                                args['max_date'],
                                                args['start_date_val'],
                                                args['end_date_val'],
                                                args['start_date_test'],
                                                args['end_date_test'],
                                                args['spatial_idx_name'])
    else:
        x_delta_train = None
        x_delta_val = None
        x_delta_test = None
        x_delta_all_dates = None

    # x_data used for predicting across all times
    x_all_dates, _, _ = separate_trn_tst(x_xr,
                                        args['time_idx_name'],
                                        args['min_date'], # min and max of dataset
                                        args['max_date'],
                                        args['start_date_val'],
                                        args['end_date_val'],
                                        args['start_date_test'],
                                        args['end_date_test'],
                                        args['spatial_idx_name'])

    pretrain_train, pretrain_val, pretrain_test = separate_trn_tst(pretrain_xr,
                                                                args['time_idx_name'],
                                                                args['start_date_train'],
                                                                args['end_date_train'],
                                                                args['start_date_val'],
                                                                args['end_date_val'],
                                                                args['start_date_test'],
                                                                args['end_date_test'],
                                                                args['spatial_idx_name'],
                                                                args['y_vars'])

    obs_train, obs_val, obs_test = separate_trn_tst(obs_xr,
                                                    args['time_idx_name'],
                                                    args['start_date_train'],
                                                    args['end_date_train'],
                                                    args['start_date_val'],
                                                    args['end_date_val'],
                                                    args['start_date_test'],
                                                    args['end_date_test'],
                                                    args['spatial_idx_name'],
                                                    args['y_vars'])

    obs_all_dates, _, _ = separate_trn_tst(obs_xr,
                                            args['time_idx_name'],
                                            args['min_date'], # min and max of dataset
                                            args['max_date'],
                                            args['start_date_val'],
                                            args['end_date_val'],
                                            args['start_date_test'],
                                            args['end_date_test'],
                                            args['spatial_idx_name'])

    # z-scoring input features
    x_train_scl, x_std, x_mean = scale(x_train)

    # z-scoring if val and test are available, use standard deviation and mean from x training set for scaling
    if x_val:
        x_val_scl, _, _ = scale(x_val, std=x_std, mean=x_mean)
    else:
        x_val_scl = None

    if x_test:
        x_test_scl, _, _ = scale(x_test, std=x_std, mean=x_mean)
    else:
        x_test_scl = None

    if x_all_dates:
        x_all_dates_scl, _, _ = scale(x_all_dates, std=x_std, mean=x_mean)
    else:
        x_all_dates_scl = None

    if x_delta_train:
        x_delta_train_scl, x_delta_std, x_delta_mean = scale(x_delta_train)
    else:
        x_delta_train_scl = None

    if x_delta_val:
        x_delta_val_scl, _, _ = scale(x_delta_val, std=x_delta_std, mean=x_delta_mean)
    else:
        x_delta_val_scl = None

    if x_delta_test:
        x_delta_test_scl, _, _ = scale(x_delta_test, std=x_delta_std, mean=x_delta_mean)
    else:
        x_delta_test_scl = None

    if x_delta_all_dates:
        x_delta_all_dates_scl, _, _ = scale(x_delta_all_dates, std=x_delta_std, mean=x_delta_mean)
    else:
        x_delta_all_dates_scl = None

    if lag_var is not None:
        lag_var_mean = x_mean[lag_var].values
        lag_var_std = x_std[lag_var].values
    else:
        lag_var_mean = np.nan
        lag_var_std = np.nan

    seq_len = args.get('seq_len', 365)  # LSTM memory length
    offset = args.get('offset', 1.0)    # For data augmentation, how much to offset the time index
    spatial_idx_name = args.get('spatial_idx_name', 'seg_id_nat')
    time_idx_name = args.get('time_idx_name', 'date')

    x_data_dict = {
        "x_train": convert_batch_reshape(
            dataset=x_train_scl,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "x_val": convert_batch_reshape(
            dataset=x_val_scl,            
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "x_test": convert_batch_reshape(
            dataset=x_test_scl,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "x_all_dates": convert_batch_reshape(
            dataset=x_all_dates_scl,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=len(x_all_dates_scl['date']),
            fill_batch=False
            ),
        "x_std": x_std.to_array().values,
        "x_mean": x_mean.to_array().values,
        "x_vars": np.array(args['x_vars']),
        
        "ids_train": coord_as_reshaped_array(
            dataset=x_train_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "times_train": coord_as_reshaped_array(
            dataset=x_train_scl,
            coord_name=time_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset,
            fill_time = True),
        "padded_train": coord_as_reshaped_array(
            dataset=x_train_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset,
            fill_pad=True),
        "ids_val": coord_as_reshaped_array(
            dataset=x_val_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "times_val": coord_as_reshaped_array(
            dataset=x_val_scl,
            coord_name=time_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "padded_val": coord_as_reshaped_array(
            dataset=x_val_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset,
            fill_pad=True
            ),
        "ids_test": coord_as_reshaped_array(
            dataset=x_test_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "times_test": coord_as_reshaped_array(
            dataset=x_test_scl,
            coord_name=time_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "padded_test": coord_as_reshaped_array(
            dataset=x_test_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset,
            fill_pad=True
            ),
        "ids_all_dates": coord_as_reshaped_array(
            dataset=x_all_dates_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=len(x_all_dates_scl['date']),
            fill_batch=False
            ),
        "times_all_dates": coord_as_reshaped_array(
            dataset=x_all_dates_scl,
            coord_name=time_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=len(x_all_dates_scl['date']),
            fill_batch=False
            ),
        "padded_all_dates": coord_as_reshaped_array(
            dataset=x_all_dates_scl,
            coord_name=spatial_idx_name,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=len(x_all_dates_scl['date']),
            fill_batch=False,
            fill_pad=True
            ),
        "lag_var": lag_var,
        "lag_var_source": lag_var_source,
        "lag_var_pos": lag_var_pos,
        "lag_var_source_pos": lag_var_source_pos,
        "lag_var_mean": lag_var_mean,
        "lag_var_std": lag_var_std
    }

    if args['delta_temp_layer']:
        x_delta_data_dict = {
            "x_delta_train": convert_batch_reshape(
                dataset=x_delta_train_scl,            
                spatial_idx_name=spatial_idx_name,
                time_idx_name=time_idx_name,
                seq_len=seq_len,
                offset=offset
                ),
            "x_delta_val": convert_batch_reshape(
                dataset=x_delta_val_scl,
                spatial_idx_name=spatial_idx_name,
                time_idx_name=time_idx_name,
                seq_len=seq_len,
                offset=offset
                ),
            "x_delta_test": convert_batch_reshape(
                dataset=x_delta_test_scl,            
                spatial_idx_name=spatial_idx_name,
                time_idx_name=time_idx_name,
                seq_len=seq_len,
                offset=offset
                ),
            "x_delta_all_dates": convert_batch_reshape(
                dataset=x_delta_all_dates_scl,
                spatial_idx_name=spatial_idx_name,
                time_idx_name=time_idx_name,
                seq_len=len(x_delta_all_dates_scl['date']),
                fill_batch=False
                ),
            "x_delta_vars": np.array(args['delta_temp_vars']),
            "x_delta_std": x_delta_std.to_array().values,
            "x_delta_mean": x_delta_mean.to_array().values,
        }

    states_dict = {
        "h_train": torch.zeros(x_data_dict["x_train"].shape[0], args['hidden_units']),
        "c_train": torch.zeros(x_data_dict["x_train"].shape[0], args['hidden_units']),
        "h_val": torch.zeros(x_data_dict["x_val"].shape[0], args['hidden_units']),
        "c_val": torch.zeros(x_data_dict["x_val"].shape[0], args['hidden_units']),
        "h_test": torch.zeros(x_data_dict["x_test"].shape[0], args['hidden_units']),
        "c_test": torch.zeros(x_data_dict["x_test"].shape[0], args['hidden_units']),
        "h_all_dates": torch.zeros(x_data_dict["x_all_dates"].shape[0], args['hidden_units']),
        "c_all_dates": torch.zeros(x_data_dict["x_all_dates"].shape[0], args['hidden_units'])
    }

    weighting_matrix_dict = {
        # fake weighting matrix to make work with our LSTM
        "weighting_matrix_train": np.array(0, ndmin=2),
        "weighting_matrix_val": np.array(0, ndmin=2),
        "weighting_matrix_test": np.array(0, ndmin=2)
    }

    pretrain_train_scl, pretrain_std, pretrain_mean = scale(pretrain_train)

    if pretrain_val:
        pretrain_val_scl, _, _ = scale(pretrain_val, std=pretrain_std, mean=pretrain_mean)
    else:
        pretrain_val_scl = None

    if pretrain_test:
        pretrain_test_scl, _, _ = scale(pretrain_test, std=pretrain_std, mean=pretrain_mean)
    else:
        pretrain_test_scl = None

    pretrain_data_dict = {
        "pretrain_train": convert_batch_reshape(
            dataset=pretrain_train,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "pretrain_val": convert_batch_reshape(
            dataset=pretrain_val,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "pretrain_test": convert_batch_reshape(
            dataset=pretrain_test,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "pretrain_std": pretrain_std.to_array().values,
        "pretrain_mean": pretrain_mean.to_array().values,
        "pretrain_obs_vars": args['y_vars']
    }

    obs_train_scl, obs_std, obs_mean = scale(obs_train)

    if obs_val:
        obs_val_scl, _, _ = scale(obs_val, std=obs_std, mean=obs_mean)
    else:
        obs_val_scl = None

    if obs_test:
        obs_test_scl, _, _ = scale(obs_test, std=obs_std, mean=obs_mean)
    else:
        obs_test_scl = None

    if obs_all_dates:
        obs_all_dates_scl, _, _ = scale(obs_all_dates, std=obs_std, mean=obs_mean)
    else:
        obs_all_dates_scl = None

    obs_data_dict = {
        "obs_train": convert_batch_reshape(
            dataset=obs_train,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "obs_val": convert_batch_reshape(
            dataset=obs_val,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "obs_test": convert_batch_reshape(
            dataset=obs_test,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=seq_len,
            offset=offset
            ),
        "obs_all_dates": convert_batch_reshape(
            dataset=obs_all_dates,
            spatial_idx_name=spatial_idx_name,
            time_idx_name=time_idx_name,
            seq_len=len(x_all_dates_scl['date']),
            fill_batch=False
            ),
        "obs_std": obs_std.to_array().values,
        "obs_mean": obs_mean.to_array().values,
        "obs_vars": args['y_vars']
    }

    if args['delta_temp_layer']:
        all_data = {**x_data_dict, **pretrain_data_dict, **obs_data_dict, **x_delta_data_dict,
                    **states_dict, **weighting_matrix_dict}
    else:
        all_data = {**x_data_dict, **pretrain_data_dict, **obs_data_dict, **states_dict, **weighting_matrix_dict}

    np.savez_compressed(args['data_file'], **all_data)

    return all_data


if __name__ == '__main__':
    # configuration file path
    model_cfg_file = r"C:\Users\CL\Documents\GitHub\FlowTemp\models/model_config.yml"
    root_dir = r"C:\Users\CL\Documents\GitHub\FlowTemp"
    data_prep(model_cfg_file, root_dir)
