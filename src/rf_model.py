r"""
This code defines a Random Forest model that estimates uncertainty using bootstrapping.
The uncertainty estimation method is based on the following paper:

Coulston, J. W., Blinn, C. E., Thomas, V. A., & Wynne, R. H. (2016). Approximating 
Prediction Uncertainty for Random Forest Regression Models. Photogrammetric Engineering 
& Remote Sensing, 82(3), 189–197. https://doi.org/10.14358/PERS.82.3.189

This class can be saved by joblib and loaded later for predictions.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from collections import deque

class RandomForestUncertaintyModel:
    def __init__(self, x_vars, y_var, **rf_settings):
        """
        Random Forest model with uncertainty estimation.
        This model uses bootstrapping to estimate the uncertainty of predictions.
        It can also compute cross-validation RMSE for model evaluation.
        
        Parameters
        ==============
        x_vars: list of str
            List of feature variable names used for training the model.
        y_var: str
            The target variable name used for training the model.
        rf_settings: dict
            Settings for the Random Forest model. Common parameters include:
            - n_estimators: int, default=100
                The number of trees in the forest.
            - max_features: int or float, default='sqrt'
                The number of features to consider when looking for the best split.
            - n_jobs: int
                The number of jobs to run in parallel.
            - random_state: int, optional
                Controls the randomness of the estimator.
        """
        # Record variable names
        self.x_vars = x_vars
        self.y_var = y_var
        
        # General parameters for the Random Forest model
        self.rf_settings = rf_settings
        self.random_state = rf_settings.get('random_state', None)
        self.rf_final = None
        
        # For training RMSE
        self.rmse_train = None
        self.rmse_test = None
        
        # For uncertainty estimation
        self.n_bootstrap = None
        self.tau_hat = None
        self.tau_list = None
        
        # For cross-validation RMSE
        self.rmse_cross_vali = None
        self.rmse_cross_vali_test = None
        self.rmses = None
        self.n_splits = None
        self.shuffle = None

    def fit(self, X, y, test_size=None, n_jobs=-2, overwrite=False, shuffle=True):
        """
        Fit the final Random Forest model on the full dataset.
        Parameters
        ==============
        X: array-like, shape (n_samples, n_features)
            The input features.
        y: array-like, shape (n_samples,)
            The target values.
        test_size: float, optional
            If provided, the data will be split into training and test sets for evaluation.
        overwrite: bool, default=False
            If True, overwrite the existing rf_final model if it has already been fitted.
        shuffle: bool, default=True
            If True, shuffle the data before splitting into training and test sets.
        """
        if self.rf_final is not None and not overwrite:
            print("rf_final already fitted. Use overwrite=True to fit again.")
            print (f"RMSE on training set: {self.rmse_train:.3f}, RMSE on test set: {self.rmse_test:.3f}")
            return None
        
        random_state = self.random_state
        if test_size is not None:
            if shuffle:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=shuffle
                    )
            # 
            rf = RandomForestRegressor(**self.rf_settings)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_train)
            rmse_train = root_mean_squared_error(y_train, y_pred)
            
            y_pred = rf.predict(X_test)
            rmse_test = root_mean_squared_error(y_test, y_pred)
            print (f"RMSE on training set: {rmse_train:.3f}, RMSE on test set: {rmse_test:.3f}")
            self.rmse_train = rmse_train
            self.rmse_test = rmse_test
            
        else:
            # Fit final RF model on full data
            rf = RandomForestRegressor(**self.rf_settings)
            rf.fit(X, y)
            y_pred = rf.predict(X)
            rmse_train = root_mean_squared_error(y, y_pred)
            print(f"RMSE on training set: {rmse_train:.3f}")
            self.rmse_train = rmse_train
            
        self.rf_final = rf
     
    def compute_tau(self, X, y, n_bootstrap=1000, disable=True, overwrite=False):
        """
        Compute the tau values for uncertainty estimation using bootstrapping.
        
        Parameters
        ==============
        X: array-like, shape (n_samples, n_features)
            The input features.
        y: array-like, shape (n_samples,)
            The target values.
        n_bootstrap: int, default=1000
            The number of bootstrap samples to draw.
        disable: bool, default=True
            If True, disable the progress bar.
        overwrite: bool, default=False
            If True, overwrite the existing tau_list if it has already been computed.
        """
        if self.tau_list is not None and not overwrite:
            print("tau_list already computed. Use overwrite=True to recompute.")
            return None
            
        rng = np.random.RandomState(self.random_state)
        rf_settings = self.rf_settings
        n_samples = X.shape[0]
        tau_list = []
        for _ in tqdm(range(n_bootstrap), disable=disable,  desc="compute_tau"):
            # Bootstrap resample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)

            X_boot, y_boot = X[indices], y[indices]
            X_oob, y_oob = X[oob_indices], y[oob_indices]

            # Train RF model on bootstrap sample
            rf = RandomForestRegressor(**rf_settings)
            rf.fit(X_boot, y_boot)

            # Get predictions from all trees for all OOB samples at once
            tree_preds = np.stack([tree.predict(X_oob) for tree in rf.estimators_], axis=1)  # shape: (n_oob, n_trees)
            y_pred_mean = np.mean(tree_preds, axis=1)
            y_pred_var = np.var(tree_preds, axis=1)
            mask = y_pred_var > 0
            tau = ((y_oob[mask] - y_pred_mean[mask]) ** 2) / y_pred_var[mask]
            tau = np.sqrt(tau)
            tau_list += tau.tolist()
        self.tau_list = tau_list
        return None
    
    def downsample_tau(self, size=5000):
        """
        Downsample the tau_list to a specified size.
        
        Parameters
        ==============
        size: int, default=1000
            The desired size of the downsampled tau_list.
        """
        if self.tau_list is None:
            raise ValueError("tau_list has not been computed yet. Call compute_tau() first.")
        
        if len(self.tau_list) <= size:
            print("tau_list is already smaller than or equal to the desired size.")
            return None
        
        rng = np.random.RandomState(self.random_state)
        downsampled_tau = rng.choice(self.tau_list, size=size, replace=False)
        self.tau_list = downsampled_tau.tolist()
        return None
    
    def predict(self, X, quantile=None):
        """
        Predict using the fitted Random Forest model and estimate uncertainty if quantile is provided.
        
        Parameters
        ==============
        X: array-like, shape (n_samples, n_features)
            The input features for prediction.
        quantile: float, optional
            If provided, compute the uncertainty bounds based on the quantile of tau_list.
            Should be between 0 and 1 (e.g., 0.95 for 95% confidence interval).
        """
        preds = self.rf_final.predict(X)
        if quantile is not None and self.tau_list is not None:
            tree_preds = np.stack([tree.predict(X) for tree in self.rf_final.estimators_], axis=0)
            preds_std = np.std(tree_preds, axis=0)
            tau_hat = np.percentile(self.tau_list, quantile)
            half_width = tau_hat * preds_std
            lower = preds - half_width
            upper = preds + half_width
            return preds, lower, upper
        else:
            return preds
        
    def cross_val_rmse(self, X, y, n_splits=5, shuffle=True, test_size=None, overwrite=False):
        if self.rmse_cross_vali is not None and not overwrite:
            print("mean_rmse already computed. Use overwrite=True to recompute.")
            print(f"Mean RMSE from cross-validation: {self.rmse_cross_vali:.3f}")
            return None
        
        random_state = self.random_state
        rf_settings = self.rf_settings
        
        if test_size is not None:
            if shuffle:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=shuffle
                    )
            X_ = X_train
            y_ = y_train
        else:
            X_ = X
            y_ = y
        
        if shuffle:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle)
        rmses = []
        for train_idx, test_idx in kf.split(X_):
            X_train, X_vali = X_[train_idx], X_[test_idx]
            y_train, y_vali = y_[train_idx], y_[test_idx]
            rf = RandomForestRegressor(**rf_settings)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_vali)
            rmse = root_mean_squared_error(y_vali, y_pred)
            rmses.append(rmse)
            
        if test_size is not None:
            # Evaluate on the test set if provided
            y_pred_test = rf.predict(X_test)
            rmse_test = root_mean_squared_error(y_test, y_pred_test)
            print(f"RMSE on test set: {rmse_test:.3f}")
            self.rmse_cross_vali_test = rmse_test
        
        # To keep the record
        mean_rmse = np.mean(rmses)
        self.rmse_cross_vali = mean_rmse
        self.rmses = rmses
        self.n_splits = n_splits
        self.shuffle = shuffle
        print(f"Mean RMSE from cross-validation: {mean_rmse:.3f}")
        return None
    
    def compute_rmse(self, X, y):
        """
        Compute the RMSE of the model on the given data.

        Parameters
        ==============
        X: array-like, shape (n_samples, n_features)
            The input features.
        y: array-like, shape (n_samples,)
            The target values.
        """
        if self.rf_final is None:
            raise ValueError("Model has not been fitted yet. Call fit() before compute_rmse().")
        
        y_pred = self.rf_final.predict(X)
        rmse = root_mean_squared_error(y, y_pred)
        print(f"RMSE: {rmse:.3f}")
        return rmse
    
    def save(self, filename):
        """
        Save the model to a file using joblib.
        """
        joblib.dump(self, filename)
        print(f"Model saved to {filename}")
    
    @staticmethod
    def load(filename):
        """
        Load a RandomForestUncertaintyModel from a file using joblib.
        """
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    
class WaterTempRandomForestUncertaintyModel:
    def __init__(self, rf_model1, rf_model2, rf_model_map, debug=False):
        """
        WaterTempRandomForestUncertaintyModel class for temperature prediction using Random Forest models.
        
        Parameters
        ==============
        rf_model1: str
            Path to the first Random Forest model for cold water temperature prediction.
        rf_model2: str
            Path to the second Random Forest model for hot water temperature prediction.
        rf_model_map: str
            Path to the Random Forest model for mapping average temperature to local temperature.
        debug: bool, default=False
            If True, enables debug mode which records intermediate values for debugging purposes.
        """

        # RF models
        self.rf_model1 = joblib.load(rf_model1)
        self.rf_model2 = joblib.load(rf_model2)
        self.rf_model_map = joblib.load(rf_model_map)
        
        # Input data
        self.X_1 = np.nan
        self.X_2 = np.nan
        self.X_map = np.nan
        self.Q_C = np.nan
        self.Q_i = np.nan
        self.Q_L = np.nan
        
        # Dates
        self.start_date = None
        self.end_date = None
        self.length = None
        
        # Current predictions
        self.T_C = np.nan
        self.T_C_lb = np.nan
        self.T_C_ub = np.nan
        self.T_i = np.nan
        self.T_i_lb = np.nan
        self.T_i_ub = np.nan
        self.Tavg_L = np.nan
        self.Tavg_L_lb = np.nan
        self.Tavg_L_ub = np.nan
        self.T_L = np.nan
        self.T_L_lb = np.nan
        self.T_L_ub = np.nan
        
        # Forecast predictions
        self.forecast_T_L_arr = [np.nan]
        self.forecast_T_L_lb_arr = [np.nan]
        self.forecast_T_L_ub_arr = [np.nan]
        
        # Time step
        self.t = 0
        self.current_date = None
        
        # Debug mode
        self.debug = debug
        if debug:
            self.records = {}
            self.forecast_records = {}
    
    def load_data(self, database, start_date='1979-01-01', end_date='2023-12-31'):
        """
        Load data from the database for the specified date range.
        Parameters
        ==============
        database: pd.DataFrame
            The database containing the meteological, temperature and flow data.
        start_date: str, default='1979-01-01'
            The start date for the data to be loaded.
        end_date: str, default='2023-12-31'
            The end date for the data to be loaded.
        """
        
        self.start_date = pd.to_datetime(start_date) 
        self.current_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) 
        
        db = database[start_date:end_date]
        self.X_1 = db[self.rf_model1.x_vars].values
        self.X_2 = db[self.rf_model2.x_vars].values
        self.X_map = db[self.rf_model_map.x_vars].values
        
        # I use the default name here (pd.Series)
        self.Q_C = db["QbcTavg_Q_C"].values
        self.Q_i = db["QbcTavg_Q_i"].values
        
        length = db.shape[0]
        self.length = length
        if self.debug:
            self.records = {
                "Q_C": [np.nan] * length,
                "Q_i": [np.nan] * length,
                "cannonsville_storage_pct": [np.nan] * length,
                "T_C": [np.nan] * length,
                "T_C_lb": [np.nan] * length,
                "T_C_ub": [np.nan] * length,
                "T_i": [np.nan] * length,
                "T_i_lb": [np.nan] * length,
                "T_i_ub": [np.nan] * length,
                "Tavg_L": [np.nan] * length,
                "Tavg_L_lb": [np.nan] * length,
                "Tavg_L_ub": [np.nan] * length,
                "T_L": [np.nan] * length,
                "T_L_lb": [np.nan] * length,
                "T_L_ub": [np.nan] * length
            }
            self.forecast_records = {
                "Q_C": [np.nan] * length,
                "Q_i": [np.nan] * length,
                "cannonsville_storage_pct": [np.nan] * length,
                "T_C": [np.nan] * length,
                "T_C_lb": [np.nan] * length,
                "T_C_ub": [np.nan] * length,
                "T_i": [np.nan] * length,
                "T_i_lb": [np.nan] * length,
                "T_i_ub": [np.nan] * length,
                "Tavg_L": [np.nan] * length,
                "Tavg_L_lb": [np.nan] * length,
                "Tavg_L_ub": [np.nan] * length,
                "T_L": [np.nan] * length,
                "T_L_lb": [np.nan] * length,
                "T_L_ub": [np.nan] * length
            }
    
    def update(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, quantile=None):
        """
        Update the model with new data for a specific time step.
        
        Parameters
        ==============
        t: int
            The time step index to update the model.
        Q_C: float, optional
            The flow of cold water at time t.
        Q_i: float, optional
            The flow of hot water at time t.
        cannonsville_storage_pct: float, optional
            The percentage of Cannonsville storage at time t.
        quantile: float, optional
            If provided, compute the uncertainty bounds based on the quantile of tau_list.
            Should be between 0 and 1 (e.g., 0.95 for 95% confidence interval).
        """
        if Q_C is not None:
            self.Q_C[t] = Q_C
            try:
                self.X_1[t, self.rf_model1.x_vars.index("QbcTavg_Q_C")] = Q_C
            except ValueError:
                print("Warning: 'QbcTavg_Q_C' not found in rf_model1.x_vars. Skipping update.")
            try:
                self.X_2[t, self.rf_model2.x_vars.index("QbcTavg_Q_C")] = Q_C
            except ValueError:
                print("Warning: 'QbcTavg_Q_C' not found in rf_model2.x_vars. Skipping update.")
            
        if Q_i is not None:
            self.Q_i[t] = Q_i
            try:
                self.X_2[t, self.rf_model2.x_vars.index("QbcTavg_Q_i")] = Q_i
            except ValueError:
                print("Warning: 'QbcTavg_Q_i' not found in rf_model2.x_vars. Skipping update.")
            
        if cannonsville_storage_pct is not None:
            try:
                self.X_1[t, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")] = cannonsville_storage_pct
            except ValueError:
                print("Warning: 'bc_cannonsville_storage_pct' not found in rf_model1.x_vars. Skipping update.")
                
        X_1 = self.X_1[t].reshape(1, -1)
        X_2 = self.X_2[t].reshape(1, -1)
        X_map = self.X_map[t].reshape(1, -1)
        
        if quantile is not None: 
            T_C, T_C_lb, T_C_ub = self.rf_model1.predict(X_1, quantile=quantile)
            T_i, T_i_lb, T_i_ub = self.rf_model2.predict(X_2, quantile=quantile)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=self.Q_C[t], Q_i=self.Q_i[t])
            Tavg_L_lb = self.blend_hot_cold_water(T_C=T_C_lb, T_i=T_i_lb, Q_C=self.Q_C[t], Q_i=self.Q_i[t])
            Tavg_L_ub = self.blend_hot_cold_water(T_C=T_C_ub, T_i=T_i_ub, Q_C=self.Q_C[t], Q_i=self.Q_i[t])
            
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L, _, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_lb
            _, T_L_lb, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_ub
            _, _, T_L_ub = self.rf_model_map.predict(X_map, quantile=quantile)
            
            if self.debug:
                self.records["Q_C"][t] = self.Q_C[t]
                self.records["Q_i"][t] = self.Q_i[t]
                self.records["cannonsville_storage_pct"][t] = self.X_1[t, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.records["T_C"][t] = T_C[0]
                self.records["T_C_lb"][t] = T_C_lb[0]
                self.records["T_C_ub"][t] = T_C_ub[0]
                self.records["T_i"][t] = T_i[0]
                self.records["T_i_lb"][t] = T_i_lb[0]
                self.records["T_i_ub"][t] = T_i_ub[0]
                self.records["Tavg_L"][t] = Tavg_L[0]
                self.records["Tavg_L_lb"][t] = Tavg_L_lb[0]
                self.records["Tavg_L_ub"][t] = Tavg_L_ub[0]
                self.records["T_L"][t] = T_L[0]
                self.records["T_L_lb"][t] = T_L_lb[0]
                self.records["T_L_ub"][t] = T_L_ub[0]
        else:
            T_C = self.rf_model1.predict(X_1)
            T_i = self.rf_model2.predict(X_2)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=self.Q_C[t], Q_i=self.Q_i[t])
            
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L = self.rf_model_map.predict(X_map)
            
            if self.debug:
                self.records["Q_C"][t] = self.Q_C[t]
                self.records["Q_i"][t] = self.Q_i[t]
                self.records["cannonsville_storage_pct"][t] = self.X_1[t, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.records["T_C"][t] = T_C[0]
                self.records["T_i"][t] = T_i[0]
                self.records["Tavg_L"][t] = Tavg_L[0]
                self.records["T_L"][t] = T_L[0]
            T_C_lb = [np.nan]
            T_C_ub = [np.nan]
            T_i_lb = [np.nan]
            T_i_ub = [np.nan]
            Tavg_L_lb = [np.nan]
            Tavg_L_ub = [np.nan]
            T_L_lb = [np.nan]
            T_L_ub = [np.nan]
        
        self.T_C = T_C[0]
        self.T_C_lb = T_C_lb[0]
        self.T_C_ub = T_C_ub[0]
        self.T_i = T_i[0]
        self.T_i_lb = T_i_lb[0]
        self.T_i_ub = T_i_ub[0]
        self.Tavg_L = Tavg_L[0]
        self.Tavg_L_lb = Tavg_L_lb[0]
        self.Tavg_L_ub = Tavg_L_ub[0]
        self.T_L = T_L[0]
        self.T_L_lb = T_L_lb[0]
        self.T_L_ub = T_L_ub[0]
        
        self.t += 1
        self.current_date += pd.Timedelta(days=1)
        return self.T_L, self.T_L_lb, self.T_L_ub
    
    def forecast(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, quantile=None, lead_time=0):
        """
        Forecast the temperature for a given time step with a specified lead time.
        Parameters
        ==============
        t: int
            The time step index to forecast from.
        Q_C: float, optional
            The flow of cold water (Cannonsville release) at time t.
        Q_i: float, optional
            The flow of hot water (East branch) at time t.
        cannonsville_storage_pct: float, optional
            The percentage of Cannonsville storage at time t.
        quantile: float, optional
            If provided, compute the uncertainty bounds based on the quantile of tau_list.
            Should be between 0 and 1 (e.g., 0.95 for 95% confidence interval).
        lead_time: int, default=0
            The number of time steps to forecast ahead. If 0, only the current time step is forecasted.
        """
        X_1 = self.X_1[t:t+lead_time+1].reshape(lead_time+1, -1)
        X_2 = self.X_2[t:t+lead_time+1].reshape(lead_time+1, -1)
        X_map = self.X_map[t:t+lead_time+1].reshape(lead_time+1, -1)
        Q_C_ = self.Q_C[t:t+lead_time+1]
        Q_i_ = self.Q_i[t:t+lead_time+1]
        
        if Q_C is not None:
            Q_C_[0] = Q_C
            try:
                X_1[0, self.rf_model1.x_vars.index("QbcTavg_Q_C")] = Q_C
            except ValueError:
                print("Warning: 'QbcTavg_Q_C' not found in rf_model1.x_vars. Skipping update.")
            try:
                X_2[0, self.rf_model2.x_vars.index("QbcTavg_Q_C")] = Q_C
            except ValueError:
                print("Warning: 'QbcTavg_Q_C' not found in rf_model2.x_vars. Skipping update.")
            
        if Q_i is not None:
            Q_i_[0] = Q_i
            try:
                X_2[0, self.rf_model2.x_vars.index("QbcTavg_Q_i")] = Q_i
            except ValueError:
                print("Warning: 'QbcTavg_Q_i' not found in rf_model2.x_vars. Skipping update.")
            
        if cannonsville_storage_pct is not None:
            try:
                X_1[0, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")] = cannonsville_storage_pct
            except ValueError:
                print("Warning: 'bc_cannonsville_storage_pct' not found in rf_model1.x_vars. Skipping update.")
        
        if quantile is not None: 
            T_C, T_C_lb, T_C_ub = self.rf_model1.predict(X_1, quantile=quantile)
            T_i, T_i_lb, T_i_ub = self.rf_model2.predict(X_2, quantile=quantile)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=Q_C_, Q_i=Q_i_)
            Tavg_L_lb = self.blend_hot_cold_water(T_C=T_C_lb, T_i=T_i_lb, Q_C=Q_C_, Q_i=Q_i_)
            Tavg_L_ub = self.blend_hot_cold_water(T_C=T_C_ub, T_i=T_i_ub, Q_C=Q_C_, Q_i=Q_i_)
            
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L, _, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_lb
            _, T_L_lb, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_ub
            _, _, T_L_ub = self.rf_model_map.predict(X_map, quantile=quantile)
            
            if self.debug:
                self.forecast_records["Q_C"][t] = Q_C_[0]
                self.forecast_records["Q_i"][t] = Q_i_[0]
                self.forecast_records["cannonsville_storage_pct"][t] = X_1[0, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.forecast_records["T_C"][t] = T_C[0]
                self.forecast_records["T_C_lb"][t] = T_C_lb[0]
                self.forecast_records["T_C_ub"][t] = T_C_ub[0]
                self.forecast_records["T_i"][t] = T_i[0]
                self.forecast_records["T_i_lb"][t] = T_i_lb[0]
                self.forecast_records["T_i_ub"][t] = T_i_ub[0]
                self.forecast_records["Tavg_L"][t] = Tavg_L[0]
                self.forecast_records["Tavg_L_lb"][t] = Tavg_L_lb[0]
                self.forecast_records["Tavg_L_ub"][t] = Tavg_L_ub[0]
                self.forecast_records["T_L"][t] = T_L[0]
                self.forecast_records["T_L_lb"][t] = T_L_lb[0]
                self.forecast_records["T_L_ub"][t] = T_L_ub[0]
        else:
            T_C = self.rf_model1.predict(X_1)
            T_i = self.rf_model2.predict(X_2)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=self.Q_C[t], Q_i=self.Q_i[t])
            
            X_map[0, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L = self.rf_model_map.predict(X_map)
            
            if self.debug:
                self.forecast_records["Q_C"][t] = Q_C_[0]
                self.forecast_records["Q_i"][t] = Q_i_[0]
                self.forecast_records["cannonsville_storage_pct"][t] = X_1[0, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.forecast_records["T_C"][t] = T_C[0]
                self.forecast_records["T_i"][t] = T_i[0]
                self.forecast_records["Tavg_L"][t] = Tavg_L[0]
                self.forecast_records["T_L"][t] = T_L[0]
            T_C_lb = np.nan
            T_C_ub = np.nan
            T_i_lb = np.nan
            T_i_ub = np.nan
            Tavg_L_lb = np.nan
            Tavg_L_ub = np.nan
            T_L_lb = [np.nan]
            T_L_ub = [np.nan]
        
        self.forecast_T_L_arr = T_L
        self.forecast_T_L_lb_arr = T_L_lb
        self.forecast_T_L_ub_arr = T_L_ub
        
        return None

    def update_until(self, t=None, date=None, quantile=None):
        if t is not None:
            if t >= len(self.X_1) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X_1) + 1}.")
        if date is not None:
            if isinstance(date, str):
                date = date = pd.to_datetime(date)
            t = (date - self.current_date).days
            if t >= len(self.X_1) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")

        length = t - self.t
        if length == 0:
            return None
        
        t = self.t
        X_1 = self.X_1[t:t+length].reshape(length, -1)
        X_2 = self.X_2[t:t+length].reshape(length, -1)
        X_map = self.X_map[t:t+length].reshape(length, -1)
        Q_C_ = self.Q_C[t:t+length]
        Q_i_ = self.Q_i[t:t+length]
        
        if quantile is not None: 
            T_C, T_C_lb, T_C_ub = self.rf_model1.predict(X_1, quantile=quantile)
            T_i, T_i_lb, T_i_ub = self.rf_model2.predict(X_2, quantile=quantile)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=Q_C_, Q_i=Q_i_)
            Tavg_L_lb = self.blend_hot_cold_water(T_C=T_C_lb, T_i=T_i_lb, Q_C=Q_C_, Q_i=Q_i_)
            Tavg_L_ub = self.blend_hot_cold_water(T_C=T_C_ub, T_i=T_i_ub, Q_C=Q_C_, Q_i=Q_i_)
            
            X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L, _, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_lb
            _, T_L_lb, _ = self.rf_model_map.predict(X_map, quantile=quantile)
            X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_ub
            _, _, T_L_ub = self.rf_model_map.predict(X_map, quantile=quantile)
            
            if self.debug:
                self.records["Q_C"][t:t+length] = self.Q_C[t:t+length]
                self.records["Q_i"][t:t+length] = self.Q_i[t:t+length]
                self.records["cannonsville_storage_pct"][t:t+length] = X_1[:, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.records["T_C"][t:t+length] = T_C
                self.records["T_C_lb"][t:t+length] = T_C_lb
                self.records["T_C_ub"][t:t+length] = T_C_ub
                self.records["T_i"][t:t+length] = T_i
                self.records["T_i_lb"][t:t+length] = T_i_lb
                self.records["T_i_ub"][t:t+length] = T_i_ub
                self.records["Tavg_L"][t:t+length] = Tavg_L
                self.records["Tavg_L_lb"][t:t+length] = Tavg_L_lb
                self.records["Tavg_L_ub"][t:t+length] = Tavg_L_ub
                self.records["T_L"][t:t+length] = T_L
                self.records["T_L_lb"][t:t+length] = T_L_lb
                self.records["T_L_ub"][t:t+length] = T_L_ub
        else:
            T_C = self.rf_model1.predict(X_1)
            T_i = self.rf_model2.predict(X_2)
            Tavg_L = self.blend_hot_cold_water(T_C=T_C, T_i=T_i, Q_C=Q_C_, Q_i=Q_i_)
            
            X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L
            T_L = self.rf_model_map.predict(X_map)
            
            if self.debug:
                self.records["Q_C"][t:t+length] = self.Q_C[t:t+length]
                self.records["Q_i"][t:t+length] = self.Q_i[t:t+length]
                self.records["cannonsville_storage_pct"][t:t+length] = X_1[:, self.rf_model1.x_vars.index("bc_cannonsville_storage_pct")]
                self.records["T_C"][t:t+length] = T_C
                self.records["T_i"][t:t+length] = T_i
                self.records["Tavg_L"][t:t+length] = Tavg_L
                self.records["T_L"][t:t+length] = T_L
            T_C_lb = [np.nan]
            T_C_ub = [np.nan]
            T_i_lb = [np.nan]
            T_i_ub = [np.nan]
            Tavg_L_lb = [np.nan]
            Tavg_L_ub = [np.nan]
            T_L_lb = [np.nan]
            T_L_ub = [np.nan]
        
        self.T_C = T_C[-1]
        self.T_C_lb = T_C_lb[-1]
        self.T_C_ub = T_C_ub[-1]
        self.T_i = T_i[-1]
        self.T_i_lb = T_i_lb[-1]
        self.T_i_ub = T_i_ub[-1]
        self.Tavg_L = Tavg_L[-1]
        self.Tavg_L_lb = Tavg_L_lb[-1]
        self.Tavg_L_ub = Tavg_L_ub[-1]
        self.T_L = T_L[-1]
        self.T_L_lb = T_L_lb[-1]
        self.T_L_ub = T_L_ub[-1]
                
        self.t += length
        self.current_date += pd.Timedelta(days=length)
        return self.T_L, self.T_L_lb, self.T_L_ub
        
    def blend_hot_cold_water(self, T_C, T_i, Q_C, Q_i):
        """
        Blend the hot and cold water temperatures based on their respective flows.
        
        Parameters
        ==============
        T_C: float
            Temperature of cold water (Cannonsville releases).
        T_i: float
            Temperature of hot water (East branch).
        Q_C: float
            Flow of cold water.
        Q_i: float
            Flow of hot water.
        
        Returns
        ==============
        T_L: float
            Blended temperature.
        """
        Q_L = Q_C + Q_i
        return (T_C * Q_C + T_i * Q_i) / Q_L
    
    
  
class SaltfrontRandomForestUncertaintyModel:
    def __init__(self, rf_model_saltfront, debug=False):
        """
        WaterTempRandomForestUncertaintyModel class for temperature prediction using Random Forest models.
        
        Parameters
        ==============
        rf_model_saltfront: str
            Path to the Random Forest model for salt front prediction.
        debug: bool, default=False
            If True, enables debug mode which records intermediate values for debugging purposes.
        """

        # RF models
        self.rf_model_saltfront = joblib.load(rf_model_saltfront)
        
        # Input data
        self.X = np.nan
        self.Q_Trenton = np.nan
        self.Q_Schuylkill = np.nan
        self.Q_Trenton_7d_avg = np.nan
        self.Q_Schuylkill_7d_avg = np.nan
        self.Q_Trenton_7darr = np.nan
        self.Q_Schuylkill_7darr = np.nan
        
        # Dates
        self.start_date = None
        self.end_date = None
        self.length = None
        
        # Current predictions
        self.saltfront = np.nan
        self.saltfront_lb = np.nan
        self.saltfront_ub = np.nan
        
        # Forecast predictions
        self.forecast_saltfront_arr = [np.nan]
        self.forecast_saltfront_lb_arr = [np.nan]
        self.forecast_saltfront_ub_arr = [np.nan]
        
        # Time step
        self.t = 0
        self.current_date = None
        
        # Debug mode
        self.debug = debug
        if debug:
            self.records = {}
            self.forecast_records = {}
    
    def load_data(self, database, start_date='1979-01-01', end_date='2023-12-31'):
        """
        Load data from the database for the specified date range.
        Parameters
        ==============
        database: pd.DataFrame
            The database containing the meteological, temperature and flow data.
        start_date: str, default='1979-01-01'
            The start date for the data to be loaded.
        end_date: str, default='2023-12-31'
            The end date for the data to be loaded.
        """
        
        self.start_date = pd.to_datetime(start_date) 
        self.current_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) 
        
        db = database[start_date:end_date]
        self.X = db[self.rf_model_saltfront.x_vars].values
        
        self.Q_Trenton = db["Q_Trenton_bc"].values
        self.Q_Schuylkill = db["Q_Schuylkill_bc"].values
        self.Q_Trenton_7d_avg = db["Q_Trenton_bc_7d_avg"].values
        self.Q_Schuylkill_7d_avg = db["Q_Schuylkill_bc_7d_avg"].values
        
        self.Q_Trenton_7darr = deque(maxlen=7)
        self.Q_Trenton_7darr.extend([self.Q_Trenton_7d_avg[0]]*7)
        self.Q_Schuylkill_7darr = deque(maxlen=7)
        self.Q_Schuylkill_7darr.extend([self.Q_Schuylkill_7d_avg[0]]*7)
        
        length = db.shape[0]
        self.length = length
        if self.debug:
            self.records = {
                "Q_Trenton": [np.nan] * length,
                "Q_Schuylkill": [np.nan] * length,
                "Q_Trenton_7d_avg": [np.nan] * length,
                "Q_Schuylkill_7d_avg": [np.nan] * length,
                "saltfront": [np.nan] * length,
                "saltfront_lb": [np.nan] * length,
                "saltfront_ub": [np.nan] * length,
            }
            self.forecast_records = {
                "Q_Trenton": [np.nan] * length,
                "Q_Schuylkill": [np.nan] * length,
                "Q_Trenton_7d_avg": [np.nan] * length,
                "Q_Schuylkill_7d_avg": [np.nan] * length,
                "saltfront": [np.nan] * length,
                "saltfront_lb": [np.nan] * length,
                "saltfront_ub": [np.nan] * length,
            }
    
    def update(self, t, Q_Trenton=None, Q_Schuylkill=None, quantile=None):
        """
        Update the model with new data for a specific time step.
        
        Parameters
        ==============
        t: int
            The time step index to update the model.
        Q_C: float, optional
            The flow of cold water at time t.
        Q_i: float, optional
            The flow of hot water at time t.
        cannonsville_storage_pct: float, optional
            The percentage of Cannonsville storage at time t.
        quantile: float, optional
            If provided, compute the uncertainty bounds based on the quantile of tau_list.
            Should be between 0 and 1 (e.g., 0.95 for 95% confidence interval).
        """
        if Q_Trenton is not None:
            self.Q_Trenton[t] = Q_Trenton
            try:
                self.X[t, self.rf_model1.x_vars.index("Q_Trenton_bc")] = Q_Trenton
            except ValueError:
                print("Warning: 'Q_Trenton_bc' not found in rf_model_saltfront.x_vars. Skipping update.")
            
        if Q_Schuylkill is not None:
            self.Q_Schuylkill[t] = Q_Schuylkill
            try:
                self.X[t, self.rf_model2.x_vars.index("Q_Schuylkill_bc")] = Q_Schuylkill
            except ValueError:
                print("Warning: 'Q_Schuylkill_bc' not found in rf_model_saltfront.x_vars. Skipping update.")
                
        self.Q_Trenton_7darr.append(self.Q_Trenton[t])
        self.Q_Schuylkill_7darr.append(self.Q_Schuylkill[t])        
        self.Q_Trenton_7d_avg[t] = np.mean(self.Q_Trenton_7darr)
        self.Q_Schuylkill_7d_avg[t] = np.mean(self.Q_Schuylkill_7darr)
        try:
            self.X[t, self.rf_model_saltfront.x_vars.index("Q_Trenton_bc_7d_avg")] = self.Q_Trenton_7d_avg[t]
        except ValueError:
            print("Warning: 'Q_Trenton_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update.")
        try:
            self.X[t, self.rf_model_saltfront.x_vars.index("Q_Schuylkill_bc_7d_avg")] = self.Q_Schuylkill_7d_avg[t]
        except ValueError:
            print("Warning: 'Q_Schuylkill_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update.") 
       
        X = self.X[t].reshape(1, -1)
        
        if quantile is not None: 
            saltfront, saltfront_lb, saltfront_ub = self.rf_model_saltfront.predict(X, quantile=quantile)
            
            if self.debug:
                self.records["Q_Trenton"][t] = self.Q_Trenton[t]
                self.records["Q_Schuylkill"][t] = self.Q_Schuylkill[t]
                self.records["Q_Trenton_7d_avg"][t] = self.Q_Trenton_7d_avg[t]
                self.records["Q_Schuylkill_7d_avg"][t] = self.Q_Schuylkill_7d_avg[t]
                self.records["saltfront"][t] = saltfront[0]
                self.records["saltfront_lb"][t] = saltfront_lb[0]
                self.records["saltfront_ub"][t] = saltfront_ub[0]
          
        else:
            saltfront = self.rf_model_saltfront.predict(X)
        
            if self.debug:
                self.records["Q_Trenton"][t] = self.Q_Trenton[t]
                self.records["Q_Schuylkill"][t] = self.Q_Schuylkill[t]
                self.records["Q_Trenton_7d_avg"][t] = self.Q_Trenton_7d_avg[t]
                self.records["Q_Schuylkill_7d_avg"][t] = self.Q_Schuylkill_7d_avg[t]
                self.records["saltfront"][t] = saltfront[0]
            saltfront_lb = [np.nan]
            saltfront_ub = [np.nan]
        
        self.saltfront = saltfront[0]
        self.saltfront_lb = saltfront_lb[0]
        self.saltfront_ub = saltfront_ub[0]
        
        self.t += 1
        self.current_date += pd.Timedelta(days=1)
        return self.saltfront, self.saltfront_lb, self.saltfront_ub
    
    def forecast(self, t, Q_Trenton=None, Q_Schuylkill=None, quantile=None, lead_time=0):
        """
        Forecast the temperature for a given time step with a specified lead time.
        Parameters
        ==============
        t: int
            The time step index to forecast from.
        Q_C: float, optional
            The flow of cold water (Cannonsville release) at time t.
        Q_i: float, optional
            The flow of hot water (East branch) at time t.
        cannonsville_storage_pct: float, optional
            The percentage of Cannonsville storage at time t.
        quantile: float, optional
            If provided, compute the uncertainty bounds based on the quantile of tau_list.
            Should be between 0 and 1 (e.g., 0.95 for 95% confidence interval).
        lead_time: int, default=0
            The number of time steps to forecast ahead. If 0, only the current time step is forecasted.
        """
        X = self.X[t:t+lead_time+1].reshape(lead_time+1, -1)
        Q_Trenton_ = self.Q_Trenton[t:t+lead_time+1]
        Q_Schuylkill_ = self.Q_Schuylkill[t:t+lead_time+1]
        Q_Trenton_7d_avg_ = self.Q_Trenton_7d_avg[t:t+lead_time+1]
        Q_Schuylkill_7d_avg_ = self.Q_Schuylkill_7d_avg[t:t+lead_time+1]
        
        if Q_Trenton is not None:
            Q_Trenton_[0] = Q_Trenton
            try:
                X[0, self.rf_model_saltfront.x_vars.index("Q_Trenton_bc")] = Q_Trenton
            except ValueError:
                print("Warning: 'Q_Trenton_bc' not found in rf_model_saltfront.x_vars. Skipping update.")
        if Q_Schuylkill is not None:
            Q_Schuylkill_[0] = Q_Schuylkill
            try:
                X[0, self.rf_model_saltfront.x_vars.index("Q_Schuylkill_bc")] = Q_Schuylkill
            except ValueError:
                print("Warning: 'Q_Schuylkill_bc' not found in rf_model_saltfront.x_vars. Skipping update.")
        
        Q_Trenton_7darr = self.Q_Trenton_7darr.copy()
        Q_Schuylkill_7darr = self.Q_Schuylkill_7darr.copy()
        Q_Trenton_7darr.append(Q_Trenton_[0])
        Q_Schuylkill_7darr.append(Q_Schuylkill_[0])
        Q_Trenton_7d_avg_[0] = np.mean(Q_Trenton_7darr)
        Q_Schuylkill_7d_avg_[0] = np.mean(Q_Schuylkill_7darr)
        try:
            X[0, self.rf_model_saltfront.x_vars.index("Q_Trenton_bc_7d_avg")] = Q_Trenton_7d_avg_[0]
        except ValueError:
            print("Warning: 'Q_Trenton_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update.")
        try:
            X[0, self.rf_model_saltfront.x_vars.index("Q_Schuylkill_bc_7d_avg")] = Q_Schuylkill_7d_avg_[0]
        except ValueError:
            print("Warning: 'Q_Schuylkill_bc_7d_avg' not found in rf_model_saltfront.x_vars. Skipping update.")
                
        if quantile is not None:
            saltfront, saltfront_lb, saltfront_ub = self.rf_model_saltfront.predict(X, quantile=quantile)
            
            if self.debug:
                self.forecast_records["Q_Trenton"][t] = Q_Trenton_[0]
                self.forecast_records["Q_Schuylkill"][t] = Q_Schuylkill_[0]
                self.forecast_records["Q_Trenton_7d_avg"][t] = Q_Trenton_7d_avg_[0]
                self.forecast_records["Q_Schuylkill_7d_avg"][t] = Q_Schuylkill_7d_avg_[0]
                self.forecast_records["saltfront"][t] = saltfront[0]
                self.forecast_records["saltfront_lb"][t] = saltfront_lb[0]
                self.forecast_records["saltfront_ub"][t] = saltfront_ub[0]
        else:
            saltfront = self.rf_model_saltfront.predict(X)
            
            if self.debug:
                self.forecast_records["Q_Trenton"][t] = Q_Trenton_[0]
                self.forecast_records["Q_Schuylkill"][t] = Q_Schuylkill_[0]
                self.forecast_records["Q_Trenton_7d_avg"][t] = Q_Trenton_7d_avg_[0]
                self.forecast_records["Q_Schuylkill_7d_avg"][t] = Q_Schuylkill_7d_avg_[0]
                self.forecast_records["saltfront"][t] = saltfront[0]
            saltfront_lb = [np.nan]
            saltfront_ub = [np.nan]
            
        self.forecast_saltfront_arr = saltfront
        self.forecast_saltfront_lb_arr = saltfront_lb
        self.forecast_saltfront_ub_arr = saltfront_ub
                
        return None

    def update_until(self, t=None, date=None, quantile=None):
        if t is not None:
            if t >= len(self.X) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X) + 1}.")
        if date is not None:
            if isinstance(date, str):
                date = date = pd.to_datetime(date)
            t = (date - self.current_date).days
            if t >= len(self.X) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")

        length = t - self.t
        if length == 0:
            return None
        
        t = self.t
        X = self.X[t:t+length].reshape(length, -1)
        
        if quantile is not None: 
            saltfront, saltfront_lb, saltfront_ub = self.rf_model_saltfront.predict(X, quantile=quantile)
            
            if self.debug:
                self.records["Q_Trenton"][t:t+length] = self.Q_Trenton[t:t+length]
                self.records["Q_Schuylkill"][t:t+length] = self.Q_Schuylkill[t:t+length]
                self.records["Q_Trenton_7d_avg"][t:t+length] = self.Q_Trenton_7d_avg[t:t+length]
                self.records["Q_Schuylkill_7d_avg"][t:t+length] = self.Q_Schuylkill_7d_avg[t:t+length]
                self.records["saltfront"][t:t+length] = saltfront
                self.records["saltfront_lb"][t:t+length] = saltfront_lb
                self.records["saltfront_ub"][t:t+length] = saltfront_ub
        else:
            saltfront = self.rf_model_saltfront.predict(X)
            
            if self.debug:
                self.records["Q_Trenton"][t:t+length] = self.Q_Trenton[t:t+length]
                self.records["Q_Schuylkill"][t:t+length] = self.Q_Schuylkill[t:t+length]
                self.records["Q_Trenton_7d_avg"][t:t+length] = self.Q_Trenton_7d_avg[t:t+length]
                self.records["Q_Schuylkill_7d_avg"][t:t+length] = self.Q_Schuylkill_7d_avg[t:t+length]
                self.records["saltfront"][t:t+length] = saltfront
            saltfront_lb = [np.nan]
            saltfront_ub = [np.nan]
        
        self.saltfront = saltfront[-1]
        self.saltfront_lb = saltfront_lb[-1]
        self.saltfront_ub = saltfront_ub[-1]
                
        self.t += length
        self.current_date += pd.Timedelta(days=length)
        return self.saltfront, self.saltfront_lb, self.saltfront_ub