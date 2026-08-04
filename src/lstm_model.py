import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from tqdm import tqdm
import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

global pn
pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()

from src.torch_bmi import bmi_lstm

class WaterTempLSTMModel():
    def __init__(self,
                 model1, model2, Tavg2Tmax_coefs=None,
                 start_date='1979-01-01', end_date='2023-12-31',
                 Q_C_lstm_var_name="QbcTavg_Q_C",
                 Q_i_lstm_var_name="QbcTavg_Q_i",
                 cannonsville_storage_pct_lstm_var_name="bc_cannonsville_storage_pct",
                 thermal_mitigation_bank_size=1620,  # MGD-days ~= 2500 cfs-days
                 east_tributary_temperature_perturbation=0,
                 debug=False,
                 disable_tqdm=True
                 ):
        """
        Initialize the Water Temperature LSTM Model.

        Parameters
        ----------
        model1 : str or Path
            Path to the first LSTM model configuration file.
        model2 : str or Path
            Path to the second LSTM model configuration file.
        Tavg2Tmax_coefs : str or dict, optional
            .json path or a dictionary containing the linear mapping coefficients Tmax = a * Tavg + b and prediction uncertainty.
            Tavg2Tmax_coefs = {"a": a, "b": b, "mse": mse, "XTX_inv": XTX_inv} where:
            - a : float
                The slope of the linear mapping from Tavg to Tmax.
            - b : float
                The intercept of the linear mapping from Tavg to Tmax.
            - mse : float (optional)
                The mean squared error of the linear mapping.
            - XTX_inv : np.ndarray (optional)
                The inverse of the XTX matrix used in the linear regression.
            If None, the model will not map Tavg to Tmax.
        start_date : str or datetime, optional
            The start date for the model. Default is '1979-01-01'.
        end_date : str or datetime, optional
            The end date for the model. Default is '2023-12-31'.
        Q_C_lstm_var_name : str, optional
            The variable name for the Cannonsville reservoir release temperature (T_C). Default is "QbcTavg_Q_C".
        Q_i_lstm_var_name : str, optional
            The variable name for the East Branch reservoir release temperature (T_i). Default is "QbcTavg_Q_i".
        cannonsville_storage_pct_lstm_var_name : str, optional
            The variable name for the Cannonsville reservoir storage percentage. Default is "bc_cannonsville_storage_pct".
        thermal_mitigation_bank_size : float, optional
            The size of the thermal mitigation bank in mgd. Default is 1620.
        debug : bool, optional
            If True, enables debug mode for additional logging. Default is False.
        disable_tqdm : bool, optional
            If True, disables the tqdm progress bar and turn off the verbosity in lstm. Default is True.
        """

        ##### LSTM models
        mc_dropout = False
        # The main difference between the batch run and looping over seems to stem from
        # MC dropout: a single dropout mask is used during the batch run, whereas a new
        # mask is drawn at each step when looping through update(). When I manually
        # disable MC dropout while using the trained model, the output results remain
        # identical between the two approaches.

        # Predict the Cannonsville reservoir release temperature (T_C)
        lstm1 = bmi_lstm()
        lstm1.initialize(config_file=model1, train=False, root_dir=pn.get(), disable_tqdm=disable_tqdm)
        lstm1.mc_dropout = mc_dropout
        self.lstm1 = lstm1


        # Predict the water temperature for east branch (T_i)
        lstm2 = bmi_lstm()
        lstm2.initialize(config_file=model2, train=False, root_dir=pn.get(), disable_tqdm=disable_tqdm,
                         east_tributary_temperature_perturbation=east_tributary_temperature_perturbation)
        lstm2.mc_dropout = mc_dropout
        self.lstm2 = lstm2

        # Map Tavg to Tmax (T_L) at Lordville
        if Tavg2Tmax_coefs is None:
            self.Tavg2Tmax_coefs = None
            print("Tavg2Tmax_coefs is not given.")
        elif isinstance(Tavg2Tmax_coefs, dict):
            self.Tavg2Tmax_coefs = Tavg2Tmax_coefs
        else:
            import json
            with open(Tavg2Tmax_coefs, "r") as file:
                self.Tavg2Tmax_coefs = json.load(file)


        ##### Dates and lengths
        # Identify the start date of the LSTM models
        dt1 = lstm1.get_current_date()
        dt2 = lstm2.get_current_date()
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = np.datetime64(start_date)
        else:
            dt = max(dt1, dt2)
        start_date = min(max(dt1, dt2, dt), dt)

        # Advance the LSTM models to the beginning of the start date
        def update_until_(lstm, start_date):
            idx = np.where(lstm.dates_all == start_date)[1].item()
            if idx == 0: # Already at the right time step.
                return None
            length = idx - int(lstm.get_current_time())
            unscaled_data_arr = lstm.get_unscaled_values(lead_time=length-1)

            for var in lstm.x_vars:
                lstm.set_value(var, unscaled_data_arr[var])
            lstm.update()
            return None

        update_until_(lstm=self.lstm1, start_date=start_date)
        update_until_(lstm=self.lstm2, start_date=start_date)
        if disable_tqdm is not True:
            print(f"Updated LSTM1 to {lstm1.get_current_date()} at time step {int(lstm1.get_current_time())}.")
            print(f"Updated LSTM2 to {lstm2.get_current_date()} at time step {int(lstm2.get_current_time())}.")

        # Dates
        self.start_date = start_date
        self.end_date = np.datetime64(end_date)
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.current_date = self.start_date
        self.length = int((self.end_date - self.start_date) / np.timedelta64(1, 'D')) + 1
        self.t = 0

        # Input data
        self.X_1 = np.nan
        self.X_2 = np.nan
        self.X_map = np.nan
        self.Q_C = np.nan
        self.Q_i = np.nan

        # Current predictions
        self.T_C_mu = np.nan
        self.T_C_sd = np.nan
        self.T_i_mu = np.nan
        self.T_i_sd = np.nan
        self.Tavg_L_mu = np.nan
        self.Tavg_L_sd = np.nan
        self.T_L_mu = np.nan
        self.T_L_sd = np.nan

        # Forecast predictions
        self.forecast_T_L_mu_arr = np.nan
        self.forecast_T_L_sd_arr = np.nan
        self.forecast_Tavg_L_mu_arr = np.nan
        self.forecast_Tavg_L_sd_arr = np.nan
        self.forecast_T_C_mu_arr = np.nan
        self.forecast_T_C_sd_arr = np.nan
        self.forecast_T_i_mu_arr = np.nan
        self.forecast_T_i_sd_arr = np.nan

        # Thermal control variables
        self.thermal_mitigation_bank_size = thermal_mitigation_bank_size
        self.remained_bank_amount = thermal_mitigation_bank_size  # mgd

        # Coupling variables
        self.Q_C_lstm_var_name = Q_C_lstm_var_name
        self.Q_i_lstm_var_name = Q_i_lstm_var_name
        self.cannonsville_storage_pct_lstm_var_name = cannonsville_storage_pct_lstm_var_name

        # Debug mode
        self.debug = debug
        length = self.length
        if debug:
            self.records = {
                "Q_C": [np.nan] * length,
                "Q_i": [np.nan] * length,
                "cannonsville_storage_pct": [np.nan] * length,
                "T_C_mu": [np.nan] * length,
                "T_C_sd": [np.nan] * length,
                "T_i_mu": [np.nan] * length,
                "T_i_sd": [np.nan] * length,
                "Tavg_L_mu": [np.nan] * length,
                "Tavg_L_sd": [np.nan] * length,
                "T_L_mu": [np.nan] * length,
                "T_L_sd": [np.nan] * length,
                "thermal_releases": [np.nan] * length,
                "remained_bank_amounts": [np.nan] * length,
            }
            self.forecast_records = {
                "Q_C": [np.nan] * length,
                "Q_i": [np.nan] * length,
                "cannonsville_storage_pct": [np.nan] * length,
                "T_C_mu": [np.nan] * length,
                "T_C_sd": [np.nan] * length,
                "T_i_mu": [np.nan] * length,
                "T_i_sd": [np.nan] * length,
                "Tavg_L_mu": [np.nan] * length,
                "Tavg_L_sd": [np.nan] * length,
                "T_L_mu": [np.nan] * length,
                "T_L_sd": [np.nan] * length
            }

    def load_data(self, database=None):
        """
        Load the data from the database & lstm internal data for the specified date range.
        """

        # Load db (Future more flexible to load from different sources)
        db = database[self.start_date:self.end_date]
        self.Q_C = db[self.Q_C_lstm_var_name].values
        self.Q_i = db[self.Q_i_lstm_var_name].values


        # For control purposes
        self.doc = db["doc"].values if "doc" in db.columns else None
        self.cannonsville_storage_pct = db[self.cannonsville_storage_pct_lstm_var_name].values

        # Now we directly use the LSTM models to get the data
        length = self.length
        lstm1 = self.lstm1
        self.X_1 = lstm1.get_unscaled_values(lead_time=length-1).reset_index(drop=True) # A DataFrame
        self.x_vars_1 = list(lstm1.x_vars)  # Store x_vars for later use
        self.X_1 = self.X_1[self.x_vars_1].values  # Ensure same order as LSTM x_vars are included

        lstm2 = self.lstm2
        self.X_2 = lstm2.get_unscaled_values(lead_time=length-1).reset_index(drop=True) # A DataFrame
        self.x_vars_2 = list(lstm2.x_vars)  # Store x_vars for later use
        self.X_2 = self.X_2[self.x_vars_2].values  # Ensure same order as LSTM x_vars are included

    def update_until(self, t=None, date=None):
        """
        Update the LSTM models until the beginning of the specified time step or date.

        Parameters
        ----------
        t : int, optional
            The time step to update the models to. If None, updates to the next time step
        date : str or datetime, optional
            The date to update the models to. If None, updates to the next time step.
        Returns
        -------
        tuple
            The predicted T_L_mu and T_L_sd from the lstm model current date before
            update the t-1 of the specified time step (t).
        """
        if t is not None:
            if t >= self.length or t <= self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X_1) + 1}.")
        if date is not None:
            #if isinstance(date, str):
            date = np.datetime64(date)
            t = int((date - self.start_date) / np.timedelta64(1, 'D'))
            if t >= len(self.X_1) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")
            elif t == self.t:
                print(f"{t} is equal to model_ml current time step {self.t}. No update is implemented." )
                return self.T_L_mu, self.T_L_sd

        length = t - self.t # Minimum 1 step
        t = self.t

        # Pandas DF takes t:t but array takes t:t+1
        X_1 = self.X_1[t:t+length, :]
        X_2 = self.X_2[t:t+length, :]

        lstm = self.lstm1
        for i, var in enumerate(self.x_vars_1):
            lstm.set_value(var, X_1[:, i])
        T_C_mu, T_C_sd = lstm.update()
        if length == 1:
            T_C_mu, T_C_sd = np.array([T_C_mu]), np.array([T_C_sd])

        lstm = self.lstm2
        for i, var in enumerate(self.x_vars_2):
            lstm.set_value(var, X_2[:, i])
        T_i_mu, T_i_sd = lstm.update()
        if length == 1:
            T_i_mu, T_i_sd = np.array([T_i_mu]), np.array([T_i_sd])

        Q_C_ = self.Q_C[t:t+length]
        Q_i_ = self.Q_i[t:t+length]
        Tavg_L_mu, Tavg_L_sd = self.blend_hot_cold_water(
            T_C_mu=T_C_mu, T_i_mu=T_i_mu,
            T_C_sd=T_C_sd, T_i_sd=T_i_sd,
            Q_C=Q_C_, Q_i=Q_i_
        )

        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        Tavg2Tmax_coefs = self.Tavg2Tmax_coefs
        if Tavg2Tmax_coefs is None:
            T_L_mu = Tavg_L_mu
            T_L_sd = Tavg_L_sd
        else:
            T_L_mu, T_L_sd = WaterTempLSTMModel.tavg2tmax(Tavg_L_mu, Tavg2Tmax_coefs)
            T_L_sd += Tavg_L_sd

        if self.debug:
            records = self.records
            records["Q_C"][t:t+length] = Q_C_
            records["Q_i"][t:t+length] = Q_i_
            records["cannonsville_storage_pct"][t:t+length] = self.cannonsville_storage_pct[t:t+length]
            records["T_C_mu"][t:t+length] = T_C_mu
            records["T_C_sd"][t:t+length] = T_C_sd
            records["T_i_mu"][t:t+length] = T_i_mu
            records["T_i_sd"][t:t+length] = T_i_sd
            records["Tavg_L_mu"][t:t+length] = Tavg_L_mu
            records["Tavg_L_sd"][t:t+length] = Tavg_L_sd
            records["T_L_mu"][t:t+length] = T_L_mu
            records["T_L_sd"][t:t+length] = T_L_sd
        self.T_C_mu = float(T_C_mu[-1])
        self.T_C_sd = float(T_C_sd[-1])
        self.T_i_mu = float(T_i_mu[-1])
        self.T_i_sd = float(T_i_sd[-1])
        self.Tavg_L_mu = float(Tavg_L_mu[-1])
        self.Tavg_L_sd = float(Tavg_L_sd[-1])
        self.T_L_mu = float(T_L_mu[-1])
        self.T_L_sd = float(T_L_sd[-1])

        self.t += length
        self.current_date += np.timedelta64(length, 'D')
        return self.T_L_mu, self.T_L_sd

    def update(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, asycronized_update=False):
        """
        Update the LSTM models to the specified time step.

        Parameters
        ----------
        t : int
            The time step to update the models to.
        Q_C : float, optional
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float, optional
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float, optional
            The percentage of the Cannonsville reservoir storage.

        Returns
        -------
        tuple
            The predicted T_L_mu and T_L_sd at the specified time step (t).
        """
        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name

        if Q_C is not None:
            self.Q_C[t] = Q_C
            try:
                self.X_1[t, self.x_vars_1.index(Q_C_lstm_var_name)] = Q_C
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            try:
                self.X_2[t, self.x_vars_2.index(Q_C_lstm_var_name)] = Q_C
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")

        if Q_i is not None:
            self.Q_i[t] = Q_i
            try:
                self.X_2[t, self.x_vars_2.index(Q_i_lstm_var_name)] = Q_i
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_i_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")

        if cannonsville_storage_pct is not None:
            try:
                self.X_1[t, self.x_vars_1.index(cannonsville_storage_pct_lstm_var_name)] = cannonsville_storage_pct
            except ValueError:
                if self.debug:
                    print(f"Warning: '{cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")

        if asycronized_update:
            return None
        else:
            return self.update_until(t=t+1, date=None)

    def forecast(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0):
        """
        Forecast the water temperature at Lordville for the specified lead time.

        Parameters
        ----------
        t : int
            The time step to forecast from.
        Q_C : float, optional
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float, optional
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float, optional
            The percentage of the Cannonsville reservoir storage.
        lead_time : int, optional
            The number of time steps to forecast ahead. Default is 0, which means nowcast
            and returns the current prediction.
        Returns
        -------
        None
            The forecasted water temperature at Lordville is stored in the class attributes.
        """
        # Pandas DF takes t:t but array takes t:t+1
        X_1 = self.X_1[t:t+lead_time+1, :].copy()
        X_2 = self.X_2[t:t+lead_time+1, :].copy()
        Q_C_ = self.Q_C[t:t+lead_time+1].copy()
        Q_i_ = self.Q_i[t:t+lead_time+1].copy()

        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name

        if Q_C is not None:
            Q_C_[0] = Q_C
            try:
                X_1[0, self.x_vars_1.index(Q_C_lstm_var_name)] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm1.x_vars or lstm2.x_vars. Skipping update.")
            try:
                X_2[0, self.x_vars_2.index(Q_C_lstm_var_name)] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
        if Q_i is not None:
            Q_i_[0] = Q_i
            try:
                X_2[0, self.x_vars_2.index(Q_i_lstm_var_name)] = Q_i
            except ValueError:
                print(f"Warning: '{Q_i_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
        if cannonsville_storage_pct is not None:
            try:
                X_1[0, self.x_vars_1.index(cannonsville_storage_pct_lstm_var_name)] = cannonsville_storage_pct
            except ValueError:
                print(f"Warning: '{cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")

        lstm = self.lstm1
        for i, var in enumerate(self.x_vars_1):
            lstm.set_value(var, X_1[:, i])
        forecast_T_C_mu, forecast_T_C_sd = lstm.forecast()

        lstm = self.lstm2
        for i, var in enumerate(self.x_vars_2):
            lstm.set_value(var, X_2[:, i])
        forecast_T_i_mu, forecast_T_i_sd = lstm.forecast()

        #forecast_T_C_mu = df_T_C["mu"].values
        #forecast_T_C_sd = df_T_C["sd"].values
        #forecast_T_i_mu = df_T_i["mu"].values
        #forecast_T_i_sd = df_T_i["sd"].values

        forecast_Tavg_L_mu, forecast_Tavg_L_sd = self.blend_hot_cold_water(
            T_C_mu=forecast_T_C_mu, T_i_mu=forecast_T_i_mu,
            T_C_sd=forecast_T_C_sd, T_i_sd=forecast_T_i_sd,
            Q_C=Q_C_, Q_i=Q_i_
        )

        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        Tavg2Tmax_coefs = self.Tavg2Tmax_coefs
        if Tavg2Tmax_coefs is None:
            forecast_T_L_mu = forecast_Tavg_L_mu
            forecast_T_L_sd = forecast_Tavg_L_sd
        else:
            forecast_T_L_mu, forecast_T_L_sd = WaterTempLSTMModel.tavg2tmax(forecast_Tavg_L_mu, Tavg2Tmax_coefs)
            forecast_T_L_sd += forecast_Tavg_L_sd

        if self.debug:
            forecast_records = self.forecast_records
            forecast_records["Q_C"][t] = Q_C_[-1]
            forecast_records["Q_i"][t] = Q_i_[-1]
            forecast_records["T_C_mu"][t] = forecast_T_C_mu[-1]
            forecast_records["T_C_sd"][t] = forecast_T_C_sd[-1]
            forecast_records["T_i_mu"][t] = forecast_T_i_mu[-1]
            forecast_records["T_i_sd"][t] = forecast_T_i_sd[-1]
            forecast_records["Tavg_L_mu"][t] = forecast_Tavg_L_mu[-1]
            forecast_records["Tavg_L_sd"][t] = forecast_Tavg_L_sd[-1]
            forecast_records["T_L_mu"][t] = forecast_T_L_mu[-1]
            forecast_records["T_L_sd"][t] = forecast_T_L_sd[-1]

        self.forecast_T_L_mu_arr = forecast_T_L_mu
        self.forecast_T_L_sd_arr = forecast_T_L_sd
        self.forecast_Tavg_L_mu_arr = forecast_Tavg_L_mu
        self.forecast_Tavg_L_sd_arr = forecast_Tavg_L_sd
        self.forecast_T_C_mu_arr = forecast_T_C_mu
        self.forecast_T_C_sd_arr = forecast_T_C_sd
        self.forecast_T_i_mu_arr = forecast_T_i_mu
        self.forecast_T_i_sd_arr = forecast_T_i_sd


        return None

    @staticmethod
    def blend_hot_cold_water(T_C_mu, T_i_mu, T_C_sd, T_i_sd, Q_C, Q_i):
        """
        Blend the hot and cold water temperatures to get the average temperature at Lordville.

        Parameters
        ----------
        T_C_mu : float or np.ndarray
            The mean temperature of the Cannonsville reservoir release.
        T_i_mu : float or np.ndarray
            The mean temperature of the East Branch flow and natural inflow to Lordville.
        T_C_sd : float or np.ndarray
            The standard deviation of the Cannonsville reservoir release temperature.
        T_i_sd : float or np.ndarray
            The standard deviation of the East Branch flow and natural inflow to Lordville temperature.
        Q_C : float or np.ndarray
            The flow rate of the Cannonsville reservoir release.
        Q_i : float or np.ndarray
            The flow rate of the East Branch flow and natural inflow to Lordville.
        Returns
        tuple
            The average temperature at Lordville (Tavg_L_mu) and its standard deviation (Tavg_L_sd).
        """
        Tavg_L_mu = (T_C_mu*Q_C + T_i_mu*Q_i)/(Q_C + Q_i)
        # Assuming T_i and T_C are independent
        Tavg_L_sd = np.sqrt((T_C_sd**2 * Q_C**2 + T_i_sd**2 * Q_i**2) / (Q_C + Q_i)**2)
        return Tavg_L_mu, Tavg_L_sd

    @staticmethod
    def fit_linear_regression(x, y):
        """
        Simple linear regression

        mse = np.sum((y - y_fit)**2) / (n - 2)

        XTX_inv = np.linalg.inv(X.T @ X)
        X_new = np.vstack([x_new, np.ones_like(x_new)]).T
        se_pred = np.sqrt(mse * (1 + np.diag(X_new @ XTX_inv @ X_new.T)))

        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable

        Returns:
        --------
        tuple with slope, intercept, and XTX_inv for later prediction uncertainty
        """
        x = np.array(x)
        y = np.array(y)

        # Design matrix [x, 1] for y = ax + b
        X = np.vstack([x, np.ones_like(x)]).T

        # Least squares solution
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fit = a * x + b

        # For later prediction uncertainty calculation
        n = len(y)
        mse = np.sum((y - y_fit)**2) / (n - 2)
        XTX_inv = np.linalg.inv(X.T @ X)

        # Calculate RMSE
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse:.4f}")

        coefs = {"a": a, "b": b, "mse": mse, "XTX_inv": XTX_inv}
        return coefs

    @staticmethod
    def tavg2tmax(Tavg, Tavg2Tmax_coefs):
        """
        Convert average temperature (Tavg) to maximum temperature (Tmax) using the fitted linear regression model.

        Parameters
        ----------
        Tavg2Tmax_coefs : tuple
            A tuple containing the coefficients (a, b) of the linear regression model,
            and optionally the mean squared error (mse) and the inverse of XTX (XTX_inv).
            If mse and XTX_inv are not provided, they will be set to None.
        Tavg : array-like
            The average temperature values to convert.

        Returns
        -------
        tuple
            The maximum temperature values and their standard deviations.
        """
        Tavg = np.atleast_1d(Tavg).astype(float)

        a = Tavg2Tmax_coefs["a"]
        b = Tavg2Tmax_coefs["b"]
        mse = Tavg2Tmax_coefs.get("mse")
        XTX_inv = Tavg2Tmax_coefs.get("XTX_inv")

        # Predict
        Tmax = a * Tavg + b
        # Calculate prediction uncertainty if mse and XTX_inv are available
        if mse is not None and XTX_inv is not None:
            Tavg = np.vstack([Tavg, np.ones_like(Tavg)]).T
            se_pred = np.sqrt(mse * (1 + np.diag(Tavg @ XTX_inv @ Tavg.T)))
        else:
            se_pred = np.array([np.nan])

        return Tmax, se_pred

class SalinityLSTMModel():
    def __init__(self,
                 model_salinity,
                 start_date='1979-01-01', end_date='2023-12-31',
                 Q_Trenton_lstm_var_name="Q_Trenton_bc",
                 Q_Schuylkill_lstm_var_name="Q_Schuylkill_bc",
                 debug=False,
                 disable_tqdm=True
                 ):
        """
        Initialize the Water Temperature LSTM Model.

        Parameters
        ----------
        model_salinity : str or Path
            Path to the salinity LSTM model configuration file.
        start_date : str or datetime, optional
            The start date for the model. Default is '1979-01-01'.
        end_date : str or datetime, optional
            The end date for the model. Default is '2023-12-31'.
        Q_Trenton_lstm_var_name : str, optional
            The variable name for the Trenton flow (Q_Trenton_bc). Default is "Q_Trenton_bc".
        Q_Schuylkill_lstm_var_name : str, optional
            The variable name for the Schuylkill flow (Q_Schuylkill_bc). Default is "Q_Schuylkill_bc".
        debug : bool, optional
            If True, enables debug mode for additional logging. Default is False.
        disable_tqdm : bool, optional
            If True, disables the tqdm progress bar and turn off the verbosity in lstm. Default is True.
        """

        ##### LSTM models
        mc_dropout = False
        # The main difference between the batch run and looping over seems to stem from
        # MC dropout: a single dropout mask is used during the batch run, whereas a new
        # mask is drawn at each step when looping through update(). When I manually
        # disable MC dropout while using the trained model, the output results remain
        # identical between the two approaches.

        # Predict the salt front location
        lstm = bmi_lstm()
        lstm.initialize(config_file=model_salinity, train=False, root_dir=pn.get(), disable_tqdm=disable_tqdm)
        lstm.mc_dropout = mc_dropout
        self.lstm = lstm


        ##### Dates and lengths
        # Identify the start date of the LSTM models
        dt_lstm = lstm.get_current_date()
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = np.datetime64(start_date)
        else:
            dt = dt_lstm
        start_date = min(max(dt_lstm, dt), dt)

        # Advance the LSTM models to the beginning of the start date
        def update_until_(lstm, start_date):
            idx = np.where(lstm.dates_all == start_date)[1].item()
            if idx == 0: # Already at the right time step.
                return None
            length = idx - int(lstm.get_current_time())
            unscaled_data_arr = lstm.get_unscaled_values(lead_time=length-1)

            for var in lstm.x_vars:
                lstm.set_value(var, unscaled_data_arr[var])
            lstm.update()
            return None

        update_until_(lstm=self.lstm, start_date=start_date)
        if disable_tqdm is not True:
            print(f"Updated LSTM to {lstm.get_current_date()} at time step {int(lstm.get_current_time())}.")

        # Dates
        self.start_date = start_date
        self.end_date = np.datetime64(end_date)
        self.dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        self.current_date = self.start_date
        self.length = int((self.end_date - self.start_date) / np.timedelta64(1, 'D')) + 1
        self.t = 0

        # Input data
        self.X = np.nan

        # Current predictions
        self.sf_mu = np.nan
        self.sf_sd = np.nan

        # Forecast predictions
        self.forecast_sf_mu_arr = np.nan
        self.forecast_sf_sd_arr = np.nan

        # Coupling variables
        self.Q_Trenton_lstm_var_name = Q_Trenton_lstm_var_name
        self.Q_Schuylkill_lstm_var_name = Q_Schuylkill_lstm_var_name

        # Debug mode
        self.debug = debug
        length = self.length
        if debug:
            self.records = {
                "sf_mu": [np.nan] * length,
                "sf_sd": [np.nan] * length,
                "adj_ratio_Trenton": [np.nan] * length,
                "adj_ratio_Montague": [np.nan] * length,
                "drought_idx": [np.nan] * length,
            }
            self.forecast_records = {
                "sf_mu": [np.nan] * length,
                "sf_sd": [np.nan] * length
            }

    def load_data(self):
        """
        Load the data from the database & lstm internal data for the specified date range.
        """

        # Now we directly use the LSTM models to get the data
        length = self.length
        lstm = self.lstm
        self.X = lstm.get_unscaled_values(lead_time=length-1).reset_index(drop=True) # A DataFrame
        self.x_vars = list(lstm.x_vars)  # Store x_vars for later use
        self.X = self.X[self.x_vars].values  # Ensure same order as LSTM x_vars are included

    def update_until(self, t=None, date=None):
        """
        Update the LSTM models until the beginning of the specified time step or date.

        Parameters
        ----------
        t : int, optional
            The time step to update the models to. If None, updates to the next time step
        date : str or datetime, optional
            The date to update the models to. If None, updates to the next time step.
        Returns
        -------
        tuple
            The predicted T_L_mu and T_L_sd from the lstm model current date before
            update the t-1 of the specified time step (t).
        """
        if t is not None:
            if t >= self.length or t <= self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X) + 1}.")
        if date is not None:
            if isinstance(date, str):
                date = np.datetime64(date)
            t = int((date - self.start_date) / np.timedelta64(1, 'D'))
            if t >= len(self.X) + 1 or t <= self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")

        length = t - self.t # Minimum 1 step
        t = self.t

        # Pandas DF takes t:t but array takes t:t+1
        X = self.X[t:t+length, :]

        lstm = self.lstm
        for i, var in enumerate(self.x_vars):
            lstm.set_value(var, X[:, i])
        sf_mu, sf_sd = lstm.update()
        if length == 1:
            sf_mu, sf_sd = np.array([sf_mu]), np.array([sf_sd])

        if self.debug:
            records = self.records
            records["sf_mu"][t:t+length] = sf_mu
            records["sf_sd"][t:t+length] = sf_sd

        self.sf_mu = float(sf_mu[-1])
        self.sf_sd = float(sf_sd[-1])

        self.t += length
        self.current_date += np.timedelta64(length, 'D')
        return self.sf_mu, self.sf_sd

    def update(self, t, Q_Trenton=None, Q_Schuylkill=None, asycronized_update=False):
        """
        Update the LSTM models to the specified time step.

        Parameters
        ----------
        t : int
            The time step to update the models to.
        Q_Trenton : float
            The Trenton flow (Q_Trenton_bc).
        Q_Schuylkill : float
            The Schuylkill flow (Q_Schuylkill_bc).

        Returns
        -------
        tuple
            The predicted T_L_mu and T_L_sd at the specified time step (t).
        """
        Q_Trenton_lstm_var_name = self.Q_Trenton_lstm_var_name
        Q_Schuylkill_lstm_var_name = self.Q_Schuylkill_lstm_var_name

        if Q_Trenton is not None:
            try:
                self.X[t, self.x_vars.index(Q_Trenton_lstm_var_name)] = Q_Trenton
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_Trenton_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            try:
                if self.t == 0:
                    Q_Trenton_7d = self.X[t, self.x_vars.index(Q_Trenton_lstm_var_name+"_7d_avg")]
                else:
                    Q_Trenton_7d_t_1 = self.X[t-1, self.x_vars.index((Q_Trenton_lstm_var_name+"_7d_avg"))]
                    Q_Trenton_7d = (Q_Trenton_7d_t_1*6 + Q_Trenton) / 7
                self.X[t, self.x_vars.index((Q_Trenton_lstm_var_name+"_7d_avg"))] = Q_Trenton_7d
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_Trenton_lstm_var_name}_7d_avg' not found in lstm2.x_vars. Skipping update.")

        if Q_Schuylkill is not None:
            try:
                self.X[t, self.x_vars.index(Q_Schuylkill_lstm_var_name)] = Q_Schuylkill
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_Schuylkill_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            try:
                if self.t == 0:
                    Q_Schuylkill_7d = self.X[t, self.x_vars.index(Q_Schuylkill_lstm_var_name+"_7d_avg")]
                else:
                    Q_Schuylkill_7d_t_1 = self.X[t-1, self.x_vars.index((Q_Schuylkill_lstm_var_name+"_7d_avg"))]
                    Q_Schuylkill_7d = (Q_Schuylkill_7d_t_1*6 + Q_Schuylkill) / 7
                self.X[t, self.x_vars.index((Q_Schuylkill_lstm_var_name+"_7d_avg"))] = Q_Schuylkill_7d
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_Schuylkill_lstm_var_name}_7d_avg' not found in lstm2.x_vars. Skipping update.")

        if asycronized_update:
            return None
        else:
            return self.update_until(t=t+1, date=None)

    def forecast(self, t, Q_Trenton=None, Q_Schuylkill=None, lead_time=0):
        """
        Forecast the water temperature at Lordville for the specified lead time.

        Parameters
        ----------
        t : int
            The time step to forecast from.
        Q_Trenton : float, optional
            The Trenton flow (Q_Trenton_bc).
        Q_Schuylkill : float, optional
            The Schuylkill flow (Q_Schuylkill_bc).
        lead_time : int, optional
            The number of time steps to forecast ahead. Default is 0, which means nowcast
            and returns the current prediction.
        Returns
        -------
        None
            The forecasted water temperature at Lordville is stored in the class attributes.
        """
        # Pandas DF takes t:t but array takes t:t+1
        X = self.X[t:t+lead_time+1, :].copy()

        Q_Trenton_lstm_var_name = self.Q_Trenton_lstm_var_name
        Q_Schuylkill_lstm_var_name = self.Q_Schuylkill_lstm_var_name

        if Q_Trenton is not None:
            try:
                X[0, self.x_vars.index(Q_Trenton_lstm_var_name)] = Q_Trenton
            except ValueError:
                print(f"Warning: '{Q_Trenton_lstm_var_name}' not found in lstm.x_vars. Skipping update.")

            if self.t == 0:
                Q_Trenton_7d = X[0, self.x_vars.index(Q_Trenton_lstm_var_name+"_7d_avg")]
            else:
                Q_Trenton_7d_t_1 = self.X[t-1, self.x_vars.index((Q_Trenton_lstm_var_name+"_7d_avg"))]
                Q_Trenton_7d = (Q_Trenton_7d_t_1*6 + Q_Trenton) / 7

            try:
                X[0, self.x_vars.index(Q_Trenton_lstm_var_name+"_7d_avg")] = Q_Trenton_7d
            except ValueError:
                print(f"Warning: '{Q_Trenton_lstm_var_name}_7d_avg' not found in lstm.x_vars. Skipping update.")

        if Q_Schuylkill is not None:
            try:
                X[0, self.x_vars.index(Q_Schuylkill_lstm_var_name)] = Q_Schuylkill
            except ValueError:
                print(f"Warning: '{Q_Schuylkill_lstm_var_name}' not found in lstm.x_vars. Skipping update.")
            if self.t == 0:
                Q_Schuylkill_7d = X[0, self.x_vars.index(Q_Schuylkill_lstm_var_name+"_7d_avg")]
            else:
                Q_Schuylkill_7d = (self.Q_Schuylkill_7d[t-1]*6 + Q_Schuylkill) / 7
            try:
                X[0, self.x_vars.index(Q_Schuylkill_lstm_var_name+"_7d_avg")] = Q_Schuylkill_7d
            except ValueError:
                print(f"Warning: '{Q_Schuylkill_lstm_var_name}_7d_avg' not found in lstm.x_vars. Skipping update.")

        lstm = self.lstm
        for i, var in enumerate(self.x_vars):
            lstm.set_value(var, X[:, i])
        forecast_sf_mu, forecast_sf_sd = lstm.forecast()

        if self.debug:
            forecast_records = self.forecast_records
            forecast_records["sf_mu"][t] = forecast_sf_mu[-1]
            forecast_records["sf_sd"][t] = forecast_sf_sd[-1]

        self.forecast_sf_mu_arr = forecast_sf_mu
        self.forecast_sf_sd_arr = forecast_sf_sd
        return None