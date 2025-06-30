import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from tqdm import tqdm
import pathnavigator

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
global pn
pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()

from src.torch_bmi import bmi_lstm

class WaterTempLSTMModel():
    def __init__(self, 
                 model1, model2, model_map,
                 start_date='1979-01-01', end_date='2023-12-31', 
                 Q_C_lstm_var_name="QbcTavg_Q_C", 
                 Q_i_lstm_var_name="QbcTavg_Q_i", 
                 cannonsville_storage_pct_lstm_var_name="bc_cannonsville_storage_pct",
                 thermal_mitigation_bank_size=1620,  # mgd
                 debug=False
                 ):
        """
        A custom parameter class to predict daily maximum water temperature at Lordville using LSTM models.
        
        Parameters
        ----------
        model : pywr.core.Model
            The Pywr model object.
        start_date : str
            The start date for the model in "YYYY-MM-DD" format. If None, uses the model's start date.
        activate_thermal_control : bool
            If True, activates the thermal control mechanism.
        Q_C_lstm_var_name : str
            The variable name for the Cannonsville reservoir downstream flow in the LSTM model.
        Q_i_lstm_var_name : str
            The variable name for the East Branch flow and the natural flow to Lordville in the LSTM model.
        PywrDRB_ML_plugin_path : str
            The path to the PywrDRB_ML plugin directory containing the LSTM model configuration.
        disable_tqdm : bool
            If True, disables the tqdm progress bar during model initialization.
        debug : bool
            If True, enables debugging mode, which records intermediate values for inspection.
        **kwargs : dict
            Additional keyword arguments for the Parameter class.
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
        lstm1.initialize(config_file=model1, train=False, root_dir=pn.get())
        lstm1.mc_dropout = mc_dropout
        self.lstm1 = lstm1
        
        
        # Predict the water temperature for east branch (T_i)
        lstm2 = bmi_lstm()
        lstm2.initialize(config_file=model2, train=False, root_dir=pn.get())
        lstm2.mc_dropout = mc_dropout
        self.lstm2 = lstm2
        
        # Map Tavg to Tmax (T_L) at Lordville
        rf_model_map = joblib.load(model_map)
        self.rf_model_map = rf_model_map
        
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
        if debug:
            print(f"Updated LSTM1 to {lstm1.get_current_date()} at time step {int(lstm1.get_current_time())}.")
            print(f"Updated LSTM2 to {lstm2.get_current_date()} at time step {int(lstm2.get_current_time())}.")

        # Dates
        self.start_date = start_date
        self.end_date = np.datetime64(end_date)
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
        #self.cannonsville_storage_pct = db[self.cannonsville_storage_pct_lstm_var_name].values
        
        # Now we directly use the LSTM models to get the data
        length = self.length
        lstm1 = self.lstm1
        self.X_1 = lstm1.get_unscaled_values(lead_time=length-1).reset_index(drop=True) # A DataFrame
        
        lstm2 = self.lstm2
        self.X_2 = lstm2.get_unscaled_values(lead_time=length-1).reset_index(drop=True) # A DataFrame
        
        self.X_map = db[self.rf_model_map.x_vars].values

    
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
            if t >= self.length - 1 or t <= self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X_1) + 1}.")
        if date is not None:
            if isinstance(date, str):
                date = np.datetime64(date)
            t = int((date - self.start_date) / np.timedelta64(1, 'D'))
            if t >= len(self.X_1) + 1 or t <= self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")

        length = t - self.t # Minimum 1 step
        t = self.t
        
        # Pandas DF takes t:t but array takes t:t+1
        X_1 = self.X_1.loc[t:t+length-1, :].reset_index(drop=True)
        X_2 = self.X_2.loc[t:t+length-1, :].reset_index(drop=True)
        X_map = self.X_map[t:t+length, :]
        
        lstm = self.lstm1
        for var in lstm.x_vars:
            lstm.set_value(var, X_1[var])
        T_C_mu, T_C_sd = lstm.update()
        if length == 1:
            T_C_mu, T_C_sd = np.array([T_C_mu]), np.array([T_C_sd])
        
        lstm = self.lstm2
        for var in lstm.x_vars:
            lstm.set_value(var, X_2[var])
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
        rf_model_map = self.rf_model_map
        X_map[:, rf_model_map.x_vars.index('QbcTavg_T_L')] = Tavg_L_mu
        T_L_mu, _, _ = rf_model_map.predict(X_map, quantile=None)
        T_L_sd = Tavg_L_sd  # assuming a constant sd for T_L
        
        if self.debug:
            records = self.records
            records["Q_C"][t:t+length] = Q_C_
            records["Q_i"][t:t+length] = Q_i_
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
    
    def update(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None):
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
                self.X_1.loc[t, Q_C_lstm_var_name] = Q_C
            except ValueError:
                if self.debug:  
                    print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            try:
                self.X_2.loc[t, Q_C_lstm_var_name] = Q_C
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
            
        if Q_i is not None:
            self.Q_i[t] = Q_i
            try:
                self.X_2.loc[t, Q_i_lstm_var_name] = Q_i
            except ValueError:
                if self.debug:
                    print(f"Warning: '{Q_i_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
            
        if cannonsville_storage_pct is not None:
            try:
                self.X_1.loc[t, cannonsville_storage_pct_lstm_var_name] = cannonsville_storage_pct
            except ValueError:
                if self.debug:
                    print(f"Warning: '{cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
        
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
        X_1 = self.X_1.loc[t:t+lead_time, :].reset_index(drop=True).copy()
        X_2 = self.X_2.loc[t:t+lead_time, :].reset_index(drop=True).copy()
        X_map = self.X_map[t:t+lead_time+1, :].copy()
        Q_C_ = self.Q_C[t:t+lead_time+1].copy()
        Q_i_ = self.Q_i[t:t+lead_time+1].copy()
        
        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name
        
        if Q_C is not None:
            Q_C_[0] = Q_C
            try:
                X_1[Q_C_lstm_var_name][0] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm1.x_vars or lstm2.x_vars. Skipping update.")
            try:
                X_2[Q_C_lstm_var_name][0] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
        if Q_i is not None:
            Q_i_[0] = Q_i
            try:
                X_2[Q_i_lstm_var_name][0] = Q_i
            except ValueError:
                print(f"Warning: '{Q_i_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
        if cannonsville_storage_pct is not None:
            try:
                X_1[cannonsville_storage_pct_lstm_var_name][0] = cannonsville_storage_pct
            except ValueError:
                print(f"Warning: '{cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            
        lstm = self.lstm1
        for var in lstm.x_vars:
            lstm.set_value(var, self.X_1[var])
        df_T_C = lstm.forecast(lead_time=lead_time)
        
        lstm = self.lstm2
        for var in lstm.x_vars:
            lstm.set_value(var, self.X_2[var])
        df_T_i = lstm.forecast(lead_time=lead_time)
        
        forecast_T_C_mu = df_T_C["mu"].values
        forecast_T_C_sd = df_T_C["sd"].values
        forecast_T_i_mu = df_T_i["mu"].values
        forecast_T_i_sd = df_T_i["sd"].values
        
        forecast_Tavg_L_mu, forecast_Tavg_L_sd = self.blend_hot_cold_water(
            T_C_mu=forecast_T_C_mu, T_i_mu=forecast_T_i_mu, 
            T_C_sd=forecast_T_C_sd, T_i_sd=forecast_T_i_sd, 
            Q_C=Q_C_, Q_i=Q_i_
        )
        
        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = forecast_Tavg_L_mu
        forecast_T_L_mu, _, _ = self.rf_model_map.predict(X_map, quantile=None)
        forecast_T_L_sd = forecast_Tavg_L_sd  # assuming a constant sd for T_L  
        
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
        return None
        
    def blend_hot_cold_water(self, T_C_mu, T_i_mu, T_C_sd, T_i_sd, Q_C, Q_i):
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
    
    





