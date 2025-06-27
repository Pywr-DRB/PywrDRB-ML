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
                 disable_tqdm=True,
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
        # Predict the Cannonsville reservoir release temperature (T_C)
        lstm1 = bmi_lstm()
        lstm1.initialize(config_file=model1, train=False, root_dir=pn.get())
        self.lstm1 = lstm1
        
        # Predict the water temperature for east branch (T_i)
        lstm2 = bmi_lstm()
        lstm2.initialize(config_file=model2, train=False, root_dir=pn.get())
        self.lstm2 = lstm2
        
        # Map Tavg to Tmax (T_L) at Lordville
        rf_model_map = joblib.load(model_map)
        self.rf_model_map = rf_model_map
        
        ##### Dates and lengths        
        # Identify the start date of the LSTM models
        dt1 = pd.to_datetime(lstm1.get_current_date())
        dt2 = pd.to_datetime(lstm2.get_current_date())
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            dt = max(dt1, dt2)
            
        start_date = min(max(dt1, dt2, dt), dt)
        length1 = max((start_date - dt1).days, 0)
        length2 = max((start_date - dt2).days, 0)
        if length1 == 0 and length2 == 0:
            start_date = max(dt1, dt2)
        elif length1 == 0 and length2 > 0:
            start_date = dt1
        elif length1 > 0 and length2 == 0:
            start_date = dt2
        length1 = max((start_date - dt1).days, 0)
        length2 = max((start_date - dt2).days, 0)
        
        if disable_tqdm is False: # For debugging
            print(f"Advancing the TempLSTM1 model to the start date: {start_date} (length={length1} days)")
            print(f"Advancing the TempLSTM2 model to the start date: {start_date} (length={length2} days)")
        
        # Advance the LSTM models to the start date 
        def update_until_(lstm, length):
            # If the length is 0, we do not need to update the LSTM model
            if length == 0:
                return None
            
            # Get unscaled lstm input data
            unscaled_data = lstm.get_unscaled_values(lead_time=length) 
            for var in lstm.x_vars:
                lstm.set_value(var, unscaled_data[var])
            lstm.update_until(length)
                
        if disable_tqdm is False: # For debugging
            print(f"Advancing TempLSTM models to the {start_date} ...")
            
        update_until_(lstm=self.lstm1, length=length1)
        update_until_(lstm=self.lstm2, length=length2)
        
        # Dates
        self.start_date = start_date
        self.end_date = pd.to_datetime(end_date)
        self.current_date = self.start_date 
        self.length = (self.end_date - self.start_date).days + 1
        self.t = 0
        
        # Input data
        self.X_1 = np.nan
        self.X_2 = np.nan
        self.X_map = np.nan
        self.Q_C = np.nan
        self.Q_i = np.nan
        self.Q_L = np.nan
        
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
        self.thermal_release = np.nan # mgd
        self.remained_bank_amount = thermal_mitigation_bank_size  # mgd
        
        # Others        
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
            
    
    def load_data(self, database):
        # Load db
        
        db = database[self.start_date:self.end_date] 
        self.Q_C = db["QbcTavg_Q_C"].values
        self.Q_i = db["QbcTavg_Q_i"].values
        
        length = self.length
        lstm1 = self.lstm1
        self.X_1 = lstm1.get_unscaled_values(lead_time=length) # A DataFrame
        
        lstm2 = self.lstm2
        self.X_2 = lstm2.get_unscaled_values(lead_time=length) # A DataFrame
        
        self.X_map = db[self.rf_model_map.x_vars].values

    
    def update_until(self, t=None, date=None):
        if t is not None:
            if t >= len(self.X_1) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t}. Must be between current time step {self.t + 1} and {len(self.X_1) + 1}.")
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            t = (date - self.start_date).days
            if t >= len(self.X_1) + 1 or t < self.t:
                raise ValueError(f"Invalid time step {t} for date {date}. Must be between current date {self.current_date} and {self.end_date + pd.Timedelta(days=1)}.")

        length = t - self.t
        if length == 0:
            return None
        
        t = self.t
        # Somehow the data length need to be 1 step higher for lstm
        # # python slicing is exclusive for the end index so need to add 1
        X_map = self.X_map[t:t+length].reshape(length, -1)
        Q_C_ = self.Q_C[t:t+length]
        Q_i_ = self.Q_i[t:t+length]
        
        lstm = self.lstm1
        for var in lstm.x_vars:
            lstm.set_value(var, self.X_1[var][t:t+length+1])
        lstm.update_until(t+length)
        T_C_mu = np.zeros(length)
        T_C_sd = np.zeros(length)
        lstm.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
        lstm.get_value("channel_water_surface_water__sd_max_of_temperature", T_C_sd)
        
        lstm = self.lstm2
        for var in lstm.x_vars:
            lstm.set_value(var, self.X_2[var][t:t+length+1])
        lstm.update_until(t+length)
        T_i_mu = np.zeros(length)
        T_i_sd = np.zeros(length)
        lstm.get_value("channel_water_surface_water__mu_max_of_temperature", T_i_mu)
        lstm.get_value("channel_water_surface_water__sd_max_of_temperature", T_i_sd)
        
        Tavg_L_mu, Tavg_L_sd = self.blend_hot_cold_water(
            T_C_mu=T_C_mu, T_i_mu=T_i_mu, 
            T_C_sd=T_C_sd, T_i_sd=T_i_sd, 
            Q_C=Q_C_, Q_i=Q_i_
        )
        
        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        X_map[:, self.rf_model_map.x_vars.index("QbcTavg_T_L")] = Tavg_L_mu
        T_L_mu, _, _ = self.rf_model_map.predict(X_map, quantile=None)
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
        self.current_date += pd.Timedelta(days=length)
        return self.T_L_mu, self.T_L_sd
    
    def update(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, dQ_C=0):
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
        dQ_C : float, optional
            The change in Cannonsville reservoir downstream flow.
        
        Returns
        -------
        tuple
            The predicted T_L_mu and T_L_sd at the specified time step.
        """
        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name
        
        if Q_C is not None:
            self.Q_C[t] = Q_C
            try:
                self.X_1.loc[t, Q_C_lstm_var_name] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
            try:
                self.X_2.loc[t, Q_C_lstm_var_name] = Q_C
            except ValueError:
                print(f"Warning: '{Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
            
        if Q_i is not None:
            self.Q_i[t] = Q_i
            try:
                self.X_2.loc[t, Q_i_lstm_var_name] = Q_i
            except ValueError:
                print(f"Warning: '{Q_i_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")
            
        if cannonsville_storage_pct is not None:
            try:
                self.X_1.loc[t, cannonsville_storage_pct_lstm_var_name] = cannonsville_storage_pct
            except ValueError:
                print(f"Warning: '{cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
        
        #Weired thing here
        #return self.update_until(t=t+1, date=None)
        
        
        
    
    def forecast(self, t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0):
        X_1 = self.X_1.loc[t:t+lead_time+1, :].copy().reset_index(drop=True)
        X_2 = self.X_2.loc[t:t+lead_time+1, :].copy().reset_index(drop=True)
        X_map = self.X_map[t:t+lead_time+1].reshape(lead_time+1, -1)
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
        Tavg_L_mu = (T_C_mu*Q_C + T_i_mu*Q_i)/(Q_C + Q_i)
        # Assuming T_i and T_C are independent
        Tavg_L_sd = np.sqrt((T_C_sd**2 * Q_C**2 + T_i_sd**2 * Q_i**2) / (Q_C + Q_i)**2)
        return Tavg_L_mu, Tavg_L_sd
    
    





