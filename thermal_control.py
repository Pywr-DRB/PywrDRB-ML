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

class TemperatureModel():
    def __init__(self, start_date, activate_thermal_control, activate_input_bias_correction,
                 Q_C_lstm_var_name, Q_i_lstm_var_name, cannonsville_storage_pct_lstm_var_name,
                 subfolder, 
                 disable_tqdm, debug):
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
        self.debug = debug
        
        # Final water temperature at Lordville
        self.mu, self.sd = np.nan, np.nan
        
        # Forecasted water temperature at Lordville
        self.forecasted_mu_arr, self.forecasted_sd_arr = np.nan, np.nan
        
        # Indicate whether to activate thermal control
        self.activate_thermal_control = activate_thermal_control
        self.activate_input_bias_correction = activate_input_bias_correction
        self.Q_C_lstm_var_name = Q_C_lstm_var_name
        self.Q_i_lstm_var_name = Q_i_lstm_var_name
        self.cannonsville_storage_pct_lstm_var_name = cannonsville_storage_pct_lstm_var_name
        
        # Load db
        self.db = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), parse_dates=True, index_col=[0])[["QbcTavg_Q_C", "QbcTavg_Q_i"]]
        
        #!! For debugging purposes
        if debug:
            self.records = {
                "date": [],
                "T_C_mu": [],
                "T_C_sd": [],
                "T_i_mu": [],
                "T_i_sd": [],
                "Tavg_mu": [],
                "Tavg_sd": [],
                "T_L_mu": [],
                "T_L_sd": [],
                "Q_C": [],
                "Q_i": [],
                "cannonsville_storage_pct": [],
                "forecasted_mu_arr": [],
                "forecasted_sd_arr": [],
                "bias_Q_i": [],
                "bias_Q_C": [],
                "bias_cannonsville_storage_pct": []
            }
        
        # Predict the Cannonsville reservoir release temperature (T_C)
        self.subfolder = subfolder
        lstm1 = bmi_lstm()
        lstm1.initialize(config_file=pn.models.get(f"{subfolder}/TempLSTM1.yml"), train=False, root_dir=pn.get())
        self.lstm1 = lstm1
        
        # Predict the water temperature for east branch (T_i)
        lstm2 = bmi_lstm()
        lstm2.initialize(config_file=pn.models.get(f"{subfolder}/TempLSTM2.yml"), train=False, root_dir=pn.get())
        self.lstm2 = lstm2
        
        # Map Tavg to Tmax (T_L) at Lordville
        rf_model = joblib.load(pn.models.get("rf_model.gz"))
        self.rf_model = rf_model
        
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Identify the start date of the LSTM models
        dt1 = pd.to_datetime(lstm1.get_current_date())
        dt2 = pd.to_datetime(lstm2.get_current_date())
        # Get the start date, which the latest among LSTM1, LSTM2, and pywrdrb start dates
        if start_date is not None:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            dt = max(dt1, dt2)
            
        self.start_date = min(max(dt1, dt2, dt), dt)
        length1 = max((self.start_date - dt1).days, 0)
        length2 = max((self.start_date - dt2).days, 0)
        if length1 == 0 and length2 == 0:
            self.start_date = max(dt1, dt2)
        elif length1 == 0 and length2 > 0:
            self.start_date = dt1
        elif length1 > 0 and length2 == 0:
            self.start_date = dt2
        length1 = max((self.start_date - dt1).days, 0)
        length2 = max((self.start_date - dt2).days, 0)
        
        if disable_tqdm is False: # For debugging
            print(f"Advancing the TempLSTM1 model to the start date: {self.start_date} (length={length1} days)")
            print(f"Advancing the TempLSTM2 model to the start date: {self.start_date} (length={length2} days)")
        
        # Advance the LSTM models to the start date 
        def update_until(lstm, length):
            # If the length is 0, we do not need to update the LSTM model
            if length == 0:
                return None
            
            # Get unscaled lstm input data
            unscaled_data = lstm.get_unscaled_values(lead_time=length) 
            for var in lstm.x_vars:
                lstm.set_value(var, unscaled_data[var])
            lstm.update_until(length)
                
        if disable_tqdm is False: # For debugging
            print(f"Advancing TempLSTM models to the {self.start_date} ...")
            
        update_until(lstm=self.lstm1, length=length1)
        update_until(lstm=self.lstm2, length=length2)
        
        # Safenet to ensure the LSTM is only update once per timestep
        self.current_date = self.start_date 
        self.t = 0
        self.db = self.db.loc[self.current_date:, :]
        
        # Initialize thermal mitigation bank size (MGD)
        self.thermal_mitigation_bank_size = 1620
        self.remained_bank_amount = 1620
        
        # Contorl algorithm (externally provided) -> used in make_control_release
        # We will turn this into a built-in control algorithm in the future
        self.control_algorithm = lambda ml_model, Q_C, Q_i, cannonsville_storage_pct, current_date: 3 #np.nan
        
        # Bias correction for the LSTM models inputs for forecasting. 
        # If the thermal release happens in the previous timestep, the state dynamics 
        # will be different than the training inputs. We calculate the difference between
        # the two to shift the input array for forecasting.
        self.bias_correction_dict = {
            "Q_C": 0.0,  # Bias correction for Cannonsville downstream flow
            "Q_i": 0.0,  # Bias correction for East Branch downstream flow and natural inflow to Lordville
            "cannonsville_storage_pct": 0.0  # Bias correction for Cannonsville reservoir storage percentage
            }
        
    def make_control_release(self, **kwargs):
        """
        Make the thermal control release decision based on the LSTM model predictions.
        
        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float
            The percentage of the Cannonsville reservoir storage.
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.
        
        Returns
        -------
        float
            The thermal control release amount in million gallons per day (MGD).
        """
        # activate if self.activate_thermal_control is True
        # Here is the place to plugin control algorithm
        
        control_algorithm = self.control_algorithm
        if callable(control_algorithm) is False:
            raise ValueError("The control_algorithm must be a callable function.")
        
        thermal_release = control_algorithm(
            ml_model=self,
            **kwargs
            )
        return thermal_release
    
    def update(self, Q_C=None, Q_i=None, cannonsville_storage_pct=None, dQ_C=0): #, current_date):
        """
        Forward the LSTM models to one step.
        
        Parameters
        ----------
        Q_C : float
            The Cannonsville reservoir downstream flow (01425000).
        Q_i : float
            The East Branch downstream flow (01417000) and natural inflow to Lordville.
        cannonsville_storage_pct : float
            The percentage of the Cannonsville reservoir storage.        
        current_date : pywr.core.CurrentDate
            The current date in the model, used to determine if the LSTM models need to be updated.
        """      
        lstm1 = self.lstm1
        lstm2 = self.lstm2
        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name
        activate_input_bias_correction = self.activate_input_bias_correction
        
        # Update the LSTM1 models with the current flow values
        unscaled_data = lstm1.get_unscaled_values(lead_time=0) # Retrieve unscaled data for the current date
        for var in lstm1.x_vars:
            if var == Q_C_lstm_var_name and Q_C is not None:
                lstm1.set_value(Q_C_lstm_var_name, Q_C) + dQ_C
                if activate_input_bias_correction:
                    self.bias_correction_dict["Q_C"] = Q_C - unscaled_data.loc[0, Q_C_lstm_var_name]
            elif var == cannonsville_storage_pct_lstm_var_name and cannonsville_storage_pct is not None:
                lstm1.set_value(cannonsville_storage_pct_lstm_var_name, cannonsville_storage_pct)
                if activate_input_bias_correction:
                    self.bias_correction_dict["cannonsville_storage_pct"] = cannonsville_storage_pct - unscaled_data.loc[0, cannonsville_storage_pct_lstm_var_name]
            else:
                lstm1.set_value(var, unscaled_data.loc[0, var]) 
        lstm1.update()
        
        # Update the LSTM2 models with the current flow values
        unscaled_data = lstm2.get_unscaled_values(lead_time=0) # Retrieve unscaled data for the current date
        for var in lstm2.x_vars:
            if var == Q_i_lstm_var_name and Q_i is not None:
                lstm2.set_value(Q_i_lstm_var_name, Q_i)
                if activate_input_bias_correction:
                    self.bias_correction_dict["Q_i"] = Q_i - unscaled_data.loc[0, Q_i_lstm_var_name]
            elif var == Q_C_lstm_var_name and Q_C is not None: # connected model
                lstm2.set_value(Q_C_lstm_var_name, Q_C) + dQ_C
            else:
                lstm2.set_value(var, unscaled_data.loc[0, var]) 
        lstm2.update()
        
        # Can switch to iloc
        if Q_C is None:
            Q_C = self.db.loc[self.current_date, "QbcTavg_Q_C"] + dQ_C
        if Q_i is None:
            Q_i = self.db.loc[self.current_date, "QbcTavg_Q_i"]
        
        # T_C
        T_C_mu = np.zeros(1)
        T_C_sd = np.zeros(1)
        lstm1.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
        lstm1.get_value("channel_water_surface_water__sd_max_of_temperature", T_C_sd)
        T_C_mu, T_C_sd = T_C_mu[0], T_C_sd[0]
        
        # T_i
        T_i_mu = np.zeros(1)
        T_i_sd = np.zeros(1)
        lstm2.get_value("channel_water_surface_water__mu_max_of_temperature", T_i_mu)
        lstm2.get_value("channel_water_surface_water__sd_max_of_temperature", T_i_sd)
        T_i_mu, T_i_sd = T_i_mu[0], T_i_sd[0]
        
        # Tavg
        Tavg_mu = (T_C_mu*Q_C + T_i_mu*Q_i)/(Q_C + Q_i)
        # Assuming T_i and T_C are independent
        Tavg_sd = np.sqrt((T_C_sd**2 * Q_C**2 + T_i_sd**2 * Q_i**2) / (Q_C + Q_i)**2)
        
        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        rf_model = self.rf_model
        T_L_mu = rf_model.predict([[Tavg_mu]])[0]
        T_L_sd = Tavg_sd # assuming a constant sd for T_L
        self.mu, self.sd = T_L_mu, T_L_sd
        
        # For debugging purposes
        if self.debug:
            records = self.records
            #records["date"].append(previous_date)
            records["T_C_mu"].append(T_C_mu)
            records["T_C_sd"].append(T_C_sd)
            records["T_i_mu"].append(T_i_mu)
            records["T_i_sd"].append(T_i_sd)
            records["Tavg_mu"].append(Tavg_mu)
            records["Tavg_sd"].append(Tavg_sd)
            records["T_L_mu"].append(T_L_mu)
            records["T_L_sd"].append(T_L_sd)
            records["Q_C"].append(Q_C)
            records["Q_i"].append(Q_i)
            records["cannonsville_storage_pct"].append(cannonsville_storage_pct)
        
        self.current_date += timedelta(days=1) # Avoid updating the LSTM models multiple times in a single timestep
        self.t += 0
        return T_L_mu, T_L_sd
    
    def forecast(self, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0, dQ_C=0):
            
        lstm1 = self.lstm1
        lstm2 = self.lstm2
        Q_C_lstm_var_name = self.Q_C_lstm_var_name
        Q_i_lstm_var_name = self.Q_i_lstm_var_name
        cannonsville_storage_pct_lstm_var_name = self.cannonsville_storage_pct_lstm_var_name
        
        # Update the LSTM1 models with the current flow values
        unscaled_data = lstm1.get_unscaled_values(lead_time=lead_time) # Retrieve unscaled data for the current date
        for var in lstm1.x_vars:
            if var == Q_C_lstm_var_name and Q_C is not None:
                Q_C_array = unscaled_data[var] + self.bias_correction_dict["Q_C"] + dQ_C
                Q_C_array[Q_C_array < 0] = 0
                Q_C_array[0] = Q_C  # Ensure the first value is the current Q_C
                # Do a bias correction for the Q_C variable 
                lstm1.set_value(Q_C_lstm_var_name, Q_C_array)
            elif var == cannonsville_storage_pct_lstm_var_name and cannonsville_storage_pct is not None:
                cannonsville_storage_pct_array = unscaled_data[var] + self.bias_correction_dict["cannonsville_storage_pct"]
                cannonsville_storage_pct_array[cannonsville_storage_pct_array < 0] = 0
                cannonsville_storage_pct_array[0] = cannonsville_storage_pct  # Ensure the first value is the current cannonsville_storage_pct
                lstm1.set_value(cannonsville_storage_pct_lstm_var_name, cannonsville_storage_pct_array)
            else:
                lstm1.set_value(var, unscaled_data[var]) 
        df_T_C = lstm1.forecast(lead_time=lead_time)
        
        # Update the LSTM2 models with the current flow values
        unscaled_data = lstm2.get_unscaled_values(lead_time=lead_time) # Retrieve unscaled data for the current date
        for var in lstm2.x_vars:
            if var == Q_i_lstm_var_name and Q_i is not None:
                Q_i_array = unscaled_data[var] + self.bias_correction_dict["Q_i"]
                Q_i_array[Q_i_array < 0] = 0
                Q_i_array[0] = Q_i  # Ensure the first value is the current Q_i
                lstm2.set_value(Q_i_lstm_var_name, Q_i_array)
            elif var == Q_C_lstm_var_name and Q_C is not None: # connected model
                Q_C_array = unscaled_data[var] + self.bias_correction_dict["Q_C"] + dQ_C
                Q_C_array[Q_C_array < 0] = 0
                Q_C_array[0] = Q_C  # Ensure the first value is the current Q_C
                lstm2.set_value(Q_C_lstm_var_name, Q_C_array)
            else:
                lstm2.set_value(var, unscaled_data[var]) 
        df_T_i = lstm2.forecast(lead_time=lead_time)
        
        # Can switch to iloc
        if Q_C is None:
            Q_C_array = self.db.loc[self.current_date:self.current_date+timedelta(days=lead_time), "QbcTavg_Q_C"].values + dQ_C
        if Q_i is None:
            Q_i_array = self.db.loc[self.current_date:self.current_date+timedelta(days=lead_time), "QbcTavg_Q_i"].values
        
        # Tavg
        Tavg_mu = (df_T_C["mu"]*Q_C_array + df_T_i["mu"]*Q_i_array)/(Q_C_array + Q_i_array)
        # Assuming T_i and T_C are independent
        Tavg_sd = np.sqrt((df_T_C["sd"]**2 * Q_C_array**2 + df_T_i["sd"]**2 * Q_i_array**2) / (Q_C_array + Q_i_array)**2)
        
        # T_L (Tmax at Lordville) Using a random forest model to map Tavg to T_L
        rf_model = self.rf_model
        T_L_mu = rf_model.predict(Tavg_mu.values.reshape(-1, 1))
        T_L_sd = Tavg_sd.values # assuming a constant sd for T_L
        self.forecasted_mu_arr, self.forecasted_sd_arr = T_L_mu, T_L_sd
        
        # For debugging purposes
        if self.debug:
            records = self.records
            records["forecasted_mu_arr"].append(T_L_mu)
            records["forecasted_sd_arr"].append(T_L_sd)
            records["bias_Q_i"].append(self.bias_correction_dict["Q_i"])
            records["bias_Q_C"].append(self.bias_correction_dict["Q_C"])
            records["bias_cannonsville_storage_pct"].append(self.bias_correction_dict["cannonsville_storage_pct"])
        
            print(f"T_C: {df_T_C['mu'].values}; T_i: {df_T_i['mu'].values}")
        return T_L_mu, T_L_sd

data = {
    "start_date": "2020-08-01", #"1979-01-01",
    "activate_thermal_control": True,
    "activate_input_bias_correction": False,
    "Q_C_lstm_var_name": "QbcTavg_Q_C",
    "Q_i_lstm_var_name": "QbcTavg_Q_i",
    "cannonsville_storage_pct_lstm_var_name": "bc_cannonsville_storage_pct",
    "subfolder": "TempLSTM_lag1_Q_C",
    "disable_tqdm": True,
    "debug": True,
}

#%%
model = TemperatureModel(**data)

model.update()
model.forecast(dQ_C=100)
model.update()
model.update()

#%%
def return_dps_func(*params):
    def dps_func(ml_model, **kwargs):
        if ml_model.current_date.month in [6, 7, 8]:
            ml_model.forecast()
            if ml_model.forecasted_mu_arr[0] > 24:
                return 300 # Thermal release MGD
            else:
                return 0
        else:
            return 0
    
    return dps_func

params = []
dps_func = return_dps_func(*params)

temperature_model = TemperatureModel(**data)
temperature_model.control_algorithm = dps_func

for i in tqdm(range(16436)):
    dQ_C = temperature_model.make_control_release()
    temperature_model.update(dQ_C=dQ_C)





