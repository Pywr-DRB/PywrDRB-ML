import os
import yaml
# Configuration file functionality
from pathlib import Path
# Basic utilities
import numpy as np
import pandas as pd
from tqdm import tqdm
# Need these for BMI
from bmipy import Bmi
# LSTM here is based on PyTorch
import torch
from .torch_models import LSTMWithHead
from .training_utils import * #(MaskedGMMLoss, MaskedCMALLoss, MaskedUMALLoss, rmse_masked, fit_torch_model)
from .sampling_utils import * #(sample_GMM, sample_CMAL, sample_UMAL)

USE_PATH = True

class bmi_lstm(Bmi):

    def __init__(self):
        """
        Initializes the BMI LSTM model for stream temperature prediction with necessary attributes and sets up the model for simulation.

        Attributes:
        - _name (str): Name of the model, set to "LSTM for Stream Temperature".
        - _values (dict): Dictionary to store model variables.
        - _var_loc (str): Location where model variables are defined, set to "node".
        - _start_time (float): Start time of the simulation, set to 0.
        - _end_time (float): End time of the simulation, set to maximum possible float value.
        - _time_units (str): Units of time used in the simulation, set to "day".
        - _time_step_size (float): Size of the time step used in the simulation, set to 1.0.
        - _input_var_names (list): A list of input variable names (CSDMS standard names).
        - _output_var_names (list): A list of output variable names (CSDMS standard names).
        - _var_name_units_map (dict): A dictionary mapping CSDMS standard names to internal variable names and units.

        Note: The _end_time attribute is set to the maximum possible float value to indicate that there is no predefined end time for the simulation.

        Static Attributes:
        - _att_map (dict): Dictionary containing static attributes of the model such as model_name, version, and author_name.

        Author:
        - Jacob Zwart (with model contributions from Jeremy Diaz and others)
        """
        super(bmi_lstm, self).__init__()
        self._name = "LSTM for Stream Temperature"
        self._values = {}
        self._var_loc = "node"
        # self._var_grid_id = 0
        # self._var_grid_type = "scalar"
        self._start_time = 0
        self._end_time = np.finfo("d").max
        self._time_units = "day"
        self._time_step_size = 1.0

    #----------------------------------------------
    # Static attributes of the model
    #----------------------------------------------
    _att_map = {
        'model_name':         'LSTM for Stream Temperature',
        'version':            '1.0',
        'author_name':        'Jacob Zwart' } # Jeremy Diaz and many others

    #---------------------------------------------
    # Input variable names (CSDMS standard names)
    #---------------------------------------------
    
    _input_var_names = [
        'tmmx',
        'tmmn',
        'pr',
        'srad',
        'vs',
        'rmax',
        'rmin',
        'sph',
        'QobsTmax_Q_i',
        'QobsTmax_Q_C',
        'QobsTmax_Q_L',
        'QobsTmax_T_i',
        'QobsTmax_T_C',
        'QobsTmax_T_L',
        'QobsTmax_Q_Cnotr',
        'QobsTmax_Q_Lnotr',
        'QobsTmax_T_Lnotr',
        'QobsTavg_Q_i',
        'QobsTavg_Q_C',
        'QobsTavg_Q_L',
        'QobsTavg_T_i',
        'QobsTavg_T_C',
        'QobsTavg_T_L',
        'QobsTavg_Q_Cnotr',
        'QobsTavg_Q_Lnotr',
        'QobsTavg_T_Lnotr',
        'QbcTmax_Q_i',
        'QbcTmax_Q_C',
        'QbcTmax_Q_L',
        'QbcTmax_T_i',
        'QbcTmax_T_C',
        'QbcTmax_T_L',
        'QbcTavg_Q_i',
        'QbcTavg_Q_C',
        'QbcTavg_Q_L',
        'QbcTavg_T_i',
        'QbcTavg_T_C',
        'QbcTavg_T_L',
        'tmmx_water_lordville_src',
        'tmmn_water_lordville_src',
        'tavg_water_lordville_src',
        'discharge_lordville_src',
        'tmmx_water_cannonsville_src',
        'tmmn_water_cannonsville_src',
        'tavg_water_cannonsville_src',
        'discharge_cannonsville_src',
        'tmmx_water_src',
        'tmmn_water_src',
        'tavg_water_src',
        'discharge_src',
        'nyc_storage_pct',
        'cannonsville_storage_pct',
        'pepacton_storage_pct',
        'neversink_storage_pct',
        'bc_cannonsville_storage_pct',
        'bc_pepacton_storage_pct',
        'rel_thermal',
        'seg_id_nat',
        'QobsTmax_T_C_lag_1',
        'QobsTavg_T_C_lag_1',
        'QbcTmax_T_C_lag_1',
        'QbcTavg_T_C_lag_1',
        'QobsTmax_T_L_lag_1',
        'QobsTmax_T_Lnotr_lag_1',
        'QobsTavg_T_L_lag_1',
        'QobsTavg_T_Lnotr_lag_1',
        'QbcTmax_T_L_lag_1',
        'QbcTavg_T_L_lag_1',
        'QobsTmax_T_i_lag_1',
        'QobsTavg_T_i_lag_1',
        'QbcTmax_T_i_lag_1',
        'QbcTavg_T_i_lag_1',
        'saltfront',
        'saltfront_src',
        '01463500',
        '01474500',
        'Q_Trenton_bc',
        'Q_Schuylkill_bc',
        'doy_cos',
        'doy',
        'seg_id_nat',
        'y_src_lag_1'
        ]


    #---------------------------------------------
    # Output variable names (CSDMS standard names)
    #---------------------------------------------
    _output_var_names = ['channel_water_surface_water__mu_max_of_temperature',
                        'channel_water_surface_water__sd_max_of_temperature']


    #------------------------------------------------------
    # Create a Python dictionary that maps CSDMS Standard
    # Names to the model's internal variable names.
    #------------------------------------------------------
    _var_name_units_map = {
        'tmmx': ['tmmx', 'degC'],
        'tmmn': ['tmmn', 'degC'],
        'pr': ['pr', 'm day-1'],
        'srad': ['srad', 'W m-2'],
        'vs': ['vs', 'm s-1'],
        'rmax': ['rmax', 'fraction'],
        'rmin': ['rmin', 'fraction'],
        'sph': ['sph', 'kg/kg'],
        'QobsTmax_Q_i': ['QobsTmax_Q_i', '--'],
        'QobsTmax_Q_C': ['QobsTmax_Q_C', '--'],
        'QobsTmax_Q_L': ['QobsTmax_Q_L', '--'],
        'QobsTmax_T_i': ['QobsTmax_T_i', '--'],
        'QobsTmax_T_C': ['QobsTmax_T_C', '--'],
        'QobsTmax_T_L': ['QobsTmax_T_L', '--'],
        'QobsTmax_Q_Cnotr': ['QobsTmax_Q_Cnotr', '--'],
        'QobsTmax_Q_Lnotr': ['QobsTmax_Q_Lnotr', '--'],
        'QobsTmax_T_Lnotr': ['QobsTmax_T_Lnotr', '--'],
        'QobsTavg_Q_i': ['QobsTavg_Q_i', '--'],
        'QobsTavg_Q_C': ['QobsTavg_Q_C', '--'],
        'QobsTavg_Q_L': ['QobsTavg_Q_L', '--'],
        'QobsTavg_T_i': ['QobsTavg_T_i', '--'],
        'QobsTavg_T_C': ['QobsTavg_T_C', '--'],
        'QobsTavg_T_L': ['QobsTavg_T_L', '--'],
        'QobsTavg_Q_Cnotr': ['QobsTavg_Q_Cnotr', '--'],
        'QobsTavg_Q_Lnotr': ['QobsTavg_Q_Lnotr', '--'],
        'QobsTavg_T_Lnotr': ['QobsTavg_T_Lnotr', '--'],
        'QbcTmax_Q_i': ['QbcTmax_Q_i', '--'],
        'QbcTmax_Q_C': ['QbcTmax_Q_C', '--'],
        'QbcTmax_Q_L': ['QbcTmax_Q_L', '--'],
        'QbcTmax_T_i': ['QbcTmax_T_i', '--'],
        'QbcTmax_T_C': ['QbcTmax_T_C', '--'],
        'QbcTmax_T_L': ['QbcTmax_T_L', '--'],
        'QbcTavg_Q_i': ['QbcTavg_Q_i', '--'],
        'QbcTavg_Q_C': ['QbcTavg_Q_C', '--'],
        'QbcTavg_Q_L': ['QbcTavg_Q_L', '--'],
        'QbcTavg_T_i': ['QbcTavg_T_i', '--'],
        'QbcTavg_T_C': ['QbcTavg_T_C', '--'],
        'QbcTavg_T_L': ['QbcTavg_T_L', '--'],
        'tmmx_water_lordville_src': ['tmmx_water_lordville_src', '--'],
        'tmmn_water_lordville_src': ['tmmn_water_lordville_src', '--'],
        'tavg_water_lordville_src': ['tavg_water_lordville_src', '--'],
        'discharge_lordville_src': ['discharge_lordville_src', '--'],
        'tmmx_water_cannonsville_src': ['tmmx_water_cannonsville_src', '--'],
        'tmmn_water_cannonsville_src': ['tmmn_water_cannonsville_src', '--'],
        'tavg_water_cannonsville_src': ['tavg_water_cannonsville_src', '--'],
        'discharge_cannonsville_src': ['discharge_cannonsville_src', '--'],
        'tmmx_water_src': ['tmmx_water_src', '--'],
        'tmmn_water_src': ['tmmn_water_src', '--'],
        'tavg_water_src': ['tavg_water_src', '--'],
        'discharge_src': ['discharge_src', '--'],
        'nyc_storage_pct': ['nyc_storage_pct', '%'],
        'cannonsville_storage_pct': ['cannonsville_storage_pct', '%'],
        'pepacton_storage_pct': ['pepacton_storage_pct', '%'],
        'neversink_storage_pct': ['neversink_storage_pct', '%'],
        'bc_cannonsville_storage_pct': ['bc_cannonsville_storage_pct', '%'],
        'bc_pepacton_storage_pct': ['bc_pepacton_storage_pct', '%'],
        'rel_thermal': ['rel_thermal', 'mgd'],
        'seg_id_nat': ['seg_id_nat', '--'],
        'QobsTmax_T_C_lag_1': ['QobsTmax_T_C_lag_1', '--'],
        'QobsTavg_T_C_lag_1': ['QobsTavg_T_C_lag_1', '--'],
        'QbcTmax_T_C_lag_1': ['QbcTmax_T_C_lag_1', '--'],
        'QbcTavg_T_C_lag_1': ['QbcTavg_T_C_lag_1', '--'],
        'QobsTmax_T_L_lag_1': ['QobsTmax_T_L_lag_1', '--'],
        'QobsTmax_T_Lnotr_lag_1': ['QobsTmax_T_Lnotr_lag_1', '--'],
        'QobsTavg_T_L_lag_1': ['QobsTavg_T_L_lag_1', '--'],
        'QobsTavg_T_Lnotr_lag_1': ['QobsTavg_T_Lnotr_lag_1', '--'],
        'QbcTmax_T_L_lag_1': ['QbcTmax_T_L_lag_1', '--'],
        'QbcTavg_T_L_lag_1': ['QbcTavg_T_L_lag_1', '--'],
        'QobsTmax_T_i_lag_1': ['QobsTmax_T_i_lag_1', '--'],
        'QobsTavg_T_i_lag_1': ['QobsTavg_T_i_lag_1', '--'],
        'QbcTmax_T_i_lag_1': ['QbcTmax_T_i_lag_1', '--'],
        'QbcTavg_T_i_lag_1': ['QbcTavg_T_i_lag_1', '--'],
        'saltfront': ['saltfront', '--'],
        'saltfront_src': ['saltfront_src', '--'],
        '01463500': ['01463500', '--'],
        '01474500': ['01474500', '--'],
        'Q_Trenton_bc': ['Q_Trenton_bc', '--'],
        'Q_Schuylkill_bc': ['Q_Schuylkill_bc', '--'],
        'doy_cos': ['doy_cos', '--'],
        'doy': ['doy', '--'],
        'seg_id_nat': ['seg_id_nat', '--'],
        'y_src_lag_1': ['y_src_lag_1', '--'],
        'saltfront_lag_1': ['saltfront_lag_1', '--']
        }
    
    def __getattribute__(self, item):
        """
        Customize instance attribute access.

        For those items that correspond to BMI input or output variables (which should be in numpy arrays) and have
        values that are just a single-element array, deviate from the standard behavior and return the single array
        element. Fall back to the default behavior in any other case.

        This supports having a BMI variable be backed by a numpy array, while also allowing the attribute to be used as
        just a scalar, as it is in many places for this type.

        Parameters:
        - item (str): The name of the attribute item to get.

        Returns:
        The value of the named item.
        """
        # Have these work explicitly (or else loops)
        if item == '_input_var_names' or item == '_output_var_names':
            return super(bmi_lstm, self).__getattribute__(item)

        # By default, for things other than BMI variables, use normal behavior
        if item not in super(bmi_lstm, self).__getattribute__('_input_var_names') and item not in super(bmi_lstm, self).__getattribute__('_output_var_names'):
            return super(bmi_lstm, self).__getattribute__(item)

        # Return the single scalar value from any ndarray of size 1
        value = super(bmi_lstm, self).__getattribute__(item)
        if isinstance(value, np.ndarray) and value.size == 1:
            return value[0]
        else:
            return value


    def __setattr__(self, key, value):
        """
        Customized instance attribute mutator functionality.

        For those attribute with keys indicating they are a BMI input or output variable (which should be in numpy
        arrays), wrap any scalar ``value`` as a one-element numpy array and use that in a nested call to the superclass
        implementation of this function.  In any other cases, just pass the given ``key`` and ``value`` to a nested
        call.

        This supports automatically having a BMI variable be backed by a numpy array, even if it is initialized using a
        scalar, while otherwise maintaining standard behavior.

        Parameters:
        - key (str): The name of the attribute to set.
        - value: The value to assign to the attribute.

        Returns:
        None
        """
        # Have these work explicitly (or else loops)
        if key == '_input_var_names' or key == '_output_var_names':
            super(bmi_lstm, self).__setattr__(key, value)

        # Pass thru if value is already an array
        if isinstance(value, np.ndarray):
            super(bmi_lstm, self).__setattr__(key, value)
        # Override to put scalars into ndarray for BMI input/output variables
        elif key in self._input_var_names or key in self._output_var_names:
            super(bmi_lstm, self).__setattr__(key, np.array([value]))
        # By default, use normal behavior
        else:
            super(bmi_lstm, self).__setattr__(key, value)


    #------------------------------------------------------------
    #------------------------------------------------------------
    # BMI: Model Control Functions
    #------------------------------------------------------------
    #------------------------------------------------------------

    #-------------------------------------------------------------------
    def initialize(self, config_file = None, torch_seed = None, train = False, root_dir = None):
        """Initialize the BMI LSTM model with BMI configuration file.

        This function initializes the BMI LSTM model by performing the following steps:
        - Creates lookup tables for variable names and units.
        - Initializes all variables to zero.
        - Reads the BMI configuration file.
        - Loads training configurations and data.
        - Loads scaler values and LSTM states from trained model.
        - Initializes an LSTM model with specified parameters.
        - Initializes values for input to the LSTM.
        - Loads pre-trained LSTM weights.
        - Sets the simulation start time.
        - Retrieves verbosity level from the BMI configuration.

        Parameters:
            config_file (str): Path to the BMI configuration file in YAML format. Default is None.

        Note:
            The BMI configuration file directs the subsequent actions of the model.
        """

        # ----- Create some lookup tables from the long variable names --------#
        self._var_name_map_long_first = {long_name:self._var_name_units_map[long_name][0] for \
                                         long_name in self._var_name_units_map.keys()}
        self._var_name_map_short_first = {self._var_name_units_map[long_name][0]:long_name for \
                                          long_name in self._var_name_units_map.keys()}
        self._var_units_map = {long_name:self._var_name_units_map[long_name][1] for \
                                          long_name in self._var_name_units_map.keys()}

        # -------------- Initalize all the variables --------------------------#
        # -------------- so that they'll be picked up with the get functions --#
        for var_name in list(self._var_name_units_map.keys()):
            # ---------- All the variables are single values ------------------#
            # ---------- so just set to zero for now.        ------------------#
            self._values[var_name] = 0
            setattr(self, var_name, 0)

        # -------------- Read in the BMI configuration -------------------------#
        # This will direct all the next moves.
        if config_file is not None:
            config_file = Path(config_file)
            #----------------------------------------------------------
            # Note: config_file should have type 'str', vs. being a
            #       Path object. So apply Path in initialize().
            #----------------------------------------------------------
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file {config_file} not found.")

            with open(config_file, 'r') as fp:
                cfg = yaml.safe_load(fp)
            self.cfg_bmi = self._parse_config(cfg)
        else:
            raise ValueError("Error: No configuration provided, nothing to do...")
        self.config_file = config_file
        # ---------- Set torch random seed -----------------
        seed = None
        if torch_seed is not None:
            seed = torch_seed
        elif config_file is not None:
            seed = self.cfg_bmi.get("seed")
        if seed is not None:
            print(f"Set torch seed to {seed}")
            torch.manual_seed(seed)
        else:
            print("No torch seed is assigned")

        if root_dir is not None:
            self.cfg_bmi['root_dir'] = root_dir
        # ------------- Load in the configuration file for the specific LSTM --#
        # This will include all the details about how the model was trained
        self.get_training_configurations()
        self.get_data(train=train)

        if train:
            print("Initialized model for training. Use <model_object_name>.train_model() to start the training")

        else:
            # scalar values
            self.get_scaler_values()

            # load LSTM states from trained model
            h = np.load(self.out_h_file, allow_pickle=True)
            c = np.load(self.out_c_file, allow_pickle=True)
            self.h_t = torch.from_numpy(h).float()
            self.c_t = torch.from_numpy(c).float()
            self.dist_mat_all = torch.from_numpy(self.dist_mat_all).float()

            # ------------- Initialize an LSTM model ------------------------------#
            self.lstm = LSTMWithHead(input_dim = self.n_feat,
                                    lstm_hidden_dim = self.hidden_units,
                                    adj_matrix = self.dist_mat_all,
                                    dropout = self.dropout_rate,
                                    recur_dropout = self.recurrent_dropout_rate,
                                    head = self.head,
                                    head_hidden_dim = self.head_hidden_dim,
                                    head_n_dist = self.head_n_dist,
                                    delta_temp_input_dim=self.n_feat_delta)

            # ------------------ Initialize values for the input to the LSTM -----------#
            self.initialize_forcings()

            self.lstm.load_state_dict(torch.load(self.weights_dir + '/weights.pth', weights_only=True))

            # start of the simulation time
            self.t = self._start_time

        # Gather verbosity lvl from bmi-config for stdout printing, etc.
        self.verbose = self.cfg_bmi['verbose']
        self.results = None # For simple run

    def update(self):
        """Update the LSTM model for a single time step.

        This function updates the LSTM model for a single time step by performing the following steps:
        - Prepares input data for the LSTM model.
        - Makes predictions using the LSTM model.
        - Generates samples based on the prediction using different uncertainty quantification methods.
        - Saves predicted values to the appropriate output variables.
        - Advances the simulation time by one time step.
        """

        self.create_scaled_input_tensor()

        # make predictions
        if self.mc_dropout:
            self.lstm.train()
            self.lstm_output, (self.h_t, self.c_t) = self.lstm(self.input_tensor,
                                                       (self.h_t, self.c_t), self.dist_mat_all, self.input_delta_tensor)
            self.lstm.eval()
        else:
            self.lstm.eval()
            self.lstm_output, (self.h_t, self.c_t) = self.lstm(self.input_tensor,
                                                       (self.h_t, self.c_t), self.dist_mat_all, self.input_delta_tensor)
        if self.produce_ensembles:
            if self.head == 'GMM':
                self.samples = sample_GMM(self.lstm_output, self.n_samples)
            if self.head == 'CMAL':
                self.samples = sample_CMAL(self.lstm_output, self.n_samples)
            if self.head == 'UMAL':
                self.samples = sample_UMAL(self.lstm_output, self.n_samples, self.head_n_dist, self.x.shape[0])
            if self.head == 'Regression':
                self.preds = self.lstm_output['y_hat'].detach().numpy()
                self.preds = np.repeat(self.preds, self.n_samples, axis = 2)
            else:
                self.preds = self.samples

        setattr(self, 'channel_water_surface_water__mu_max_of_temperature',
                self.lstm_output['mu'].detach().numpy()[0,0,0])
        setattr(self, 'channel_water_surface_water__sd_max_of_temperature',
                self.lstm_output['sigma'].detach().numpy()[0,0,0])

        self.t += self.get_time_step()

    def simple_run(self, force_zero_vars=[]):
        if self.results is not None:
            print("The results already exist.")
            return self.results
        # Get unscaled lstm input data
        mu_ft = []
        sd_ft = []
        length = self.x.shape[1]
        unscaled_data = self.get_unscaled_values(lead_time=length)
        
        for _ in tqdm(range(length)):
            for vi, var in enumerate(unscaled_data):
                if var in self.x_vars:
                    if var in force_zero_vars:
                        self.set_value(var, 0)
                    else:
                        self.set_value(var, unscaled_data.loc[int(self.t), var])
                
                if self.delta_temp_layer and var in self.delta_vars:
                    if var in force_zero_vars:
                        self.set_value(var, 0)
                    else:
                        self.set_value(var, unscaled_data.loc[int(self.t), var])           
                
            self.update()

            mu_pred = np.zeros(1)
            sd_pred = np.zeros(1)
            self.get_value("channel_water_surface_water__mu_max_of_temperature", mu_pred)
            self.get_value("channel_water_surface_water__sd_max_of_temperature", sd_pred)
            mu_ft.append(mu_pred[0])
            sd_ft.append(sd_pred[0])
            
        res = pd.DataFrame()
        res["date"] = self.dates_all.flatten()
        res['mu_ft'] = mu_ft
        res['sd_ft'] = sd_ft
        res.set_index("date", inplace=True)
        res = res.reindex(pd.date_range(start=res.index.min(), end=res.index.max(), freq="D"))
        self.results = res
        return res

    def update_until(self, timestep):
        # Get unscaled lstm input data
        # mu_ft = []
        # sd_ft = []
        while self.t < timestep:
            unscaled_data = self.get_unscaled_values(lead_time=0)
            for vi, var in enumerate(unscaled_data):
                if var in self.x_vars:
                    self.set_value(var, unscaled_data.loc[0, var])
                
                # if self.delta_temp_layer and var in self.delta_vars:
                #     self.set_value(var, unscaled_data.loc[int(self.t), var])           
                
            self.update()

            # mu_pred = np.zeros(1)
            # sd_pred = np.zeros(1)
            # self.get_value("channel_water_surface_water__mu_max_of_temperature", mu_pred)
            # self.get_value("channel_water_surface_water__sd_max_of_temperature", sd_pred)
            # mu_ft.append(mu_pred[0])
            # sd_ft.append(sd_pred[0])
            
        #return mu_ft, sd_ft
        
    def update_until_org(self, time):
        """Update model until a particular model time step.
        Parameters
        ----------
        time : float
            Time to run model until.
        """
        cur_step = int(self.get_current_time())

        if time <= cur_step:
            raise ValueError(f"End time, {time}, must be larger than current time, {cur_step}")

        # Check if the requested time to update until extends beyond the available data,
        # if it does, then throw an error
        input_data_len = len(self.get_value_ptr(self.x_vars[0]))
        if (time+1) != (input_data_len + cur_step):
            target_time = input_data_len + cur_step - 1
            raise ValueError(f"The end time, {time}, does not match the length of data provided.\n Please use an end time of {target_time}, or provide a different amount of input data.")
        self.create_scaled_input_tensor()
        # make predictions
        if self.mc_dropout:
            self.lstm.train()
            self.lstm_output, (self.h_t, self.c_t) = self.lstm(self.input_tensor,
                                                        (self.h_t, self.c_t), self.dist_mat)
            self.lstm.eval()
        else:
            self.lstm.eval()
            self.lstm_output, (self.h_t, self.c_t) = self.lstm(self.input_tensor,
                                                        (self.h_t, self.c_t), self.dist_mat)

        setattr(self, 'channel_water_surface_water__mu_max_of_temperature',
                self.lstm_output['mu'].detach().numpy()[0,:,0])
        setattr(self, 'channel_water_surface_water__sd_max_of_temperature',
                self.lstm_output['sigma'].detach().numpy()[0,:,0])

        self.t = time + 1


    def forecast(self, lead_time=None):
        """
        Forecast for a particular lead time, which is the number of days into the future from time t.

        Parameters:
            lead_time (int): Number of days into the future to forecast.

        Returns:
            pd.DataFrame: A DataFrame containing forecasted values ('mu' and 'sd') for each time step up to the lead time.
        """
        # Save the current states of the model's hidden and cell states,
        #  as well as the current time step and input vars
        saved_h_t = self.h_t.clone()
        saved_c_t = self.c_t.clone()
        saved_t = self.t
        # Save the current random state in torch
        saved_rng_state = torch.get_rng_state()

        # If lead_time is not provided, use the default forecast horizon from the configuration
        if lead_time is None:
            lead_time = self.f_horizon
            print(f' setting forecat lead time to f_horizon from the config file: {lead_time}')

        # Check if the requested lead time extends beyond the available data,
        #  if it does, then throw an error
        input_val = self.get_value_ptr(self.x_vars[0])
        # Check if the value is a float or an array
        if isinstance(input_val, float):
            # If it's a float, the length is 1
            input_data_len = 1
        elif isinstance(input_val, (np.ndarray, list)):
            # If it's an array or list, get the length
            input_data_len = len(input_val)
        else:
            # Raise an error if it's an unexpected type
            raise TypeError("Unexpected type returned from get_value_ptr(). Expected a float or an array.")

        if (lead_time+1) > input_data_len:
            raise ValueError(f"The forecast lead time, {lead_time} days, extends beyond the length of data provided.\n Please use a forecast lead time of {input_data_len-1} or smaller.")

        # Save the current input variables so they aren't overwritten in the forecast loop
        if self.delta_temp_layer:
            saved_input_values, saved_input_delta_values = self.get_input_array(return_array=True)
            input_values_ndims = saved_input_values.ndim
            input_delta_values_ndims = saved_input_delta_values.ndim
        else:
            saved_input_values = self.get_input_array(return_array=True)
            input_values_ndims = saved_input_values.ndim

        # Initialize an empty DataFrame to store the forecasted values
        forecasts = pd.DataFrame(columns=['mu','sd'])

        try:
            # Run the model until lead_time
            for i in range(lead_time + 1):
                # Using yesterday's temp prediction as today's input
                if np.isfinite(self.lag_var_mean):
                    if i > 0:
                        # Get temp predictions from yesterday and insert into today's lagged temp value
                        saved_input_values[self.lag_var_pos, i] = forecasted_mu
                    elif input_values_ndims == 2:
                        if np.isnan(saved_input_values[self.lag_var_pos, i]):
                            raise ValueError(f'Lagged temperature input for day 0 is NaN, please initialize with a prediction or observation of temperature')
                    elif input_values_ndims == 1:
                        if np.isnan(saved_input_values[self.lag_var_pos]):
                            raise ValueError(f'Lagged temperature input for day 0 is NaN, please initialize with a prediction or observation of temperature')

                # Set the values
                for k in range(len(self.x_vars)):
                    if input_values_ndims == 2:
                        self.set_value(self.x_vars[k], saved_input_values[k,i])
                    elif input_values_ndims == 1:
                        self.set_value(self.x_vars[k], saved_input_values[k])
                if self.delta_temp_layer:
                    for k in range(len(self.delta_vars)):
                        if input_delta_values_ndims == 2:
                            self.set_value(self.delta_vars[k], saved_input_delta_values[k,i])
                        elif input_delta_values_ndims == 1:
                            self.set_value(self.delta_vars[k], saved_input_delta_values[k])

                # Update the model
                self.update()

                # Retrieve the forecasted values
                forecasted_mu = getattr(self, 'channel_water_surface_water__mu_max_of_temperature', np.zeros(1))
                forecasted_sd = getattr(self, 'channel_water_surface_water__sd_max_of_temperature', np.zeros(1))

                # append new forecast to output
                forecasts.loc[len(forecasts)] = [forecasted_mu, forecasted_sd]

        finally:
            # Restore the saved states to ensure the model's state is consistent after forecasting
            self.h_t = saved_h_t
            self.c_t = saved_c_t
            self.t = saved_t
            self.input_array = saved_input_values
            if self.delta_temp_layer:
                self.input_delta_array = saved_input_delta_values
            self.set_values_from_input_array()
            # Restore the saved random state
            torch.set_rng_state(saved_rng_state)

        return forecasts

    def train_model(self):
        """Train the model using the specified training data.

        This function prepares the training data and trains the model using either pre-training or fine-tuning
        methods or both, depending on the configuration. It converts the training data from NumPy arrays to PyTorch
        tensors, initializes model parameters, and manages the training process including the application of
        different loss functions based on the model head type.

        The function also handles the creation of necessary directories for saving model weights and outputs
        the hidden states after training.

        Returns:
            None: This function does not return any value. It performs in-place updates to the model state
            and saves the resulting hidden states to specified output files.

        Raises:
            OSError: If the specified weights directory cannot be created or accessed.

        Notes:
            - The function supports different model heads (e.g., GMM, CMAL, UMAL, Regression) and adjusts
            the training procedure accordingly.
            - The model can be pre-trained and/or fine-tuned based on the configuration flags `self.pre_train`
            and `self.fine_tune`.
            - Dropout behavior is managed based on the `self.mc_dropout` flag to ensure correct training
            and evaluation modes.
        """

        self.x_trn = torch.from_numpy(self.x_trn).float()
        self.x_val = torch.from_numpy(self.x_val).float()
        self.x_test = torch.from_numpy(self.x_test).float()
        self.x_all = torch.from_numpy(self.x_all).float()
        if self.delta_temp_layer:
            self.x_delta_trn = torch.from_numpy(self.x_delta_trn).float()
            self.x_delta_val = torch.from_numpy(self.x_delta_val).float()
            self.x_delta_test = torch.from_numpy(self.x_delta_test).float()
            self.x_delta_all = torch.from_numpy(self.x_delta_all).float()
        self.y_trn = torch.from_numpy(self.y_trn).float()
        self.y_val = torch.from_numpy(self.y_val).float()
        self.obs_trn = torch.from_numpy(self.obs_trn).float()
        self.obs_val = torch.from_numpy(self.obs_val).float()

        # self.obs_test = torch.from_numpy(self.obs_test).float()
        self.dist_mat_trn = torch.from_numpy(self.dist_mat_trn).float()
        self.dist_mat_val = torch.from_numpy(self.dist_mat_val).float()
        self.dist_mat_test = torch.from_numpy(self.dist_mat_test).float()
        self.start_h_trn = torch.from_numpy(self.start_h_trn).float()
        self.start_c_trn = torch.from_numpy(self.start_c_trn).float()
        self.start_h_val = torch.from_numpy(self.start_h_val).float()
        self.start_c_val = torch.from_numpy(self.start_c_val).float()
        self.start_h_test = torch.from_numpy(self.start_h_test).float()
        self.start_c_test = torch.from_numpy(self.start_c_test).float()
        self.start_h_all_dates = torch.from_numpy(self.start_h_all_dates).float()
        self.start_c_all_dates = torch.from_numpy(self.start_c_all_dates).float()

        self.umal_n_taus_train = [1, self.head_n_dist][self.umal_extend_batch]
        if self.head == 'GMM':
            if self.weight_loss:
                self.loss_fn = MaskedGMMLoss_weighted
            else:
                self.loss_fn = MaskedGMMLoss
        if self.head == 'CMAL':
            self.loss_fn = MaskedCMALLoss
        if self.head == 'UMAL':
            self.loss_fn = MaskedUMALLoss
            if self.umal_extend_batch == True:
                self.x_trn = self.x_trn.repeat(self.head_n_dist, 1, 1)
                self.y_trn = self.y_trn.repeat(self.head_n_dist, 1, 1)
                self.x_trn_fine = self.x_trn_fine.repeat(self.head_n_dist, 1, 1)
                self.obs_trn = self.obs_trn.repeat(self.head_n_dist, 1, 1)
        if self.head == 'Regression':
            self.loss_fn = rmse_masked

        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

        if self.pre_train:
            self.pretrain_model = LSTMWithHead(self.n_feat,
                                               self.hidden_units,
                                               self.dist_mat_trn,
                                               self.dropout_rate,
                                               self.recurrent_dropout_rate,
                                               self.head,
                                               self.head_hidden_dim,
                                               self.head_n_dist,
                                               self.n_feat_delta)

            self.pretrain_model.train() # ensure that dropout layers are active
            # train_torch(model=self.pretrain_model,
            #             loss_function=self.loss_fn,
            #             optimizer=torch.optim.Adam(self.pretrain_model.parameters(), lr = self.learn_rate_pre),
            #             x_train=self.x_trn,
            #             y_train=self.y_trn,
            #             h_train=self.start_h_trn,
            #             c_train=self.start_c_trn,
            #             h_val=self.start_h_val,
            #             c_val=self.start_c_val,
            #             weighting_matrix_train=self.dist_mat_trn,
            #             weighting_matrix_val=self.dist_mat_val,
            #             batch_size=self.x_trn.shape[0],
            #             max_epochs=self.n_epochs_pre,
            #             head=self.head,
            #             umal_extend_batch=self.umal_extend_batch,
            #             umal_n_taus_train=self.umal_n_taus_train,
            #             umal_tau_min=self.umal_tau_min,
            #             umal_tau_max=self.umal_tau_max,
            #             early_stopping_patience=self.early_stopping_patience,
            #             x_val=self.x_val,
            #             y_val=self.y_val,
            #             shuffle=False, # hard coding to False
            #             weights_file=self.weights_file,
            #             log_file=self.log_file,
            #             device='cpu', # hard coding
            #             keep_portion=None)
            fit_torch_model(model = self.pretrain_model,
                            x = self.x_trn,
                            y = self.y_trn,
                            h = self.start_h_trn,
                            c = self.start_c_trn,
                            weighting_matrix = self.dist_mat_trn,
                            epochs = self.n_epochs_pre,
                            loss_fn = self.loss_fn,
                            optimizer = torch.optim.Adam(self.pretrain_model.parameters(), lr = self.learn_rate_pre),
                            gpu = self.gpu,
                            head = self.head,
                            early_stopping_patience = self.early_stopping_patience,
                            weights_file = self.weights_file,
                            umal_extend_batch = self.umal_extend_batch,
                            umal_n_taus_train = self.umal_n_taus_train,
                            umal_tau_min = self.umal_tau_min,
                            umal_tau_max = self.umal_tau_max,
                            weight_loss=self.weight_loss,
                            weight_threshold=self.weight_threshold,
                            weight_value=self.weight_value,
                            x_delta=self.x_delta_trn)

        if self.fine_tune:
            self.fine_tune_model = LSTMWithHead(self.n_feat,
                                                self.hidden_units,
                                                self.dist_mat_trn,
                                                self.dropout_rate,
                                                self.recurrent_dropout_rate,
                                                self.head,
                                                self.head_hidden_dim,
                                                self.head_n_dist,
                                                self.n_feat_delta)

            if self.pre_train:
                self.fine_tune_model.load_state_dict(torch.load(self.weights_file, weights_only=True))

            self.fine_tune_model.train() # ensure that dropout layers are active
            # train_torch(model=self.fine_tune_model,
            #             loss_function=self.loss_fn,
            #             optimizer=torch.optim.Adam(self.fine_tune_model.parameters(), lr = self.learn_rate_fine),
            #             x_train=self.x_trn,
            #             y_train=self.obs_trn,
            #             h_train=self.start_h_trn,
            #             c_train=self.start_c_trn,
            #             h_val=self.start_h_val,
            #             c_val=self.start_c_val,
            #             weighting_matrix_train=self.dist_mat_trn,
            #             weighting_matrix_val=self.dist_mat_val,
            #             batch_size=self.x_trn.shape[0],
            #             max_epochs=self.n_epochs_fine,
            #             head=self.head,
            #             umal_extend_batch=self.umal_extend_batch,
            #             umal_n_taus_train=self.umal_n_taus_train,
            #             umal_tau_min=self.umal_tau_min,
            #             umal_tau_max=self.umal_tau_max,
            #             early_stopping_patience=self.early_stopping_patience,
            #             x_val=self.x_val,
            #             y_val=self.y_val,
            #             shuffle=False, # hard coding to False
            #             weights_file=self.weights_file,
            #             log_file=self.log_file,
            #             device='cpu', # hard coding
            #             keep_portion=None)
            fit_torch_model(model = self.fine_tune_model,
                            x = self.x_trn,
                            y = self.obs_trn,
                            h = self.start_h_trn,
                            c = self.start_c_trn,
                            weighting_matrix = self.dist_mat_trn,
                            epochs = self.n_epochs_fine,
                            loss_fn = self.loss_fn,
                            optimizer = torch.optim.Adam(self.fine_tune_model.parameters(), lr = self.learn_rate_fine),
                            gpu = self.gpu,
                            head = self.head,
                            early_stopping_patience = self.early_stopping_patience,
                            weights_file = self.weights_file,
                            umal_extend_batch = self.umal_extend_batch,
                            umal_n_taus_train = self.umal_n_taus_train,
                            umal_tau_min = self.umal_tau_min,
                            umal_tau_max = self.umal_tau_max,
                            weight_loss=self.weight_loss,
                            weight_threshold=self.weight_threshold,
                            weight_value=self.weight_value,
                            x_delta=self.x_delta_trn)

            self.fine_tune_model.load_state_dict(torch.load(self.weights_file, weights_only=True))
            if self.mc_dropout:
                pred_all, (self.h, self.c) = self.fine_tune_model.train()(self.x_all, [self.start_h_all_dates, self.start_c_all_dates], self.dist_mat_test, self.x_delta_all) # ensure that dropout layers are active w/ .train()
            else:
                pred_all, (self.h, self.c) = self.fine_tune_model.eval()(self.x_all, [self.start_h_all_dates, self.start_c_all_dates], self.dist_mat_test, self.x_delta_all) # ensure that dropout layers are inactive w/ .eval()

            # pred_mean = unscale_output(y_scl=pred_all['mu'].detach().numpy()[0,:,0],
            #                            y_std=self.obs_data_sd,
            #                            y_mean=self.obs_data_mean)
            # pred_sd = unscale_output(y_scl=pred_all['sigma'].detach().numpy()[0,:,0],
            #                            y_std=self.obs_data_sd,
            #                            y_mean=self.obs_data_mean)
            # obs_unscaled = unscale_output(y_scl=self.obs_all[0,:,0],
            #                                 y_std=self.obs_data_sd,
            #                                 y_mean=self.obs_data_mean)
            df_out_preds = pd.DataFrame({'date': self.dates_all[0,:,0],
                                         'mean': pred_all['mu'].detach().numpy()[0,:,0],
                                         'sd': pred_all['sigma'].detach().numpy()[0,:,0],
                                         'obs': self.obs_all[0,:,0]})
            df_out_preds.to_parquet(path = self.all_dates_preds_file)
            np.save(self.out_h_file, self.h.detach().numpy())
            np.save(self.out_c_file, self.c.detach().numpy())

        print("Done training")


    def finalize(self):
        """Finalize model"""
        self._model = None


    def initialize_forcings(self):
        """Initialize all forcings to zero.

        This function initializes all forcing variables to zero. It iterates through the list of forcing variable names
        and sets each variable to zero using the BMI standard naming convention. For BMI functions that require long variable
        names, they should be mapped to the model's short names before taking action.

        Note:
            A BMI-enabled model should not use long variable names internally. Instead, it should use convenient short names
            for internal processing.
        """
        print('Initializing all forcings to 0...')
        for forcing_name in self.x_vars:
            print('  forcing_name =', forcing_name)
            setattr(self, forcing_name, 0)
        if self.delta_temp_layer:
            for forcing_name in self.delta_vars:
                print('  forcing_name =', forcing_name)
                setattr(self, forcing_name, 0)

    #------------------------------------------------------------
    def get_scaler_values(self):
        """Calculate mean and standard deviation for the input variables for LSTM.

        This function calculates the mean and standard deviation for the input variables and LSTM outputs.
        It extracts the mean and standard deviation values from pre-calculated data and assigns them to
        corresponding attributes in the model.
        """
        self.input_mean = self.x_data_mean
        self.input_std = self.x_data_sd
        if self.delta_temp_layer:
            self.input_delta_mean = self.x_delta_mean
            self.input_delta_std = self.x_delta_sd

    #------------------------------------------------------------
    def get_unscaled_values(self, lead_time=0, vars='all') -> pd.DataFrame:
        """
        Get the unscaled input values for the model based on the current time and lead time.

        Parameters:
            lead_time (int): Number of days into the future to request unscaled data.
            vars (list): List of variable names that correspond to model inputs.

        Returns:
            pd.DataFrame: Unscaled input values for the specified lead time with columns as variable names.
        """
        cur_time = int(self.t)
        end_time = cur_time + lead_time + 1 # python slicing is exclusive for the end index so need to add 1

        if vars == 'all':
            vars = self.x_vars
            if self.delta_temp_layer:
                delta_vars = self.delta_vars

        # Check if all requested variables are in x_vars
        missing_vars = [var for var in vars if var not in self.x_vars]
        if missing_vars:
            raise ValueError(f"The following variables are not in x_vars: {missing_vars}; please select from {self.x_vars}")

        unscaled_data = self.x[0, cur_time:end_time, :] * (self.input_std + 1e-10) + self.input_mean
        # Create a DataFrame from the unscaled data
        df = pd.DataFrame(unscaled_data, columns=self.x_vars)
        filtered_df = df[vars]

        if self.delta_temp_layer:
            unscaled_delta_data = self.x_delta_all[0, cur_time:end_time, :] * (self.input_delta_std + 1e-10) + self.input_delta_mean
            df_delta = pd.DataFrame(unscaled_delta_data, columns=self.delta_vars)
            filtered_delta_df = df_delta[delta_vars]
            filtered_df = pd.concat([filtered_df, filtered_delta_df], axis = 1)
            # Drop duplicate columns
            filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()]

        return filtered_df

    #------------------------------------------------------------
    def create_scaled_input_tensor(self, VERBOSE=False):
        """Create a scaled input tensor for the LSTM model.

        This function creates a scaled input tensor for the LSTM model using the mean and standard deviation
        values of the input variables calculated previously. It iterates through the input variables, maps
        short variable names to long variable names, retrieves their values from the model, and appends them
        to an input list. The input values are then normalized and reshaped into a tensor suitable for input
        to the LSTM model.

        Args:
            VERBOSE (bool, optional): If True, print verbose information during tensor creation. Default is False.

        """
        self.get_input_array(VERBOSE)

        DEBUG = False
        if (VERBOSE):
            print('Normalizing the tensor...')
            print('  input_mean =', self.input_mean )
            print('  input_std  =', self.input_std )
            print()
        # Center and scale the input values for use in torch
        # adding small number in case there is a std of zero
        #  Final array shape should be [n_locations, n_time, n_features]
        if self.input_array.ndim == 1:
            # input array is only one time step
            n_time = 1
            self.input_array_scaled = ((self.input_array - self.input_mean) / (self.input_std + 1e-10)).reshape(self.n_segs,n_time,self.n_feat)
            if self.delta_temp_layer:
                self.input_delta_array_scaled = ((self.input_delta_array - self.input_delta_mean) / (self.input_delta_std + 1e-10)).reshape(self.n_segs,n_time,self.n_feat_delta)
        elif self.input_array.ndim == 2:
            # input array is greater than one time step
            n_time = self.input_array.shape[1]
            self.input_array_scaled = ((self.input_array - self.input_mean.reshape(self.n_feat,1)) / (self.input_std.reshape(self.n_feat,1) + 1e-10)).reshape(self.n_segs,n_time,self.n_feat)
            if self.delta_temp_layer:
                self.input_delta_array_scaled = ((self.input_delta_array - self.input_delta_mean.reshape(self.n_feat_delta,1)) / (self.input_delta_std.reshape(self.n_feat_delta,1) + 1e-10)).reshape(self.n_segs,n_time,self.n_feat_delta)
        if (DEBUG):
            print('### input_list =', self.input_list)
            print('### input_array =', self.input_array)
            print('### dtype(input_array) =', self.input_array.dtype )
            print('### type(input_array_scaled) =', type(self.input_array_scaled))
            print('### dtype(input_array_scaled) =', self.input_array_scaled.dtype )
            print()
        self.input_tensor = torch.from_numpy(self.input_array_scaled).float()
        if self.delta_temp_layer:
            self.input_delta_tensor = torch.from_numpy(self.input_delta_array_scaled).float()
        else:
            self.input_delta_tensor = None

    #------------------------------------------------------------
    def get_input_array(self, VERBOSE=False, return_array=False):
        """
        Retrieve and compile input variables into a NumPy array.

        This function gathers input variables specified by the `x_vars` attribute,
        retrieves their corresponding values, and compiles them into a single
        NumPy array. It also includes debug output if the VERBOSE or DEBUG
        flags are enabled.

        The method assumes that the input variables are stored in the object's
        attributes and that they can be accessed via the `getattr()` function.
        It also normalizes the resulting array to ensure it is of type `float64`.

        Args:
            VERBOSE (bool, optional): If True, print verbose information during tensor creation. Default is False.

        Attributes:
            self.input_list (list): A list of the retrieved input values.
            self.input_array (np.ndarray): A NumPy array containing the input values.

        Returns:
            None: This method does not return a value but sets attributes
            `input_list` and `input_array` on the instance.
        """
        n_inputs = len(self.x_vars)
        self.input_list = []
        DEBUG = False
        for k in range(n_inputs):
            short_name = self.x_vars[k]
            long_name  = self._var_name_map_short_first[ short_name ]
            # vals = self.get_value( long_name )
            vals = getattr( self, short_name )

            self.input_list.append( vals )
            if (VERBOSE or DEBUG):
                print('  short_name =', short_name )
                print('  long_name  =', long_name )
                array = getattr( self, short_name )
                # array = self.get_value( long_name )
                print('  type       =', type(vals) )
                print('  vals       =', vals )
        if self.delta_temp_layer:
            n_delta_inputs = len(self.delta_vars)
            self.input_delta_list = []
            for k in range(n_delta_inputs):
                short_name = self.delta_vars[k]
                long_name  = self._var_name_map_short_first[ short_name ]
                # vals = self.get_value( long_name )
                vals = getattr( self, short_name )

                self.input_delta_list.append( vals )
                if (VERBOSE or DEBUG):
                    print('  short_name =', short_name )
                    print('  long_name  =', long_name )
                    array = getattr( self, short_name )
                    # array = self.get_value( long_name )
                    print('  type       =', type(vals) )
                    print('  vals       =', vals )
        #--------------------------------------------------------
        # W/o setting dtype here, it was "object_", and crashed
        #--------------------------------------------------------
        ## self.input_array = np.array( self.input_list )
        self.input_array = np.array( self.input_list, dtype='float64' )
        if self.delta_temp_layer:
            self.input_delta_array = np.array(self.input_delta_list, dtype='float64')

        if return_array:
            if self.delta_temp_layer:
                return self.input_array, self.input_delta_array
            else:
                return self.input_array


    #------------------------------------------------------------
    def set_values_from_input_array(self):
        """
        Set model variable values from the input array.

        This function iterates through the input variables specified by the
        `x_vars` attribute and assigns values from the `input_array`
        to each corresponding variable. The values are set using the
        `set_value` method.

        The method handles both 1-D and 2-D NumPy arrays.

        Attributes:
            self.input_array (np.ndarray): A 1-D or 2-D NumPy array containing
            the input values to be assigned to the model variables.
            self.x_vars (list): A list of variable names that correspond
            to the rows in `input_array`.

        Returns:
            None: This method does not return a value but updates the
            model variables with the values from `input_array`.
        """
        n_inputs = len(self.x_vars)
        if self.delta_temp_layer:
            n_delta_inputs = len(self.delta_vars)

        # Check if input_array is 1-D
        if self.input_array.ndim == 1:
            if self.input_array.shape[0] != n_inputs:
                raise ValueError("The number of rows in input_array must match the number of variables in x_vars.")
            for k in range(n_inputs):
                short_name = self.x_vars[k]
                vals = self.input_array[k]
                self.set_value(short_name, vals)
            if self.delta_temp_layer:
                for k in range(n_delta_inputs):
                    short_name = self.delta_vars[k]
                    vals = self.input_delta_array[k]
                    self.set_value(short_name, vals)

        # If input_array is 2-D
        elif self.input_array.ndim == 2:
            if self.input_array.shape[0] != n_inputs:
                raise ValueError("The number of rows in input_array must match the number of variables in x_vars.")
            for k in range(n_inputs):
                short_name = self.x_vars[k]
                vals = self.input_array[k, :]
                self.set_value(short_name, vals)
            if self.delta_temp_layer:
                for k in range(n_delta_inputs):
                    short_name = self.delta_vars[k]
                    vals = self.input_delta_array[k, :]
                    self.set_value(short_name, vals)

        else:
            raise ValueError("input_array must be a 1-D or 2-D NumPy array.")

    #------------------------------------------------------------
    def get_value(self, var_name: str, dest: np.ndarray) -> np.ndarray:
        """
        Copy values for the named variable into the provided destination array.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        dest : np.ndarray
            A numpy array into which to copy the variable values.
        Returns
        -------
        np.ndarray
            Copy of values.
        """
        dest[:] = self.get_value_ptr(var_name)
        return dest

    #-------------------------------------------------------------------
    def get_value_ptr(self, var_name: str) -> np.ndarray:
        """
        Get reference to values.

        Get the backing reference - i.e., the backing numpy array - for the given variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        Returns
        -------
        np.ndarray
            Value array.
        """
        # We actually need this function to return the backing array, so bypass override of __getattribute__ (that
        # extracts scalar) and use the base implementation
        return super(bmi_lstm, self).__getattribute__(var_name)

    #-------------------------------------------------------------------
    def set_value(self, var_name: str, values):
        """Set model values.

        This function sets the values of a model variable specified by its CSDMS Standard Name.

        Parameters:
            var_name (str): Name of the variable as CSDMS Standard Name.
            values (np.ndarray or pd.Series or pd.DataFrame): Values to set for the variable.

        """
        # Ensure values is a NumPy scalar
        if isinstance(values, (pd.Series, pd.DataFrame)):
            values = values.to_numpy()  # Convert Pandas objects to NumPy array

        if isinstance(values, np.ndarray):
            if values.size == 1:
                values = values.item()  # Get the scalar value from the array
            else:
                values = values
        elif isinstance(values, (int, float)):
            values = np.array(values).item()  # Convert to NumPy scalar

        setattr( self, var_name, values )

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    # BMI: Variable Information Functions
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    def get_var_name(self, long_var_name):
        """Get the short variable name corresponding to a long variable name.

        This function retrieves the short variable name corresponding to a given long variable name.
        It looks up the variable name in the model's internal mapping from long variable names to short
        variable names.

        Parameters:
            long_var_name (str): The long variable name to look up.

        Returns:
            str: The short variable name corresponding to the given long variable name.

        """
        return self._var_name_map_long_first[ long_var_name ]

    #-------------------------------------------------------------------
    def get_var_units(self, long_var_name):
        """Get the units of a variable specified by its long name.

        This function retrieves the units of a variable specified by its long name.
        It looks up the variable name in the model's internal mapping from long variable names to units.

        Parameters:
            long_var_name (str): The long variable name for which to retrieve units.

        Returns:
            str: The units of the variable specified by its long name.

        """
        return self._var_units_map[ long_var_name ]


    def get_training_configurations(self):
        """Retrieve model configurations from the BMI configuration file.

        This function retrieves various training configurations from the BMI configuration file and assigns them
        to corresponding attributes in the model.
        """
        self.root_dir = self.cfg_bmi.get('root_dir')

        if self.root_dir is not None:
            self.train_dir = os.path.join(self.root_dir, self.cfg_bmi['train_dir'])
            self.data_file = os.path.join(self.root_dir, self.cfg_bmi['data_file'])
        else:
            self.train_dir = self.cfg_bmi['train_dir']
            self.data_file = self.cfg_bmi['data_file']
            
        if not os.path.exists(self.train_dir): # Create subfolder in models/
            os.mkdir(self.train_dir)
        
        self.model_id = self.cfg_bmi['model_id']
        self.weights_dir = os.path.join(self.train_dir, f'{self.model_id}_wgts')
        self.weights_file = os.path.join(self.weights_dir, 'weights.pth')
        self.train_preds_file = os.path.join(self.train_dir, f'{self.model_id}_train_preds.parquet')
        self.log_file = os.path.join(self.train_dir, f'{self.model_id}_train_log.csv')
        self.out_h_file = os.path.join(self.train_dir, f'{self.model_id}_h.npy')
        self.out_c_file = os.path.join(self.train_dir, f'{self.model_id}_c.npy')
        self.test_preds_file = os.path.join(self.train_dir, f'{self.model_id}_test_preds.parquet')
        self.all_dates_preds_file = os.path.join(self.train_dir, f'{self.model_id}_all_dates_preds.parquet')

        self.model_type = str(self.cfg_bmi['model_type'])
        self.delta_temp_layer = bool(self.cfg_bmi['delta_temp_layer'])
        self.mc_dropout = bool(self.cfg_bmi['mc_dropout'])
        self.recurrent_dropout_rate = float(self.cfg_bmi['recurrent_dropout_rate'])
        self.dropout_rate = float(self.cfg_bmi['dropout_rate'])
        self.temp_obs_sd = float(self.cfg_bmi['temp_obs_sd'])
        self.h_sd = float(self.cfg_bmi['h_sd'])
        self.c_sd = float(self.cfg_bmi['c_sd'])
        self.hidden_units = int(self.cfg_bmi['hidden_units'])
        self.force_pos = bool(self.cfg_bmi['force_pos'])
        self.update_h = bool(self.cfg_bmi['update_h'])
        self.update_c = bool(self.cfg_bmi['update_c'])
        self.n_segs = int(self.cfg_bmi['n_segs'])
        self.f_horizon = int(self.cfg_bmi['f_horizon'])
        self.head = str(self.cfg_bmi['head'])
        self.head_hidden_dim = int(self.cfg_bmi['head_hidden_units'])
        self.head_n_dist = int(self.cfg_bmi['head_n_distr'])
        self.weight_loss = bool(self.cfg_bmi['weight_loss'])
        self.weight_threshold = float(self.cfg_bmi['weight_threshold'])
        self.weight_value = float(self.cfg_bmi['weight_value'])
        self.produce_ensembles = bool(self.cfg_bmi['produce_ensembles'])
        self.n_samples = int(self.cfg_bmi['n_samples'])
        self.pre_train = bool(self.cfg_bmi['pre_train'])
        self.fine_tune = bool(self.cfg_bmi['fine_tune'])
        self.n_epochs_pre = int(self.cfg_bmi['n_epochs_pre'])
        self.n_epochs_fine = int(self.cfg_bmi['n_epochs_fine'])
        self.early_stopping_patience = int(self.cfg_bmi['early_stopping_patience'])
        self.gpu = int(self.cfg_bmi['gpu'])
        self.umal_extend_batch = bool(self.cfg_bmi['umal_extend_batch'])
        self.umal_tau_min = float(self.cfg_bmi['umal_tau_min'])
        self.umal_tau_max = float(self.cfg_bmi['umal_tau_max'])
        self.learn_rate_pre = float(self.cfg_bmi['learn_rate_pre'])
        self.learn_rate_fine = float(self.cfg_bmi['learn_rate_fine'])
        self.torch_seed = int(self.cfg_bmi['seed'])

    def get_data(self, train = False):
        """Load and retrieve data from the data file.

        This function loads data from the specified data file and retrieves various data arrays and attributes
        needed for model training and evaluation. It assigns the retrieved data to corresponding attributes
        in the model.

        Parameters:
            train (bool): A flag indicating whether to load training data (True) or test data (False).
                        Default is False.

        Returns:
            None: This function does not return any value. It assigns the loaded data to the instance attributes.
        """
        data = np.load(self.data_file, allow_pickle=True)
        self.filter_data(data, train)

    def filter_data(self, data, train):
        """Filter and assign data based on the training flag.

        This function processes the loaded data to filter out the relevant variables based on whether the
        data is for training or testing. It retrieves lagged variable information and assigns the appropriate
        data arrays to the instance attributes.

        Parameters:
            data (dict): A dictionary containing the loaded data arrays and attributes.
            train (bool): A flag indicating whether the data is for training (True) or testing (False).

        Returns:
            None: This function does not return any value. It assigns the filtered data to the instance attributes.
        """
        vars = data['x_vars']
        lag_var_name = data['lag_var']
        if self.delta_temp_layer:
            self.delta_vars = data['x_delta_vars']
        else:
            self.delta_vars = None

        if lag_var_name in vars:
            # get the new position of the lagged variable
            self.lag_var_pos = data['lag_var_pos'][0]
            self.lag_var_source_pos = data['lag_var_source_pos'][0]
            self.lag_var_mean = data['lag_var_mean']
            self.lag_var_sd = data['lag_var_std']
            self.lag_days = self.cfg_bmi['lag_days']
        else:
            self.lag_var_mean = float('NaN')
            self.lag_var_sd = float('NaN')
            self.lag_var_pos = float('NaN')
            self.lag_var_source_pos = float('NaN')
            self.lag_days = float('NaN')

        self.x_trn = data['x_train']
        self.x_val = data['x_val']
        self.x_test = data['x_test']
        self.x_all = data['x_all_dates']
        self.x = data['x_all_dates']

        if self.delta_temp_layer:
            self.x_delta_trn = data['x_delta_train']
            self.x_delta_val = data['x_delta_val']
            self.x_delta_test = data['x_delta_test']
            self.x_delta_all = data['x_delta_all_dates']
            self.n_feat_delta = self.x_delta_trn.shape[2]
            self.x_delta_mean = data['x_delta_mean']
            self.x_delta_sd = data['x_delta_std']
        else:
            self.x_delta_trn = None
            self.x_delta_val = None
            self.x_delta_test = None
            self.x_delta_all = None
            self.n_feat_delta = None
            self.x_delta_mean = None
            self.x_delta_sd = None

        self.y_trn = data['pretrain_train']
        self.y_val = data['pretrain_val']
        self.y_test = data['pretrain_test']

        self.obs_trn = data['obs_train']
        self.obs_val = data['obs_val']
        self.obs_test = data['obs_test']
        self.obs_all = data['obs_all_dates']
        self.obs = data['obs_all_dates']

        self.x_vars = vars
        self.obs_vars = data['obs_vars']
        self.n_feat = self.x_trn.shape[2]

        self.dates_trn = data['times_train']
        self.dates_val = data['times_val']
        self.dates_test = data['times_test']
        self.dates_all = data['times_all_dates']

        self.dist_mat_trn = data['weighting_matrix_train']
        self.dist_mat_val = data['weighting_matrix_val']
        self.dist_mat_test = data['weighting_matrix_test']
        self.dist_mat_all = data['weighting_matrix_test']

        self.start_h_trn = data['h_train']
        self.start_c_trn = data['c_train']
        self.start_h_val = data['h_val']
        self.start_c_val = data['c_val']
        self.start_h_test = data['h_test']
        self.start_c_test = data['c_test']
        self.start_h_all_dates = data['h_test']
        self.start_c_all_dates = data['c_test']

        # select vars based on dictionary
        self.x_data_mean = data['x_mean']
        self.x_data_sd = data['x_std']
        self.obs_data_mean = data['obs_mean']
        self.obs_data_sd = data['obs_std']


    def calc_feature_importance(self):
        """
        Calculate feature importance based on the change in Negative Log-Likelihood (NLL).

        This method initializes the feature importance model and checks for the existence of the weights file.
        It then evaluates the model's performance on the original input data and calculates the NLL.
        For each feature variable, it generates a hypothesis by modifying the feature values, evaluates the model again,
        and computes the change in NLL (delta NLL). The results are stored in a Pandas DataFrame.

        The feature importance is determined by the impact of each feature on the model's performance,
        as measured by the change in NLL when the feature values are perturbed.

        Raises:
            FileNotFoundError: If the weights file specified by `self.weights_file` does not exist.

        Returns:
            None: The method stores the calculated feature importance in the instance variable `self.feat_importance`.

        Attributes:
            self.feat_importance_model: An instance of the LSTMWithHead model initialized with specified parameters.
            self.feat_importance: A Pandas DataFrame containing the feature names and their corresponding delta NLL values.
        """

        self.feat_importance_model = LSTMWithHead(self.n_feat,
                                                self.hidden_units,
                                                self.dist_mat_all,
                                                self.dropout_rate,
                                                self.recurrent_dropout_rate,
                                                self.head,
                                                self.head_hidden_dim,
                                                self.head_n_dist,
                                                self.n_feat_delta)

        # Check if the weights file exists
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"The weights file '{self.weights_file}' does not exist.")
        else:
            # load model
            self.feat_importance_model.load_state_dict(torch.load(self.weights_file, weights_only=True))

        self.x_feat_importance = torch.from_numpy(self.x).float()
        self.obs_feat_importance = torch.from_numpy(self.obs).float()
        if self.delta_temp_layer:
            self.x_delta_feat_importance = torch.from_numpy(self.x_delta_all).float()
        else:
            self.x_delta_feat_importance = None

        # need to give some initial h and c
        self.start_h = torch.zeros(self.n_segs, self.hidden_units)
        self.start_c = torch.zeros(self.n_segs, self.hidden_units)

        if self.mc_dropout:
            pred_orig, (self.h, self.c) = self.feat_importance_model.train()(self.x_feat_importance, [self.start_h, self.start_c], self.dist_mat_all, self.x_delta_feat_importance) # ensure that dropout layers are active w/ .train()
        else:
            pred_orig, (self.h, self.c) = self.feat_importance_model.eval()(self.x_feat_importance, [self.start_h, self.start_c], self.dist_mat_all, self.x_delta_feat_importance) # ensure that dropout layers are inactive w/ .eval()

        nll_orig = MaskedGMMLoss(self.obs_feat_importance, pred_orig)
        fi_data = {
            'x_var': [],
            'delta_nll': []
        }
        for var in range(len(self.x_vars)):
            x_hypothesis = self.x_feat_importance.detach().clone()
            ## Identify the 10th and 90th percentile of data distribution
            var_range = torch.quantile(x_hypothesis[:,:,var].flatten(),torch.tensor([.1,.9]))
            ## Make random distribution within the range of the target variable
            x_hypothesis[:, :, var] = (var_range[0]-var_range[1])*torch.rand_like(x_hypothesis[:, :, var])+var_range[1]
            if self.mc_dropout:
                y_hypothesis, (self.h, self.c) = self.feat_importance_model.train()(x_hypothesis, [self.start_h, self.start_c], self.dist_mat_all, self.x_delta_feat_importance) # ensure that dropout layers are active w/ .train()
            else:
                y_hypothesis, (self.h, self.c) = self.feat_importance_model.eval()(x_hypothesis, [self.start_h, self.start_c], self.dist_mat_all, self.x_delta_feat_importance) # ensure that dropout layers are inactive w/ .eval()

            nll_hypothesis = MaskedGMMLoss(self.obs_feat_importance, y_hypothesis)
            delta_nll = nll_hypothesis-nll_orig
            # Append the feature name and delta_nll to the DataFrame
            fi_data['x_var'].append(self.x_vars[var])
            fi_data['delta_nll'].append(delta_nll.item())
        if self.delta_temp_layer:
            for var in range(len(self.delta_vars)):
                x_hypothesis = self.x_delta_feat_importance.detach().clone()
                ## Identify the 10th and 90th percentile of data distribution
                var_range = torch.quantile(x_hypothesis[:,:,var].flatten(),torch.tensor([.1,.9]))
                ## Make random distribution within the range of the target variable
                x_hypothesis[:, :, var] = (var_range[0]-var_range[1])*torch.rand_like(x_hypothesis[:, :, var])+var_range[1]
                if self.mc_dropout:
                    y_hypothesis, (self.h, self.c) = self.feat_importance_model.train()(self.x_feat_importance, [self.start_h, self.start_c], self.dist_mat_all, x_hypothesis) # ensure that dropout layers are active w/ .train()
                else:
                    y_hypothesis, (self.h, self.c) = self.feat_importance_model.eval()(self.x_feat_importance, [self.start_h, self.start_c], self.dist_mat_all, x_hypothesis) # ensure that dropout layers are inactive w/ .eval()

                nll_hypothesis = MaskedGMMLoss(self.obs_feat_importance, y_hypothesis)
                delta_nll = nll_hypothesis-nll_orig
                # Append the feature name and delta_nll to the DataFrame
                fi_data['x_var'].append(self.delta_vars[var])
                fi_data['delta_nll'].append(delta_nll.item())

        self.feat_importance = pd.DataFrame(fi_data)


    def expected_gradients_lstm(self, n_samples=200, temporal_focus=None):
        """
        Calculate expected gradients for the LSTM model based on input sequences and
        return a DataFrame containing the gradients.

        This method initializes an LSTM model, loads its weights, and
        computes gradients of the model's output with respect to its input features.
        It samples from the input data, computes gradients, and
        aggregates them over a specified number of samples.

        Parameters:
        ----------
        n_samples : int, optional
            The number of samples to draw for gradient computation. Default is 200.

        temporal_focus : int, optional
            If specified, the index of the time step to focus on when calculating gradients.
            If None, gradients for all time steps are computed. Default is None.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the expected gradients for each feature, with dates corresponding to the input sequences.
            The DataFrame has the following columns:
            - 'date': The date corresponding to each observation.
            - Features specified in self.x_vars.

        Raises:
        ------
        FileNotFoundError
            If the weights file specified by self.weights_file does not exist.

        Notes:
        -----
        The function follows the methodology described in
        Erion et al. (2021) for calculating expected gradients.
        """
        self.eg_model = LSTMWithHead(self.n_feat,
                                    self.hidden_units,
                                    self.dist_mat_all,
                                    self.dropout_rate,
                                    self.recurrent_dropout_rate,
                                    self.head,
                                    self.head_hidden_dim,
                                    self.head_n_dist,
                                    self.n_feat_delta)
        # Check if the weights file exists
        if not os.path.exists(self.weights_file):
            raise FileNotFoundError(f"The weights file '{self.weights_file}' does not exist.")
        else:
            # load model
            self.eg_model.load_state_dict(torch.load(self.weights_file, weights_only=True))

        seq_length = 365
        x_eg = self.x
        x_eg = x_eg.squeeze(0)
        if self.delta_temp_layer:
            delta_eg = self.x_delta_all
            delta_eg = delta_eg.squeeze(0)

        # Calculate the number of complete sequences
        n_sequences = x_eg.shape[0] // seq_length

        # Truncate to keep only complete sequences
        x_truncated = x_eg[:n_sequences * seq_length]
        dates_truncated = self.dates_all[0,:n_sequences * seq_length,0]

        # Reshape into sequences
        x_sequences = x_truncated.reshape(n_sequences, seq_length, self.n_feat)
        dates_sequences = dates_truncated.reshape(n_sequences, seq_length)

        if self.delta_temp_layer:
            delta_truncated = delta_eg[:n_sequences * seq_length]
            delta_sequences = delta_truncated.reshape(n_sequences, seq_length, self.n_feat_delta)
            delta_sequences = torch.from_numpy(delta_sequences).float()
        else:
            delta_sequences = None

        x_sequences = torch.from_numpy(x_sequences).float()
        # need to give some initial h and c
        start_h = torch.zeros(n_sequences, self.hidden_units)
        start_c = torch.zeros(n_sequences, self.hidden_units)

        ## See Erion et al (2021) https://doi.org/10.1038/s42256-021-00343-w

        for k in range(n_samples):
            ## Sample a series from our data
            rand_seq = np.random.choice(n_sequences)
            baseline_x = x_sequences[rand_seq].to(torch.device('cpu'))
            if self.delta_temp_layer:
                baseline_delta = delta_sequences[rand_seq].to(torch.device('cpu'))

            ## Sample a random scale along the difference
            scale = np.random.uniform()

            ## Calculate the gradient of f(x) with regards to x
            x_diff = x_sequences - baseline_x
            curr_x = baseline_x + scale*x_diff
            if self.delta_temp_layer:
                delta_diff = delta_sequences - baseline_delta
                curr_delta = baseline_delta + scale*delta_diff
            else:
                curr_delta = None
            if curr_x.requires_grad == False:
                curr_x.requires_grad = True
                if self.delta_temp_layer:
                    curr_delta.requires_grad = True
            self.eg_model.zero_grad()
            y, (h, c) = self.eg_model(curr_x, [start_h, start_c], self.dist_mat_all, curr_delta)
            y_mu = y['mu']

            ## Pull out the gradient
            # if self.delta_temp_layer:
            #     curr_x = torch.cat((curr_x, curr_delta), dim=-1)
            #     x_diff = torch.cat((x_diff, delta_diff), dim=-1)
            if temporal_focus == None:
                gradients = torch.autograd.grad(y_mu[:, :, :], curr_x, torch.ones_like(y_mu[:, :, :]))
            else:
                gradients = torch.autograd.grad(y_mu[:, temporal_focus, :], curr_x, torch.ones_like(y_mu[:,temporal_focus, :]))

            if k == 0:
                expected_gradients = x_diff*gradients[0] * 1/n_samples
            else:
                expected_gradients = expected_gradients + ((x_diff*gradients[0]) * 1/n_samples)

        reshaped_output = expected_gradients.view(-1, expected_gradients.shape[-1])

        # Flatten the dates_sequences to match the reshaped output
        flattened_dates = dates_sequences.flatten()

        # if self.delta_temp_layer:
        #     out_vars = np.concatenate((self.x_vars, self.delta_vars))
        # else:
        out_vars = self.x_vars

        df = pd.DataFrame(reshaped_output.numpy(), columns=out_vars)  # Convert to DataFrame with feature names
        df['date'] = flattened_dates

        # Reorder columns to have 'date' as the first column
        df = df[['date'] + list(out_vars)]

        return(df)


    ### functions not implemented but needed for BMI class
    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_current_time(self):
        return self.t

    def get_current_date(self):
        return self.dates_all[0,int(self.t),0]

    def get_end_time(self):
        return self._end_time

    def get_grid_edge_count(self, grid):
        raise NotImplementedError("get_grid_edge_count")

    def get_grid_edge_nodes(self, grid):
        raise NotImplementedError("get_grid_edge_nodes")

    def get_grid_face_count(self, grid):
        raise NotImplementedError("get_grid_face_count")

    def get_grid_face_edges(self, grid):
        raise NotImplementedError("get_grid_face_edges")

    def get_grid_face_nodes(self, grid):
        raise NotImplementedError("get_grid_face_nodes")

    def get_grid_node_count(self, grid):
        raise NotImplementedError("get_grid_node_count")

    def get_grid_nodes_per_face(self, grid):
        raise NotImplementedError("get_grid_nodes_per_face")

    def get_grid_origin(self, grid):
        raise NotImplementedError("get_grid_origin")

    def get_grid_rank(self, grid):
        raise NotImplementedError("get_grid_rank")

    def get_grid_shape(self, grid):
        raise NotImplementedError("get_grid_shape")

    def get_grid_size(self, grid):
        raise NotImplementedError("get_grid_size")

    def get_grid_spacing(self, grid):
        raise NotImplementedError("get_grid_spacing")

    def get_grid_type(self, grid):
        raise NotImplementedError("get_grid_type")

    def get_grid_x(self, grid):
        raise NotImplementedError("get_grid_x")

    def get_grid_y(self, grid):
        raise NotImplementedError("get_grid_y")

    def get_grid_z(self, grid):
        raise NotImplementedError("get_grid_z")

    def get_input_item_count(self):
        """Get number of input variables."""
        return len(self._input_var_names)

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_item_count(self):
        """Get number of output variables."""
        return len(self._output_var_names)

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_start_time(self):
        return self._start_time

    def get_time_step(self):
        return self._time_step_size

    def get_time_units(self):
        return self._time_units

    def get_value_at_indices(self, var, indices):
        raise NotImplementedError("get_value_at_indices")

    def get_var_grid(self, var):
        raise NotImplementedError("get_var_grid")

    def get_var_itemsize(self, var):
        raise NotImplementedError("get_var_itemsize")

    #------------------------------------------------------------
    def get_var_location(self, name):
        # Note: all vars have location node but check if its in names list first
        if name in (self._output_var_names + self._input_var_names):
            return self._var_loc

    def get_var_nbytes(self, var):
        raise NotImplementedError("get_var_nbytes")

    #-------------------------------------------------------------------
    def get_var_type(self, long_var_name):
        """Get the data type of a variable specified by its long name.

        This function retrieves the data type of a variable specified by its long name.

        Parameters:
            long_var_name (str): The long variable name for which to retrieve the data type.

        Returns:
            str: The data type of the variable.

        """
        return self.get_value_ptr(long_var_name).dtype.name

    def set_value_at_indices(self, var, indices, value):
        raise NotImplementedError("set_value_at_indices")

    #------------------------------------------------------------
    #------------------------------------------------------------
    #-- Utility functions
    #------------------------------------------------------------
    #------------------------------------------------------------

    def _parse_config(self, cfg):
        """Parse configuration settings.

        This function parses configuration settings provided in a dictionary format. It converts path strings
        to `Path` objects, and converts date strings to pandas `DatetimeIndex` objects.

        Parameters:
            cfg (dict): A dictionary containing configuration settings.

        Returns:
            dict: A dictionary with parsed configuration settings.

        """
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            if (USE_PATH):
                                temp_list.append( Path(element) )
                            else:
                                temp_list.append( element )  # (SDP)
                        cfg[key] = temp_list
                    else:
                        if (USE_PATH):
                            cfg[key] = Path( val )
                        else:
                            cfg[key] = val  # (SDP)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%Y-%m-%d'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%Y-%m-%d')

            else:
                pass

        # Add more config parsing if necessary
        return cfg

