import pathnavigator
import numpy as np
from copy import deepcopy

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.torch_bmi import bmi_lstm

config_file = pn.models.get(f"TempLSTM1.yml")
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())


lstm.t

lstm.update()
T_C_mu = np.zeros(1)
T_C_sd = np.zeros(1)
lstm.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
#lstm.get_value("channel_water_surface_water__sd_max_of_temperature", T_C_sd)



end_time_step = 200
unscaled_data = lstm.get_unscaled_values(lead_time=end_time_step) 




for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data[var])
if lstm.delta_temp_layer:
    for var in lstm.delta_vars:
        lstm.set_value(var, unscaled_data[var])

lstm.update_until(end_time_step)



lstm.t


# lstm.update_until_loop(100)
