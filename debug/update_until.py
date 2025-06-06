import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()

from src.torch_bmi import bmi_lstm

config_file = pn.models.get(f"SalinityLSTM.yml")
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())

end_time_step = 200
unscaled_data = lstm.get_unscaled_values(lead_time=end_time_step) 
for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data[var])
if lstm.delta_temp_layer:
    for var in lstm.delta_vars:
        lstm.set_value(var, unscaled_data[var])

lstm.update_until_org(end_time_step)

# lstm.update_until(100)
