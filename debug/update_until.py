import pathnavigator
import numpy as np
import pandas as pd
from copy import deepcopy

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()

from src.torch_bmi import bmi_lstm

config_file = pn.models.get(f"TempLSTM1.yml")
config_file = r"C:\Users\CL\Documents\GitHub\PywrDRB-ML\models\TempLSTM1_comparison\TempLSTM1_none.yml"
# lstm = bmi_lstm()
# lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())

# lstm._start_time
# lstm._end_time
# x = lstm.x
# dates_all = lstm.dates_all



#%%
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
length = 10000
unscaled_data_arr = lstm.get_unscaled_values(lead_time=length-1)

lstm.mc_dropout = False
for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data_arr[var])

T_C_mu_arr, T_C_sd_arr = lstm.update()


lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
length = 10000
unscaled_data_arr = lstm.get_unscaled_values(lead_time=length-1)
lstm.mc_dropout = True
for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data_arr[var])

T_C_mu_arr_, T_C_sd_arr_ = lstm.update()

#%% Test for startdate
from datetime import datetime
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")

dt = np.datetime64("1980-01-01")
idx = np.where(lstm.dates_all == dt)[1].item()
length = idx - int(lstm.get_current_time())
unscaled_data_arr = lstm.get_unscaled_values(lead_time=length-1)
lstm.mc_dropout = False
for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data_arr[var])
lstm.update()
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")

#%%
lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
lstm.mc_dropout = False

length = 600
T_C_mu_loop, T_C_sd_loop = np.zeros(length), np.zeros(length)
unscaled_data_loop = pd.DataFrame()
for i in range(length):
    unscaled_data = lstm.get_unscaled_values(lead_time=0)
    unscaled_data_loop = pd.concat([unscaled_data_loop, unscaled_data])
    for var in lstm.x_vars:
        lstm.set_value(var, unscaled_data[var])

    T_C_mu, T_C_sd = lstm.update()
    T_C_mu_loop[i], T_C_sd_loop[i] = T_C_mu, T_C_sd


lstm = bmi_lstm()
lstm.initialize(config_file=config_file, train=False, root_dir=pn.get())
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
lstm.mc_dropout = True

length = 600
T_C_mu_loop_, T_C_sd_loop_ = np.zeros(length), np.zeros(length)
unscaled_data_loop = pd.DataFrame()
for i in range(length):
    unscaled_data = lstm.get_unscaled_values(lead_time=0)
    unscaled_data_loop = pd.concat([unscaled_data_loop, unscaled_data])
    for var in lstm.x_vars:
        lstm.set_value(var, unscaled_data[var])

    T_C_mu, T_C_sd = lstm.update()
    T_C_mu_loop_[i], T_C_sd_loop_[i] = T_C_mu, T_C_sd

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(T_C_mu_arr, label='T_C_mu_arr', lw=1)
#ax.plot(T_C_mu_loop, label='T_C_mu_loop')
ax.plot(T_C_mu_arr_, label='T_C_mu_arr_withMC', lw=1, alpha = 0.5)
#ax.plot(T_C_mu_loop_, label='T_C_mu_loop_withMC')
ax.legend()
plt.show()

#%%
fig, ax = plt.subplots()
#ax.plot(T_C_mu_arr-T_C_mu_loop, label='diff')
ax.plot(T_C_mu_arr-T_C_mu_arr_, label='diff_arr (noMC - withMC)')
#ax.plot(T_C_mu_loop-T_C_mu_loop_, label='diff_loop (noMC - withMC)')
ax.legend()
plt.show()

r"""
Why Results Differ:
LSTM State Evolution:

Array form: LSTM processes a sequence of 5 time steps in one forward pass, with hidden states evolving step-by-step through the sequence
Loop form: LSTM processes 1 time step at a time, with hidden states updated and carried forward between separate update() calls
Input Tensor Shape:

Array form: Input tensor shape is [batch, 5, features] - the LSTM sees the temporal sequence
Loop form: Input tensor shape is [batch, 1, features] each time - the LSTM only sees one time step per call
Hidden State Continuity:

Array form: Hidden states flow naturally through the 5-step sequence within a single LSTM forward pass
Loop form: Hidden states are manually carried between separate LSTM calls, which may have subtle differences in how gradients/computations are handled
Potential Dropout Differences:

If the model uses dropout, each forward pass may apply different dropout masks
Array form: One set of dropout masks for the entire sequence
Loop form: Different dropout masks for each individual update call
The Key Issue:
Even with the same random seed, the computational path is different:

Array form: One LSTM forward pass with a 5-step sequence
Loop form: Five separate LSTM forward passes with 1-step sequences
LSTMs are inherently sensitive to how sequences are processed, and these two approaches create different computational graphs even with identical input data and random seeds.

For consistent results, you should use the same processing approach throughout your analysis.
"""
#%%
print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")
unscaled_data = lstm.get_unscaled_values(lead_time=16435)

unscaled_data = lstm.get_unscaled_values()#lead_time=1)

rng = pd.date_range('1979-01-01', '2023-12-31')

# for var in lstm.x_vars:
#     lstm.set_value(var, unscaled_data.loc[1, var])
# lstm.update()


# T_C_mu = np.zeros(1)
# T_C_sd = np.zeros(1)
# lstm.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
#lstm.get_value("channel_water_surface_water__sd_max_of_temperature", T_C_sd)

#%%

#%%
T_C_mu = np.zeros(length)
lstm.get_value("channel_water_surface_water__mu_max_of_temperature", T_C_mu)
#%%

end_time_step = 1
unscaled_data = lstm.get_unscaled_values(lead_time=end_time_step)

for var in lstm.x_vars:
    lstm.set_value(var, unscaled_data[var])

lstm.update_until(end_time_step)



print(f"{int(lstm.get_current_time())}: {lstm.get_current_date()}")


# lstm.update_until_loop(100)
