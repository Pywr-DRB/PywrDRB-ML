# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:19:22 2026

@author: CL
"""

import numpy as np
import pandas as pd
import pathnavigator
import pywrdrb

import warnings
# Suppress the NumPy deprecation warning about array to scalar conversion
warnings.filterwarnings('ignore', message='.*Conversion of an array with ndim > 0 to a scalar is deprecated.*', category=DeprecationWarning)

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()

# Output folder for simulation results
output_folder = "coupled_pywrdrb_RBF_sols2"
pn.outputs.mkdir(output_folder)

total_volume_mg_dict = {}
shortage_days_dict = {}
df_demand = pd.DataFrame()
df_delivery = pd.DataFrame()
#%% 

#Simulation with control
name = "coupled_pywrdrb_pub_nhmv10_BC_withObsScaled_with_ctrl"
model_filename = str(pn.models.get(name) / f"{name}.json")

policy = "GaussianRBFPolicy"
job_id = "143990"
names = ["Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", 
         "Enlarged bank"]
for i, sol_idx in enumerate([28, 151, 114]):
    df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_demand_nyc_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)
    df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_delivery_nyc_RBF{i+1}_{sol_idx}.csv", index_col=[0], parse_dates=True)
    df_demand[names[i]] = df_demand_nyc["demand_nyc"]        # MGD
    df_delivery[names[i]] = df_delivery_nyc["delivery_nyc"]  # MGD
 
# No control
df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_demand_nyc_no_ctrl.csv", index_col=[0], parse_dates=True)
df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_delivery_nyc_no_ctrl.csv", index_col=[0], parse_dates=True)
df_demand["No control"] = df_demand_nyc["demand_nyc"]        # MGD
df_delivery["No control"] = df_delivery_nyc["delivery_nyc"]  # MGD

# Rule-based
df_demand_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_demand_nyc_rulebased.csv", index_col=[0], parse_dates=True)
df_delivery_nyc = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_delivery_nyc_rulebased.csv", index_col=[0], parse_dates=True)
df_demand["Fixed-release\n(baseline)"] = df_demand_nyc["demand_nyc"]        # MGD
df_delivery["Fixed-release\n(baseline)"] = df_delivery_nyc["delivery_nyc"]  # MGD


#%%
df_shortage = df_demand - df_delivery
df_shortage[df_shortage<1] = 0

min_duration = 3
def filter_short_events(series):
    positive = series > 0
    event_groups = positive.ne(positive.shift()).cumsum()
    event_lengths = positive.groupby(event_groups).transform("sum")
    return series.mask(positive & (event_lengths < min_duration), 0.0)

if min_duration > 0:
    df_shortage = df_shortage.apply(filter_short_events)

df_shortage.plot(ylabel="Shortage Volumn (MG)")

total_volume_mg = df_shortage.sum()        # total shortage
shortage_days = (df_shortage > 0).sum()    # total shortage days

total_volume_mg.plot(ylabel="Shortage Volumn (MG)")
shortage_days.plot(ylabel="Shortage Days")
#%%
 
# Daily NYC diversion shortage series (MGD)
shortage = calculate_shortage_series(
    demand, delivery,
    tolerance=1.0,     # MGD; deficits below this are set to zero
    min_duration=3,    # drop shortage runs shorter than 3 days (0 to turn this off)
    warmup_days=3,     # zero the first 3 days of the simulation (0 to turn this off)
)
 
total_volume_mg = shortage.sum()        # total shortage
shortage_days = (shortage > 0).sum()    # total shortage days

(demand-delivery).plot()
