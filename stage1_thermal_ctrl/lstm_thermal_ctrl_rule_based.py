import pandas as pd
import pathnavigator
from tqdm import tqdm

if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
    
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.lstm_model import WaterTempLSTMModel
from src.policies import RuleBasedPolicy

disable = False  # Set to True to disable tqdm progress bar
folder = "RFModels_deeptree"
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']

ml_model = WaterTempLSTMModel(
    model1=r"C:\Users\CL\Documents\GitHub\PywrDRB-ML\models\TempLSTM1_comparison\TempLSTM1_Qc.yml",
    model2=r"C:\Users\CL\Documents\GitHub\PywrDRB-ML\models\TempLSTM2_comparison\TempLSTM2_Qc.yml",
    model_map=pn.get() / "models/RFModels/rf_model_map.gz",
    disable_tqdm=False,
    debug=True,
    thermal_mitigation_bank_size=1620,  # mgd
    )
ml_model.load_data(database)

ml_model.update(t=ml_model.t)

#%%
def return_dps_func(*params):
    # Initialize the thermal control policy with specific parameters
    policy = RuleBasedPolicy(threshold=24, thermal_release_amount=300)
    
    # Define the function that will be used for the control algorithm
    def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        # Retrieve the ml_model from the model
        ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
        
        # Update until the current date if it is within the control period.
        # The temp simulation of the current date will not be implemented until the next update.
        # Namely, update until the beginning of the current date.
        if current_date.month in [6, 7, 8]: # Control period
            # Reset the bank amount at the beginning of June
            if current_date.day == 1 and current_date.month == 6:
                ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
            ml_model.update_until(date=current_date)
            # Complete the preparation for thermal control
        else:
            # Record
            ml_model.thermal_release = 0
            ml_model.records["thermal_releases"][ml_model.t] = 0 
            ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount 
            return 0
        
        # Make a forecast for the current date
        ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0)

        # Make thermal release decision and record the thermal release
        thermal_release = policy.run(X=ml_model.forecast_T_L_arr)
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        
        # Record
        ml_model.thermal_release = thermal_release
        ml_model.remained_bank_amount -= thermal_release
        ml_model.records["thermal_releases"][ml_model.t] = thermal_release 
        ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount 

        return thermal_release
    
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

#%% Run the simulation
dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for date in tqdm(dates, desc="Running thermal control policy", disable=disable):
    Q_C = None  # Placeholder for controlled release
    Q_i = None  # Placeholder for inflow
    cannonsville_storage_pct = None  # Placeholder for storage percentage
    thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
    
    # Update data in the ml_model for the next step(s) model update.
    t = ml_model.t
    ml_model.Q_C[t] += thermal_release
    try:
        ml_model.X_1[t, ml_model.rf_model1.x_vars.index("QbcTavg_Q_C")] += thermal_release
    except ValueError:
        pass
    try:
        ml_model.X_2[t, ml_model.rf_model2.x_vars.index("QbcTavg_Q_C")] = Q_C
    except ValueError:
        pass
# Update the model until the end of the simulation period
ml_model.update_until(date="2024-01-01")

#%% Calculate objectives
df = pd.DataFrame(ml_model.records, index=dates)
df.to_csv(pn.get() / f"models/{folder}/rule_based.csv")

#%%
total_nodes = sum(tree.tree_.node_count for tree in ml_model.rf_model1.rf_final.estimators_)
print(f"Total number of nodes in the forest: {total_nodes}")

