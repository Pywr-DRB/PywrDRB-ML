import pandas as pd
import numpy as np
import pathnavigator
if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-LSTMs"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-LSTMs")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("database")


def create_SalinityLSTM_database(pn, start="1963-10-01", end="2024-12-31", filename="SalinityLSTM_database.csv"):
    # Load saltfront and flow
    # 01463500 = delTrenton
    # 01474500 = outletSchuylkill
    df_salinity_flow_bc = pd.read_csv(pn.data.raw.get() / "pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])[start:end][["01463500", "01474500"]]
    df_salinity_flow_bc.columns = ["01463500_bc", "01474500_bc"]
    df_salinity_obs = pd.read_csv(pn.data.raw.get("salt_front_data.csv"), parse_dates=True, index_col=["date"])[start:end]
    
    df_all = pd.concat(
        [df_salinity_obs, df_salinity_flow_bc], 
        axis=1
        )
    
    df_all['doy_cos'] = np.cos(
        np.pi + 2 * np.pi * df_all.index.dayofyear /
        np.where(df_all.index.is_leap_year, 366, 365)  # Fix: use np.where instead of apply
    )
    df_all['doy'] = df_all.index.dayofyear
    
    # Required by Jake's LSTM prep
    df_all["seg_id_nat"] = 1573
    
    df_all.replace("obv", "obs", inplace=True) # in case of "obv" in the data
    df_all.to_csv(pn.data.database.get() / filename)
    
    # Form bmi attributes
    input_vars = list(df_all)
    input_vars += ['y_src_lag_1']
    
    unit_map = {}
    for v in input_vars:
        if "rel_" in v and "_pct" not in v:
            unit_map[v] = [v, "mgd"]
        elif ("tmmx" in v or "tmmn" in v or "tavg" in v) and "_src" not in v:
            unit_map[v] = [v, "degC"]
        elif "pr" in v:
            unit_map[v] = [v, "m day-1"]
        elif "rmax" in v or "rmin" in v:
            unit_map[v] = [v, "fraction"]
        elif "srad" in v:
            unit_map[v] = [v, "W m-2"]
        elif "vs" in v:
            unit_map[v] = [v, "m s-1"]
        elif "sph" in v:
            unit_map[v] = [v, "kg/kg"]
        elif "vs" in v:
            unit_map[v] = [v, "m s-1"]
        elif "_pct" in v:
            unit_map[v] = [v, "%"]
        else:
            unit_map[v] = [v, "--"]
    
    return df_all, input_vars, unit_map

df_all, input_vars, unit_map = create_SalinityLSTM_database(pn=pn, start="1963-10-01", end="2024-12-31")
