import pandas as pd
import numpy as np
import pathnavigator
root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-LSTMs"
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("database")


def create_TempLSTM_database(pn, start="1979-01-01", end="2024-12-31", filename="TempLSTM_database.csv"):
    # Load gridmet
    df_gridmet = pd.read_csv(pn.data.raw.get("gridmet_lordville.csv"), parse_dates=True, index_col="date")[start:end]
    
    # LSTM1 and LSTM2
    df_QobsTmax = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QobsTmax.csv"), parse_dates=True, index_col="date")[start:end]
    df_QobsTavg = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QobsTavg.csv"), parse_dates=True, index_col="date")[start:end]
    df_QbcTmax = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QbcTmax.csv"), parse_dates=True, index_col="date")[start:end]
    df_QbcTavg = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QbcTavg.csv"), parse_dates=True, index_col="date")[start:end]
    df_QobsTmax.columns = [f"QobsTmax_{i}" for i in df_QobsTmax.columns]
    df_QobsTavg.columns = [f"QobsTavg_{i}" for i in df_QobsTavg.columns]
    df_QbcTmax.columns = [f"QbcTmax_{i}" for i in df_QbcTmax.columns]
    df_QbcTavg.columns = [f"QbcTavg_{i}" for i in df_QbcTavg.columns]
    
    # Load the source
    df_lordville = pd.read_csv(pn.data.raw.get("nwis_Lordville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
    df_lordville = df_lordville[[i for i in df_lordville.columns if "_src" in i]]
    df_cannonsville = pd.read_csv(pn.data.raw.get("nwis_Cannonsville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
    df_cannonsville = df_cannonsville[[i for i in df_cannonsville.columns if "_src" in i]]
    
    df_lordville_cannonsville = df_lordville.copy()
    df_lordville_cannonsville.columns = [i.replace("_lordville", "") for i in df_lordville_cannonsville.columns]
    df_cannonsville_src = df_cannonsville.copy()
    df_cannonsville_src.columns = [i.replace("_cannonsville", "") for i in df_cannonsville_src.columns]
    df_lordville_cannonsville[df_cannonsville_src != "obs"] = "linear_interpolation"
    
    df_all = pd.concat(
        [df_gridmet, df_QobsTmax, df_QobsTavg, df_QbcTmax, df_QbcTavg, df_lordville, 
         df_cannonsville, df_lordville_cannonsville], 
        axis=1
        )
    
    # Load reservoir storage
    df_storage = pd.read_csv(pn.data.raw.get("drb_reservoir_storage_mg_2000_2024.csv"), parse_dates=True, index_col="date")[start:end]
    df_all["nyc_storage_pct"] = df_storage["nyc_total_pct"]
    df_all["cannonsville_storage_pct"] = df_storage["cannonsville_pct"]
    df_all["pepacton_storage_pct"] = df_storage["pepacton_pct"]
    df_all["neversink_storage_pct"] = df_storage["neversink_pct"]
    
    # Load simulated storage
    df_bc = pd.read_csv(pn.data.raw.get("pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv"), parse_dates=True, index_col=["date"])[start:end]
    df_all["bc_cannonsville_storage_pct"] = df_bc["cannonsville_storage_pct"]
    df_all["bc_pepacton_storage_pct"] = df_bc["pepacton_storage_pct"]
    
    df_all['doy_cos'] = np.cos(
        np.pi + 2 * np.pi * df_all.index.dayofyear /
        np.where(df_all.index.is_leap_year, 366, 365)  # Fix: use np.where instead of apply
    )
    df_all['doy'] = df_all.index.dayofyear
    
    # Load thermal release
    df_res = pd.read_excel(pn.data.get("reservoir_release_sep_thermal_ctrl_mgd_manually_picked.xlsx"), parse_dates=True, index_col="date")[start:end]
    df_all["rel_thermal"] = df_res["thermal_release"]
    
    df_all.replace("obv", "obs", inplace=True) # in case of "obv" in the data
    df_all.to_csv(pn.data.database.get() / filename)
    
    # Form bmi attributes
    input_vars = list(df_all)
    input_vars += [i+"_lag_1" for i in input_vars if "T_C" in i]  
    input_vars += [i+"_lag_1" for i in input_vars if "T_L" in i]  
    input_vars += [i+"_lag_1" for i in input_vars if "T_i" in i] 
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

df_all, input_vars, unit_map = create_TempLSTM_database(pn=pn, start="1979-01-01", end="2024-12-31")
