#%%
import pandas as pd
import numpy as np
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("database")


def create_TempLSTM_database(pn, start="1945-01-01", end="2024-12-31", filename="TempLSTM_database_1945.csv"):
    # Load gridmet
    df_gridmet = pd.read_csv(pn.data.raw.get("gridmet_lordville.csv"), parse_dates=True, index_col="date")[start:end]
    df_livneh = pd.read_csv(pn.data.raw.get("livneh_lordville.csv"), parse_dates=True, index_col="date")[start:end]
    df_climate = pd.concat([df_livneh[:"1979-01-01"], df_gridmet[df_livneh.columns]], axis=0)

    # LSTM1 and LSTM2
    df_bc = pd.read_csv(pn.data.raw.get("pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage_1945.csv"), parse_dates=True, index_col=["date"])[start:end]
    df_bc_Q = pd.DataFrame(index=df_bc.index)
    df_bc_Q["bc_Q_C"] = df_bc["flow_01425000"]
    df_bc_Q["bc_Q_i"] = df_bc["flow_01417000"] + df_bc["inflow_delLordville"] - df_bc["consumption_delLordville"]
    # There are minor (<0.2) differences between Q_C + Q_i and df_bc["flow_lordville"] likely due to the numerical issue in pywr simulation.
    # For LSTM training purpose, we directly calculate Q_L by Q_C + Q_i to avoid water inbalance.
    df_bc_Q["bc_Q_L"] = df_bc_Q["Q_C"] + df_bc_Q["Q_i"] #df_bc["flow_lordville"]
    df_bc_Q["bc_cannonsville_storage_pct"] = df_bc["cannonsville_storage_pct"]

    df_QbcTmax = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QbcTmax.csv"), parse_dates=True, index_col="date")[start:end]
    df_QbcTavg = pd.read_csv(pn.data.decomposed_Ti_Qi.get("df_QbcTavg.csv"), parse_dates=True, index_col="date")[start:end]
    df_QbcTmax.columns = [f"QbcTmax_{i}" for i in df_QbcTmax.columns]
    df_QbcTavg.columns = [f"QbcTavg_{i}" for i in df_QbcTavg.columns]

    df_all = pd.concat(
        [df_climate, df_QbcTmax, df_QbcTavg],
        axis=1
        )

    df_all['doy_cos'] = np.cos(
        np.pi + 2 * np.pi * df_all.index.dayofyear /
        np.where(df_all.index.is_leap_year, 366, 365)  # Fix: use np.where instead of apply
    )
    df_all['doy'] = df_all.index.dayofyear

    # Load thermal release
    df_res = pd.read_excel(pn.data.get("reservoir_release_sep_thermal_ctrl_mgd_manually_picked.xlsx"), parse_dates=True, index_col="date")[start:end]
    df_all["rel_thermal"] = df_res["thermal_release"]
    df_all["rel_thermal_org"] = df_res["thermal_release_org"]

    # Required by Jake's LSTM prep
    df_all["seg_id_nat"] = 1573

    df_all.replace("obv", "obs", inplace=True) # in case of "obv" in the data

    # Create doc (day of cooling season) column: 6/1 = 1, 6/2 = 2, ..., 8/31 = 92, rest = NaN
    df_all["doc"] = np.nan
    cooling_season_mask = (df_all.index.month >= 6) & (df_all.index.month <= 8)
    # Compute day-of-year for June 1st for each year
    june_1_doy_series = pd.to_datetime(
        df_all.index.year.astype(str) + '-06-01'
    ).dayofyear.values
    # Align June 1st DOY with index
    june_1_doy = pd.Series(june_1_doy_series, index=df_all.index)
    # Calculate day of cooling season (starting at 1 on June 1st)
    df_all.loc[cooling_season_mask, "doc"] = (
        df_all.index.dayofyear[cooling_season_mask] - june_1_doy[cooling_season_mask] + 1
    )

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

df_all, input_vars, unit_map = create_TempLSTM_database(pn=pn, start="1945-01-01", end="2024-12-31")
