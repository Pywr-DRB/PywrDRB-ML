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


def create_SalinityLSTM_database(pn, start="1963-10-01", end="2024-12-31", filename="SalinityLSTM_database.csv"):
    # Load saltfront and flow
    # 01463500 = delTrenton
    # 01474500 = outletSchuylkill
    df_salinity_flow_bc = pd.read_csv(pn.data.raw.get() / "pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])[start:end][["flow_delTrenton", "flow_outletSchuylkill"]]
    df_salinity_flow_bc.columns = ["Q_Trenton_bc", "Q_Schuylkill_bc"]
    df_salinity_flow_bc["Q_Trenton_bc_7d_avg"] = df_salinity_flow_bc["Q_Trenton_bc"].rolling(window=7, min_periods=1).mean()
    df_salinity_flow_bc["Q_Schuylkill_bc_7d_avg"] = df_salinity_flow_bc["Q_Schuylkill_bc"].rolling(window=7, min_periods=1).mean()


    df_salinity_obs = pd.read_csv(pn.data.raw.get("salt_front_data.csv"), parse_dates=True, index_col=["date"])[start:end]

    # Fill gaps by lstm simed data
    lstm_simed_saltfront = pd.read_csv(pn.data.raw.get("lstm_simed_saltfront.csv"), parse_dates=True, index_col=["date"])[start:end]
    lstm_simed_saltfront = lstm_simed_saltfront.reindex(df_salinity_obs.index)
    mask = (~lstm_simed_saltfront["saltfront"].isna()) & (df_salinity_obs["saltfront_src"] == "other") & (df_salinity_obs["saltfront"].isna())
    df_salinity_obs.loc[mask, "saltfront"] = lstm_simed_saltfront.loc[mask, "saltfront"]
    df_salinity_obs.loc[mask, "saltfront_src"] = "lstm"

    # ignore saltfront below 54 rm
    mask = df_salinity_obs["saltfront"] < 54
    df_salinity_obs.loc[mask, "saltfront_src"] = df_salinity_obs.loc[mask, "saltfront_src"] + "_(<54)"

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
    df_all["seg_id_nat"] = 1573 # Lordville

    # Manually add lag1 y as an input variable (for RF model)
    lag1_vars = ["saltfront"]
    for var in lag1_vars:
        df_all[f"{var}_lag1"] = df_all[var].shift(1)

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
    input_vars += ['y_src_lag_1', 'saltfront_lag_1'] # For LSTM

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
