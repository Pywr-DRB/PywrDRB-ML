import pandas as pd
import numpy as np
import pathnavigator
import clt
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("decomposed_Ti_Qi")

#mode = "LinearInterpolated" # Simply use linear interpolation to fill gaps
#mode = "Dwallin"            # Use dwallin to fill gaps mixed with linear interpolation
mode = "TempLSTMGapFiller"  # Use trained lstm to fill all gaps

start = "1978-01-01" #"1945-01-01"
end = "2024-12-31"
rng = pd.date_range(start, end, freq="D")

df_lordville = pd.read_csv(pn.data.raw.get("nwis_Lordville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
df_cannonsville = pd.read_csv(pn.data.raw.get("nwis_Cannonsville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
df_bc = pd.read_csv(pn.data.raw.get("pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv"), parse_dates=True, index_col=["date"])[start:end]
df_res = pd.read_excel(pn.data.get("reservoir_release_sep_thermal_ctrl_mgd_manually_picked.xlsx"), parse_dates=True, index_col="date")[start:end]
df_res = clt.utils.re_datetime_index(df_res, start, end, fill_value=0)
if mode=="Dwallin":
    df_dwallin_pred = clt.io.read_parquet(pn.data.raw.get("dwallin_stream_preds.parquet"))
    df_dwallin_pred['date'] = pd.to_datetime(df_dwallin_pred['date'])
    df_dwallin_pred.set_index('date', inplace=True)
    df_dwallin_pred = df_dwallin_pred.pivot(columns='seg_id_nat', values='dwallin_temp_c')
    df_dwallin_pred = df_dwallin_pred[[1566, 1573]]
    df_dwallin_pred.columns = ["tavg_water_cannonsville", "tavg_water_lordville"]

if mode=="TempLSTMGapFiller":
    df_lstm_simed = pd.read_csv(pn.data.raw.get() / "lstm_simed_T_degC.csv", parse_dates=True, index_col=["date"])


# Fill gaps with dwallin
# set tavg_water_lordville values to NaN where more than 3 consecutive days are flagged as "linear_interpolation" in tavg_water_lordville_src
def remove_long_linear_interpolations(df, value_col, src_col, thres=3, cutoff_date="2020-04-22"):
    # to start as dwallin
    new_index = pd.date_range(start="1982-04-01", end=df.index.max(), freq="D")
    df = df.reindex(new_index)

    # Only apply to dates before cutoff
    mask_date = df.index < pd.to_datetime(cutoff_date)
    is_interp = (df[src_col] == "linear_interpolation") & mask_date

    # Group consecutive True values
    group = (is_interp != is_interp.shift()).cumsum()

    # Count group sizes only where interpolation is True
    interp_group_sizes = is_interp.groupby(group).transform("sum")

    # Mask values where linear interpolation runs longer than threshold
    mask_to_nan = (is_interp) & (interp_group_sizes > thres)
    df.loc[mask_to_nan, value_col] = np.nan

    # Replace 'linear_interpolation' with 'dwallin' only before cutoff and not masked
    df.loc[is_interp & mask_to_nan, src_col] = "dwallin"
    df[src_col] = df[src_col].fillna("dwallin")
    for c in df:
        if "_src" in c:
            df[c] = df[c].fillna("no_value")
    return df
if mode=="Dwallin":
    df_lordville = remove_long_linear_interpolations(df=df_lordville, value_col="tavg_water_lordville", src_col="tavg_water_lordville_src", thres=3)
    df_cannonsville = remove_long_linear_interpolations(df=df_cannonsville, value_col="tavg_water_cannonsville", src_col="tavg_water_cannonsville_src", thres=3)
    df_lordville.loc[df_lordville["tavg_water_lordville_src"]=="dwallin", "tavg_water_lordville"] = df_dwallin_pred.loc[df_lordville["tavg_water_lordville_src"]=="dwallin", "tavg_water_lordville"]
    df_cannonsville.loc[df_cannonsville["tavg_water_cannonsville_src"]=="dwallin", "tavg_water_cannonsville"] = df_dwallin_pred.loc[df_cannonsville["tavg_water_cannonsville_src"]=="dwallin", "tavg_water_cannonsville"]

# Fill gaps with lstm
if mode=="TempLSTMGapFiller":
    new_index = pd.date_range(start=df_lstm_simed.index.min(), end=df_lordville.index.max(), freq="D")
    df_lordville = df_lordville.reindex(new_index)
    df_lordville.loc[df_lordville["tavg_water_lordville_src"]!="obs", "tavg_water_lordville"] = df_lstm_simed.loc[df_lordville["tavg_water_lordville_src"]!="obs", "Tavg"]
    df_lordville.loc[df_lordville["tavg_water_lordville_src"]!="obs", "tavg_water_lordville_src"] = "lstm"
    df_lordville.loc[df_lordville["tmmx_water_lordville_src"]!="obs", "tmmx_water_lordville"] = df_lstm_simed.loc[df_lordville["tmmx_water_lordville_src"]!="obs", "T_L"]
    df_lordville.loc[df_lordville["tmmx_water_lordville_src"]!="obs", "tmmx_water_lordville_src"] = "lstm"

    new_index = pd.date_range(start=df_lstm_simed.index.min(), end=df_cannonsville.index.max(), freq="D")
    df_cannonsville = df_cannonsville.reindex(new_index)
    df_cannonsville.loc[df_cannonsville["tavg_water_cannonsville_src"]!="obs", "tavg_water_cannonsville"] = df_lstm_simed.loc[df_cannonsville["tavg_water_cannonsville_src"]!="obs", "T_C"]
    df_cannonsville.loc[df_cannonsville["tavg_water_cannonsville_src"]!="obs", "tavg_water_cannonsville_src"] = "lstm"

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(df_dwallin_pred["Tavg_canonsville"])
# ax.plot(df_cannonsville["tavg_water_cannonsville"])
# plt.show()

df_Qobs = pd.DataFrame(index=rng)
df_Qobs["Q_C"] = df_cannonsville["discharge_cannonsville"]
df_Qobs["Q_L"] = df_lordville["discharge_lordville"]

df_bc_Q = pd.DataFrame(index=rng)
df_bc_Q["Q_C"] = df_bc["flow_01425000"]
df_bc_Q["Q_i"] = df_bc["flow_01417000"] + df_bc["inflow_delLordville"] - df_bc["consumption_delLordville"]
# There are minor (<0.2) differences between Q_C + Q_i and df_bc["flow_lordville"] likely due to the numerical issue in pywr simulation.
# For LSTM training purpose, we directly calculate Q_L by Q_C + Q_i to avoid water inbalance.
df_bc_Q["Q_L"] = df_bc_Q["Q_C"] + df_bc_Q["Q_i"] #df_bc["flow_lordville"]

df_Tmax = pd.DataFrame(index=rng)
df_Tmax["T_C"] = df_cannonsville["tmmx_water_cannonsville"]
df_Tmax["T_L"] = df_lordville["tmmx_water_lordville"]

df_Tavg = pd.DataFrame(index=rng)
df_Tavg["T_C"] = df_cannonsville["tavg_water_cannonsville"] # 01425000
df_Tavg["T_L"] = df_lordville["tavg_water_lordville"]

def infer_Qi_Ti(df_Q, df_T):
    """
    Infer Q_i and T_i from Q_C, Q_L, T_C, T_L
    Q_i = Q_L - Q_C
    T_i = (T_L*Q_L - T_C*Q_C)/Q_i
    """
    df_Q = df_Q.copy()
    df_T = df_T.copy()

    # Infer Q_i
    if "Q_i" not in df_Q:
        df_Q["Q_i"] = df_Q["Q_L"] - df_Q["Q_C"]
    else:
        print("Q_i is already given.")
    df_Q = df_Q[['Q_i', 'Q_C', 'Q_L']]

    # Infer T_i
    df_T["T_i"] = (df_T["T_L"]*df_Q["Q_L"] - df_T["T_C"]*df_Q["Q_C"])/df_Q["Q_i"]
    df_T = df_T[['T_i', 'T_C', 'T_L']]

    df = pd.concat([df_Q, df_T], axis=1)
    df.index.name = "date"
    return df

def update_to_rm_thermal_release_effect(df_sim, df_obv, df_res):
    # Q_Ctr T_itr (no thermal)
    df_obv["Q_Cnotr"] = df_obv["Q_C"] - df_res["thermal_release"]
    df_obv["Q_Lnotr"] = df_obv["Q_L"] - df_res["thermal_release"]
    df_obv["T_Lnotr"] = (df_obv["T_C"]*df_obv["Q_Cnotr"] + df_obv["T_i"]*df_obv["Q_i"])/df_obv["Q_Lnotr"]

    mask = df_res[df_res["thermal_release"] > 0].index
    df_sim.loc[mask, "T_L"] = df_obv.loc[mask, "T_Lnotr"]
    df_sim["T_i"] = (df_sim["T_L"]*df_sim["Q_L"] - df_sim["T_C"]*df_sim["Q_C"])/df_sim["Q_i"]
    return df_sim, df_obv

# Tmax
df_QobsTmax = infer_Qi_Ti(df_Qobs, df_Tmax)
df_QbcTmax = infer_Qi_Ti(df_bc_Q, df_Tmax)
df_QbcTmax, df_QobsTmax = update_to_rm_thermal_release_effect(df_QbcTmax, df_QobsTmax, df_res)
df_QobsTmax.to_csv(pn.data.decomposed_Ti_Qi.get() / "df_QobsTmax.csv")
df_QbcTmax.to_csv(pn.data.decomposed_Ti_Qi.get() / "df_QbcTmax.csv")
#%%
# Tavg
df_QobsTavg = infer_Qi_Ti(df_Qobs, df_Tavg)
df_QbcTavg = infer_Qi_Ti(df_bc_Q, df_Tavg)
df_QbcTavg, df_QobsTavg = update_to_rm_thermal_release_effect(df_QbcTavg, df_QobsTavg, df_res)
df_QobsTavg.to_csv(pn.data.decomposed_Ti_Qi.get() / "df_QobsTavg.csv")
df_QbcTavg.to_csv(pn.data.decomposed_Ti_Qi.get() / "df_QbcTavg.csv")

#%%
r"""
import matplotlib.pyplot as plt
def plot_QT(df, start="2010-01-01", end="2010-12-31", title=""):
    df = df[start:end].copy()
    alpha, lw = 0.6, 1
    fig, axes = plt.subplots(nrows=2, figsize=(7, 5), sharex=True)
    axes = axes.flatten()
    ax = axes[0]
    df[['Q_i', 'Q_C', 'Q_L']].plot(ax=ax, alpha=alpha, lw=lw)
    ax.set_ylabel("Q")
    ax.set_title(title)
    ax.legend(ncols=3, frameon=False, loc="upper left")

    ax = axes[1]
    df[['T_i', 'T_C', 'T_L']].plot(ax=ax, alpha=alpha, lw=lw)
    ax.set_ylabel("T")
    ax.legend(ncols=3, frameon=False, loc="upper left")

    # ax = axes[2]
    # df[['QT_i', 'QT_C', 'QT_L']].plot(ax=ax, alpha=alpha, lw=lw)
    # ax.set_ylabel("QT")
    # ax.legend(ncols=3, frameon=False, loc="upper left")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()
start = "2010-01-01"
end = "2010-12-31"
plot_QT(df_QobsTmax, start, end, "QobsTmax")
plot_QT(df_QobsTavg, start, end, "QobsTavg")
"""