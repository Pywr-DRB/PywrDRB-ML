import pandas as pd
import pathnavigator
import clt
if pathnavigator.os_name == 'Windows':  
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-LSTMs"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-LSTMs")
pn = pathnavigator.create(root_dir)
pn.chdir()
pn.data.mkdir("decomposed_Ti_Qi")

start = "1945-01-01"
end = "2024-12-31"
rng = pd.date_range(start, end, freq="D")

df_lordville = pd.read_csv(pn.data.raw.get("nwis_Lordville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
df_cannonsville = pd.read_csv(pn.data.raw.get("nwis_Cannonsville_degC_mgd.csv"), parse_dates=True, index_col=["date"])[start:end]
df_bc = pd.read_csv(pn.data.raw.get("pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv"), parse_dates=True, index_col=["date"])[start:end]
df_res = pd.read_excel(pn.data.get("reservoir_release_sep_thermal_ctrl_mgd_manually_picked.xlsx"), parse_dates=True, index_col="date")[start:end]
df_res = clt.utils.re_datetime_index(df_res, start, end, fill_value=0)

df_Qobs = pd.DataFrame(index=rng)
df_Qobs["Q_C"] = df_cannonsville["discharge_cannonsville"]  
df_Qobs["Q_L"] = df_lordville["discharge_lordville"] 

df_bc_Q = pd.DataFrame(index=rng)
df_bc_Q["Q_C"] = df_bc["01425000"] 
df_bc_Q["Q_L"] = df_bc["flow_lordville"] 

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
    df_Q["Q_i"] = df_Q["Q_L"] - df_Q["Q_C"]
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