#%% Compare thermal release
import pandas as pd
import numpy as np
import clt
import joblib
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio
from tqdm import tqdm
from src.policies import GaussianRBFPolicy
import matplotlib.pyplot as plt
import seaborn as sns

policy = "GaussianRBFPolicy"
job_id = "138146"
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))

database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])
df_hist = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historical"] = database["rel_thermal"]
df_res["rule_based"] = df_rulebased["thermal_releases"]
df_res["Tmax (no_ctrl)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (rule_based)"] = df_rulebased["T_L_mu"]
df_res["Tmax (historic)"] = df_hist["T_L_mu"]

df_objs = pd.DataFrame()

#for i, sol_idx in enumerate([47]):  # , 33, 49, 11

def reevaluation(sol_idx, update_lstm_inputs):
    params = df_ref.iloc[sol_idx, :-3]
    n_dim = 3    # Number of dimensions for the policy
    n_basis = 2  # Number of basis functions for the Gaussian RBF policy

    def eval_func(*params):
        database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
        # Initialize the thermal control policy with specific parameters
        policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
        minmaxscalers = joblib.load(pn.stage1_thermal_ctrl_decoupled_withNowcast.get() / "minmaxscalers.gz")
        policy.set_params(*params)  # Generate random parameters for the policy
        def return_dps_func():
            # Define the function that will be used for the control algorithm
            def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
                # Retrieve the ml_model from the model
                ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
                # Reset the bank amount at the beginning of June
                if current_date.day == 1 and current_date.month == 6:
                    ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
                ml_model.update_until(date=current_date)

                # Prepare the inputs
                # Nowcast/forecast
                # Have to retrieve storage info after update until such that t have been moved forward
                cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t-1]  # Placeholder for storage percentage
                ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=cannonsville_storage_pct, lead_time=0)
                forecast_T_L_mu = ml_model.forecast_T_L_mu_arr[-1]
                forecast_T_C_mu = ml_model.forecast_T_C_mu_arr[-1]

                remained_bank_ratio = ml_model.remained_bank_amount/ml_model.thermal_mitigation_bank_size

                X = np.array([
                    minmaxscalers["T_L"].transform(pd.DataFrame([[forecast_T_L_mu]], columns=["T_L"]))[0][0],
                    minmaxscalers["T_C"].transform(pd.DataFrame([[forecast_T_C_mu]], columns=["T_C"]))[0][0],
                    remained_bank_ratio,
                    ])

                # Make thermal release decision and record the thermal release
                thermal_release = policy.run(X=X) * 300 # assuming the maximum thermal release is 300 MGD per day
                # Ensure thermal release does not exceed the bank size
                thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
                return thermal_release
            return dps_func

        # Prepare the decision-making function with parameters
        dm_func = return_dps_func()

        ml_model = WaterTempLSTMModel(
            model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
            model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
            Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
            debug=True,
            thermal_mitigation_bank_size=1620 * 3,  # mgd
            )
        ml_model.load_data(database)

        dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
        for t, date in tqdm(enumerate(dates), desc="Running thermal control policy", disable=True):
            Q_C = None  # Placeholder for controlled release
            Q_i = None  # Placeholder for inflow
            cannonsville_storage_pct = None        

            if date.month in [6, 7, 8]:
                thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
            else:
                thermal_release = 0

            # Update data in the ml_model for the next step(s) model update.
            acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
            ml_model.Q_C[t] += thermal_release
            acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
            ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  
            # Use the original inputs to predict temp
            
            if update_lstm_inputs:
                Q_C = ml_model.Q_C[t]
                cannonsville_storage_pct = ml_model.cannonsville_storage_pct[t]
                try:
                    ml_model.X_1[t, ml_model.x_vars_1.index(ml_model.Q_C_lstm_var_name)] = Q_C
                except ValueError:
                    if ml_model.debug:
                        print(f"Warning: '{ml_model.Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
                try:
                    ml_model.X_1[t, ml_model.x_vars_1.index(ml_model.cannonsville_storage_pct_lstm_var_name)] = cannonsville_storage_pct
                except ValueError:
                    if ml_model.debug:
                        print(f"Warning: '{ml_model.cannonsville_storage_pct_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
                try:
                    ml_model.X_2[t, ml_model.x_vars_2.index(ml_model.Q_C_lstm_var_name)] = Q_C
                except ValueError:
                    if ml_model.debug:
                        print(f"Warning: '{ml_model.Q_C_lstm_var_name}' not found in lstm2.x_vars. Skipping update.")

            # Record
            ml_model.remained_bank_amount -= thermal_release
            ml_model.records["thermal_releases"][ml_model.t] = thermal_release
            ml_model.records["remained_bank_amounts"][ml_model.t] = ml_model.remained_bank_amount

        # Update the model until the end of the simulation period
        ml_model.update_until(date="2024-01-01")
        df = pd.DataFrame(ml_model.records, index=ml_model.dates)
        return ml_model, df
    ml_model, df_rbf = eval_func(*params)

    Jrel = compute_reliability(df_rbf, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
    Jadd = compute_max_annual_accumulated_degree_days(df_rbf, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
    Jtubr = compute_max_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
    Jtubr_avg = compute_mean_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

    objs = [Jtubr*3, -Jrel, Jadd, Jtubr_avg*3]

    return df_rbf, objs

#%% 
df_rbf_update, objs_update = reevaluation(sol_idx=47, update_lstm_inputs=True)
df_rbf_noupdate, objs_noupdate = reevaluation(sol_idx=47, update_lstm_inputs=False)

df_res[f"rbf_update"] = df_rbf_update["thermal_releases"]
df_res[f"Tmax (rbf_update)"] = df_rbf_update["T_L_mu"]
df_res[f"rbf_noupdate"] = df_rbf_noupdate["thermal_releases"]
df_res[f"Tmax (rbf_noupdate)"] = df_rbf_noupdate["T_L_mu"]
#%%
for yr in range(1979, 2024):
    df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

    colors = {
        'no_ctrl': 'k',
        'rule_based': '#E41A1C',
        'historic': "blue",
        "rbf_update": "lime",
        "rbf_noupdate": "peru", #"aquamarine",
        # "RBF-3": "saddlebrown",
        # "RBF-4": "peru"
        }

    # Create 2-row subplot with height ratio 1:3
    fig, axes = plt.subplots(
        2, 1, figsize=(6, 5),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0},  # 1. Remove vertical space
        sharex=True
    )

    # ----------------- Top subplot -----------------
    ax1 = axes[0]
    ax1.plot(df_["Tmax (no_ctrl)"], ls="-", color=colors['no_ctrl'], label="Tmax (no_ctrl)")
    ax1.plot(df_["Tmax (rule_based)"], ls="-", color=colors['rule_based'], label="Tmax (rule_based)")
    ax1.plot(df_["Tmax (historic)"], ls="-", color=colors['historic'], label="Tmax (historic)")
    for rbf_name in ["rbf_update", "rbf_noupdate"]:
        ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color=colors[f"{rbf_name}"], label=f"Tmax ({rbf_name})") #, color='dodgerblue'
    ax1.axhline(24, lw=1, c="k", ls=":")
    ax1.set_ylabel("$T_{max}$ (°C)")
    ax1.grid(True, axis='y', lw=0.3, ls="--")
    ax1.tick_params(axis='x', direction='in')  # 2. Inward x-ticks
    ax1.set_ylim([18, 27])
    ax1.set_yticks([20, 24])

    # ----------------- Bottom subplot -----------------
    ax2 = axes[1]
    # 4. historical as markers
    # Filter non-zero values
    non_zero_mask = df_["historical"] != 0
    x_vals = df_.index[non_zero_mask]
    y_vals = df_["historical"][non_zero_mask]

    # Plot only non-zero historical values with stem
    if not x_vals.empty:
        # Plot actual non-zero stems
        markerline, stemlines, baseline = ax2.stem(
            x_vals, y_vals,
            linefmt="k-", markerfmt="ko", basefmt=" ", label="historical"
        )
        markerline.set_color(colors['historic'])       # marker color
        stemlines.set_color(colors['historic'])
        plt.setp(stemlines, lw=1, zorder=80)
        plt.setp(markerline, ms=4, zorder=80)
        plt.setp(baseline, visible=False)
    else:
        # Add a dummy invisible point just to include legend entry
        ax2.plot([], [], marker='o', color='k', linestyle='None', label="historical")

    # 5. rbf and rule_based as bars
    for rbf_name, zorder in zip(["rbf_update", "rbf_noupdate"], [50, 60]):
        ax2.bar(df_.index, df_[f"{rbf_name}"], width=1.0, color=colors[f"{rbf_name}"], label=f"{rbf_name}", alpha=0.6, zorder=4) # color='dodgerblue'
    ax2.bar(df_.index, df_["rule_based"], width=1.0, color=colors['rule_based'], label="rule_based", alpha=0.6, zorder=70)

    ax2.grid(True, axis='y', lw=0.3, ls="--")
    ax2.set_ylabel("Thermal release (mgd)")
    ax2.set_xlabel(f"Date (Year={yr})")

    # Custom xticks
    custom_ticks = pd.to_datetime([
        f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15",
        f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"
    ])
    ax2.set_xticks(custom_ticks)
    ax2.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])

    ax2.set_ylim([0, 300])

    # 3. Combine legends and place outside
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc='center left',
        bbox_to_anchor=(0.85, 0.5),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for external legend
    #pn.outputs.mkdir(f"stage1_nowcast_{policy}_{job_id}/figures/RBFs")
    #clt.fig.savefig(fig, pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/figures/RBFs") / f"RBFs_{yr}.jpg")
    plt.show()
