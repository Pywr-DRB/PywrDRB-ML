import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_parallel_coords_with_kde(
        ax, df, columns, add_background_grey_lines=True,
        dict_kde_dfs=None, dict_colorlines_dfs=None,
        soln_labels=None, objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "-Jrel", "Jadd"],
        ideal_direction='top',
        fontsize=10, kde_scale=0.12,
        cmap_kdes={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'},
        cmap_lines={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'},
        cmap_highlights={
            'no_ctrl': 'k', 'rule_based': '#E41A1C',
            'historic\n(2010-2023)': "blue",
            "RBF (utilize\n1620 mgd\nbank size)": "lime",
            "RBF-1\n(Jtubr=1.00)": "lime",
            "RBF-2\n(Jtubr=0.98)": "aquamarine"
            },
        ls_highlights={},
        zorder_highlights={}
        ):
    # General layout settings
    df_subset = df.loc[:, columns].copy()   # Select only the specified columns
    num_axes = len(columns)                 # Number of axes to plot
    right_space = 1     # Space on the right side of the plot (additional axis space for KDE)
    x_spacing = np.linspace(0, num_axes - 1 + right_space, num_axes + right_space) # Spacing for the x-axis

    # Auto-compute bounds if not provided
    if objmins is None: objmins = df_subset.min().tolist()
    if objmaxs is None: objmaxs = df_subset.max().tolist()

    # Normalize objectives
    tops, bottoms = np.array(objmaxs[:num_axes]), np.array(objmins[:num_axes])
    if ideal_direction == 'top': df_subset = (df_subset - bottoms) / (tops - bottoms)
    elif ideal_direction == 'bottom': df_subset = (bottoms - df_subset) / (bottoms - tops)
    else: raise ValueError('ideal_direction must be "top" or "bottom"')

    # Normalize dict_kde_dfs & dict_colorlines_dfs if provided
    dict_kde_dfs_scaled = {}
    if dict_kde_dfs is not None:
        for i, key in enumerate(dict_kde_dfs):
            dff = dict_kde_dfs[key].copy()
            for o, obj in enumerate(columns):
                if ideal_direction == 'top': dff[obj] = (dff[obj] - bottoms[o]) / (tops[o] - bottoms[o])
                elif ideal_direction == 'bottom': dff[obj] = (bottoms[o] - dff[obj]) / (bottoms[o] - tops[o])
            dict_kde_dfs_scaled[key] = dff

    dict_colorlines_dfs_scaled = {}
    if dict_colorlines_dfs is not None:
        for i, key in enumerate(dict_colorlines_dfs):
            dff = dict_colorlines_dfs[key].copy()
            for o, obj in enumerate(columns):
                if ideal_direction == 'top': dff[obj] = (dff[obj] - bottoms[o]) / (tops[o] - bottoms[o])
                elif ideal_direction == 'bottom': dff[obj] = (bottoms[o] - dff[obj]) / (bottoms[o] - tops[o])
            dict_colorlines_dfs_scaled[key] = dff

    # Plot background lines
    if add_background_grey_lines:
        for i in range(df_subset.shape[0]):
            for j in range(num_axes - 1):
                y = [df_subset.iloc[i, j], df_subset.iloc[i, j + 1]]
                x = [x_spacing[j], x_spacing[j + 1]]
                ax.plot(x, y, c='0.8', alpha=0.4, zorder=1, lw=1)

    if dict_colorlines_dfs is not None:
        for ic, col in enumerate(dict_colorlines_dfs_scaled):
            for i in range(dict_colorlines_dfs_scaled[col].shape[0]):
                for j in range(num_axes - 1):
                    y = [dict_colorlines_dfs_scaled[col].iloc[i, j], dict_colorlines_dfs_scaled[col].iloc[i, j + 1]]
                    x = [x_spacing[j], x_spacing[j + 1]]
                    ax.plot(x, y, c=cmap_kdes[col], alpha=0.4, zorder=1, lw=1)

    # Axis lines and ticks
    for j in range(num_axes):
        ax.annotate(str(round(tops[j], 1)), [x_spacing[j], 1.02], ha='center', va='bottom', fontsize=fontsize)
        bottom_label = str(round(bottoms[j], 1))
        ax.annotate(bottom_label, [x_spacing[j], -0.02], ha='center', va='top', fontsize=fontsize)
        ax.plot([x_spacing[j], x_spacing[j]], [0, 1], c='k', zorder=2)
        for y in np.arange(0, 1.001, 0.2):
            ax.plot([x_spacing[j] - 0.03, x_spacing[j] + 0.03], [y, y], c='k', zorder=2)

    # Clean aesthetics
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(-0.4, 1.1)
    ax.patch.set_alpha(0)

    # Add arrows to indicate the ideal direction of preference
    if ideal_direction == 'top':
        ax.arrow(x_spacing[0] - 0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(x_spacing[0] - 0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(x_spacing[0] - 0.3, 0.5),
                ha='center', va='center', rotation=90, fontsize=fontsize)

    # Axis labels
    for i, l in enumerate(axes_labels[:num_axes]):
        ax.annotate(l, xy=(x_spacing[i], -0.12), ha='center', va='top', fontsize=fontsize)

    # Highlight selected solutions
    if soln_labels is not None:
        for soln_label in soln_labels:
            c = cmap_highlights.get(soln_label, "lime")
            ls = ls_highlights.get(soln_label, "-")
            zorder = zorder_highlights.get(soln_label, 0)
            soln_data = df_subset.loc[df['label'] == soln_label]
            xx, yy = [], []
            for j in range(num_axes - 1):
                x = np.linspace(x_spacing[j], x_spacing[j + 1], 11)
                y = soln_data.iloc[0, j] + (x - x_spacing[j]) * \
                    (soln_data.iloc[0, j + 1] - soln_data.iloc[0, j]) / (x_spacing[j + 1] - x_spacing[j])
                xx += list(x); yy += list(y)

            if ls == "-":
                ax.plot(xx, yy, c='k', lw=2.6, zorder=49+zorder)
                ax.plot(xx, yy, c=c, lw=1.7, zorder=50+zorder, label=soln_label)
            else:
                ax.plot(xx, yy, c=c, lw=1.7, ls=ls, zorder=50+zorder, label=soln_label)
    ax.legend(frameon=False, loc="center right")

    # KDE shading
    if dict_kde_dfs is not None:
        for i, key in enumerate(dict_kde_dfs):
            dff = dict_kde_dfs_scaled[key]
            for o, obj in enumerate(columns): #[1:]
                y = np.arange(0, 1, 0.01)
                data = dff[obj]
                kde = sm.nonparametric.KDEUnivariate(data)
                kde.fit(bw=0.025)
                kde_scale = kde_scale
                x = np.array([kde.evaluate(v)[0] * kde_scale if not np.isnan(kde.evaluate(v)[0]) else 0 for v in y])

                # Manually truncate kde
                mask = (y >= min(data)) & (y <= max(data))
                x = x[mask]
                y = y[mask]
                ax.fill_betweenx(y, x + x_spacing[o], x_spacing[o],
                                 where=(x > 0.00005), lw=1, alpha=0.6, zorder=4, fc=cmap_kdes[key], ec='k')
    ax.set_xlim([x_spacing[0] - 0.3, x_spacing[-1] + 0.3])
    return ax


#%
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

df_highlight = pd.DataFrame()
# df_highlight["Jtubr"] = [0, 0.2942*3, 0.4994*3, 1.0]
# df_highlight["-Jrel"] = [-0.2375, -0.3185, -0.3583, -0.4434]
# df_highlight["Jadd"] = [1, 0.916, 0.764, 0.8037]

# With past 3 days [155 24]
# df_highlight["Jtubr"] = [0, 0.2942*3, 0.4994*3, 1.0023, 0.9813]
# df_highlight["-Jrel"] = [-0.2375, -0.3185, -0.3583, -0.456, -0.3561]
# df_highlight["Jadd"] = [1, 0.916, 0.764, 0.8904, 0.7558]
# df_highlight["label"] = ["no_ctrl", "rule_based", "historic\n(2010-2023)",
#                          "RBF-1\n(Jtubr=1.00)",
#                          "RBF-2\n(Jtubr=0.98)",]

# Without past three days [159 108]
df_highlight["Jtubr"] = [0,        0.2942*3,  0.4994*3, 0.9999,  0.9279,  2.1774,  3.0]
df_highlight["-Jrel"] = [-0.2375, -0.3185,   -0.3583,  -0.3778, -0.4524, -0.6585, -0.4229]
df_highlight["Jadd"] =  [1,        0.916,     0.764,    0.742,   0.7684,  0.7405,  0.5146]
df_highlight["label"] = ["no_ctrl", "rule_based", "historic\n(2010-2023)",
                         "RBF-1\n(Jtubr=1.0)", # 159
                         "RBF-2\n(Jtubr=0.93)", # 108
                         "RBF-3\n(Jrel=0.66)", # 106
                         "RBF-4\n(Jadd=0.51)", # 11
                         ]
cmap_highlights={
    'no_ctrl': 'k', 'rule_based': '#E41A1C',
    'historic\n(2010-2023)': "blue",
    "RBF-1\n(Jtubr=1.0)": "lime",
    "RBF-2\n(Jtubr=0.93)": "aquamarine",
    "RBF-3\n(Jrel=0.66)": "saddlebrown",
    "RBF-4\n(Jadd=0.51)": "peru"
    }
ls_highlights={
    "RBF-3\n(Jrel=0.66)": "--",
    "RBF-4\n(Jadd=0.51)": "--"
    }
zorder_highlights={
    'no_ctrl': 0, 'rule_based': 1,
    'historic\n(2010-2023)': 2,
    "RBF-1\n(Jtubr=1.0)": 8,
    "RBF-2\n(Jtubr=0.93)": 9,
    "RBF-3\n(Jrel=0.66)": 4,
    "RBF-4\n(Jadd=0.51)": 5
    }

for policy in ["GaussianRBFPolicy"]:#, "RegressionPolicy", "CubicRBFPolicy"]:#, "GeneralizedPiecewiseLinearPolicy"]:
    #policy = "GaussianRBFPolicy"
    job_id = "135322" #"135218"

    df_ref = clt.borg.read_ref(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/borg.ref"))
    df_ref = df_ref[['obj3', 'obj1', 'obj2']]
    df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
    df_ref["Jtubr"] *= 3
    df_ref["label"] = df_ref.index

    df_ref = pd.concat([df_ref, df_highlight])

    dict_kde_dfs = {
        0: df_ref[df_ref['Jtubr'] <= 1],
        1: df_ref[(df_ref['Jtubr'] > 1) & (df_ref['Jtubr'] <= 2)],
        2: df_ref[df_ref['Jtubr'] > 2],
        }
    dict_colorlines_dfs = {
        0: df_ref[df_ref['Jtubr'] <= 1],
        }

    df = df_ref
    fig, ax = plt.subplots()
    plot_parallel_coords_with_kde(
        ax, df, columns=["Jtubr", "-Jrel", "Jadd"],
        dict_kde_dfs=dict_kde_dfs,
        dict_colorlines_dfs=dict_colorlines_dfs,
        soln_labels=df_highlight["label"].to_list(),
        objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "-Jrel", "Jadd"],
        ideal_direction='bottom', fontsize=10, kde_scale=0.05,
        cmap_highlights=cmap_highlights,
        ls_highlights=ls_highlights,
        zorder_highlights=zorder_highlights
        )
    #ax.set_title(policy)
    plt.tight_layout()
    plt.show()

#%% Compare thermal release
import joblib
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio
from tqdm import tqdm
from src.policies import GaussianRBFPolicy
df_ref = clt.borg.read_ref(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/borg.ref"))

database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historical"] = database["rel_thermal"]
df_res["rule_based"] = df_rulebased["thermal_releases"]
df_res["Tmax (no_ctrl)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (rule_based)"] = df_rulebased["T_L_mu"]

df_objs = pd.DataFrame()

sol_idx = 159 #11 #106 #108 #159 #              #24#, 155155 #
for i, sol_idx in enumerate([159, 108, 106, 11]):
    params = df_ref.iloc[sol_idx, :-3]
    n_dim = 3#4  # Number of dimensions for the policy
    n_basis = 2  # Number of basis functions for the Gaussian RBF policy


    def eval_func(*params):
        # Initialize the thermal control policy with specific parameters
        policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
        #params = policy.gen_params(seed=42)[0]
        minmaxscalers = joblib.load(pn.stage1_thermal_ctrl_decoupled_withNowcast.get() / "minmaxscalers.gz")
        policy.set_params(*params)  # Generate random parameters for the policy
        def return_dps_func():#*params):
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
                ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0)
                forecast_T_L_mu = ml_model.forecast_T_L_mu_arr[-1]
                forecast_T_C_mu = ml_model.forecast_T_C_mu_arr[-1]
                #forecast_T_i_mu = ml_model.forecast_T_i_mu_arr[-1]

                remained_bank_ratio = ml_model.remained_bank_amount/ml_model.thermal_mitigation_bank_size

                X = np.array([
                    minmaxscalers["T_L"].transform(pd.DataFrame([[forecast_T_L_mu]], columns=["T_L"]))[0][0],
                    minmaxscalers["T_C"].transform(pd.DataFrame([[forecast_T_C_mu]], columns=["T_C"]))[0][0],
                    remained_bank_ratio,
                    ])
                ml_model.X_dps.append(X)
                # Make thermal release decision and record the thermal release
                thermal_release = policy.run(X=X) * 300 # assuming the maximum thermal release is 300 MGD per day
                # Ensure thermal release does not exceed the bank size
                thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
                return thermal_release
            return dps_func

        # Prepare the decision-making function with parameters
        #params = []
        dm_func = return_dps_func() #*params

        ml_model = WaterTempLSTMModel(
            model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
            model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
            Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
            debug=True,
            thermal_mitigation_bank_size=1620 * 3,  # mgd
            )
        ml_model.load_data(database)
        ml_model.X_dps = []

        dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
        for date in tqdm(dates, desc="Running thermal control policy"):
            Q_C = None  # Placeholder for controlled release
            Q_i = None  # Placeholder for inflow
            cannonsville_storage_pct = None  # Placeholder for storage percentage

            if date.month in [6, 7, 8]:
                thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
            else:
                thermal_release = 0

            # Update data in the ml_model for the next step(s) model update.
            t = ml_model.t
            ml_model.Q_C[t] += thermal_release
            Q_C = ml_model.Q_C[t]
            try:
                ml_model.X_1[t, ml_model.x_vars_1.index(ml_model.Q_C_lstm_var_name)] = Q_C
            except ValueError:
                if ml_model.debug:
                    print(f"Warning: '{ml_model.Q_C_lstm_var_name}' not found in lstm1.x_vars. Skipping update.")
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
    #df_rbf['thermal_releases'] = df_rbf['thermal_releases'].fillna(0)
    Jrel = compute_reliability(df_rbf, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
    Jadd = compute_max_annual_accumulated_degree_days(df_rbf, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
    Jtubr = compute_max_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))
    Jtubr_avg = compute_mean_thermal_bank_usage_ratio(df_rbf, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

    objs = [Jtubr*3, -Jrel, Jadd, Jtubr_avg*3]

    df_res[f"RBF-{i+1}"] = df_rbf["thermal_releases"]
    df_res[f"Tmax (RBF-{i+1})"] = df_rbf["T_L_mu"]
    df_objs[f"RBF-{i+1}"] = objs

    df_rbf.to_csv(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}") / f"df_RBF{i+1}_{sol_idx}.csv")


# 159 [0.9999, -0.3778, 0.742, 0.9665999999999999]
# 24 (better Jadd): Out[95]: [0.9813, -0.356, 0.7558, 0.9606]
# 115 (better Jrel):         [1.0023, -0.456, 0.8904, 0.1869]
#%%
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historical"] = database["rel_thermal"]
df_res["rule_based"] = df_rulebased["thermal_releases"]
df_res["Tmax (no_ctrl)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (rule_based)"] = df_rulebased["T_L_mu"]

for i, sol_idx in enumerate([159, 108, 106, 11]):
    df_rbf = pd.read_csv(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}") / f"df_RBF{i+1}_{sol_idx}.csv", parse_dates=True, index_col=[0])
    df_res[f"RBF-{i+1}"] = df_rbf["thermal_releases"]
    df_res[f"Tmax (RBF-{i+1})"] = df_rbf["T_L_mu"]

#%%
yr = 2023
for yr in range(1979, 2024):
    df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

    colors = {
        'no_ctrl': 'k',
        'rule_based': '#E41A1C',
        'historic': "blue",
        "RBF-1": "lime",
        "RBF-2": "aquamarine",
        "RBF-3": "saddlebrown",
        "RBF-4": "peru"
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
    for rbf_name in ["RBF-1", "RBF-2", "RBF-3", "RBF-4"]:
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
    for rbf_name, zorder in zip(["RBF-1", "RBF-2", "RBF-3", "RBF-4"], [50, 60, 40, 30]):
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
    pn.outputs.mkdir(f"stage1_nowcast_{policy}_{job_id}/figures/RBFs")
    clt.fig.savefig(fig, pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/figures/RBFs") / f"RBFs_{yr}.jpg")
    plt.show()

#%%
rbf_name = "RBF-1"
for rbf_name in ["RBF-1", "RBF-2", "RBF-3", "RBF-4"]:
    yr = 2023
    for yr in range(1979, 2023):
        df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

        # Create 2-row subplot with height ratio 1:3
        fig, axes = plt.subplots(
            2, 1, figsize=(6, 5),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0},  # 1. Remove vertical space
            sharex=True
        )

        # ----------------- Top subplot -----------------
        ax1 = axes[0]
        ax1.plot(df_["Tmax (no_ctrl)"], ls="-", color='red', label="Tmax (no_ctrl)")
        ax1.plot(df_["Tmax (rule_based)"], ls="-", color='chocolate', label="Tmax (rule_based)")
        ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color='dodgerblue', label=f"Tmax ({rbf_name})")
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
            plt.setp(stemlines, lw=1, zorder=0)
            plt.setp(markerline, ms=4, zorder=0)
            plt.setp(baseline, visible=False)
        else:
            # Add a dummy invisible point just to include legend entry
            ax2.plot([], [], marker='o', color='k', linestyle='None', label="historical")

        # 5. rbf and rule_based as bars
        ax2.bar(df_.index, df_[f"{rbf_name}"], width=1.0, color='dodgerblue', label=f"{rbf_name}", alpha=0.6, zorder=4)
        ax2.bar(df_.index, df_["rule_based"], width=1.0, color='chocolate', label="rule_based", alpha=0.6, zorder=2)

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
        pn.outputs.mkdir(f"stage1_nowcast_{policy}_{job_id}/figures/{rbf_name}")
        clt.fig.savefig(fig, pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/figures/{rbf_name}") / f"{rbf_name}_{yr}.jpg")
        plt.show()

#%% Analyze X
#X = pd.DataFrame(ml_model.X_dps, columns=["forecast_T_L_mu", "T_L_past3days", "forecast_T_C_mu", "remained_bank_ratio"])
X = pd.DataFrame(ml_model.X_dps, columns=["forecast_T_L_mu", "forecast_T_C_mu", "remained_bank_ratio"])
X["thermal_release"] = df_rbf.loc[df_rbf.index.month.isin([6, 7, 8]), "thermal_releases"].values/300

# Select the last 92*5 rows
X_subset = X.iloc[-92*5:, :]

# Plot using fig and ax
fig, ax = plt.subplots(figsize=(6, 4))
X_subset.plot(ax=ax)

# Move the legend outside of the plot
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.set_title("Feature Trends Over Last 5 Summers")
ax.set_xlabel("Index")
ax.set_ylabel("Normalized Values")

plt.tight_layout()
plt.show()

#%% Plot annual historical releases
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
db = database[["rel_thermal", "rel_thermal_org"]]['2008-01-01': '2023-12-31']
fig, ax = plt.subplots(figsize=(5, 4))
db.groupby(db.index.year).sum().plot(kind="bar", ax=ax)
ax.axhline(1620, c="k", lw=1)
plt.tight_layout()
plt.show()
#%% Plot thermal releases


#%%
yr = 2019
# Slice the DataFrame
df_ = df_res.loc[f"{yr}-5-20":f"{yr}-9-10", :]

# Create 2-row subplot with height ratio 1:3
fig, axes = plt.subplots(
    2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [1, 3]}, sharex=True
)

ax = axes[0]
# Top subplot: Tmax (no_ctrl)
ax.plot(df_["Tmax (no_ctrl)"], ls="-", color='grey', label="Tmax (no_ctrl)")
ax.axhline(24, lw=1, c="k", ls=":")
ax.set_ylabel("$T_{max}$ (°C)")
ax.legend(frameon=False)
ax.grid(True, axis='y', lw=0.3, ls="--")

ax = axes[1]
# Bottom subplot: Thermal releases
ax.plot(df_["historical"], color='k', lw=2, ls="-", label="historical")
ax.plot(df_["rbf"], color='salmon', lw=2, label="rbf")
ax.plot(df_["rule_based"], color='dodgerblue', lw=2, label="rule_based")
ax.grid(True, axis='y', lw=0.3, ls="--")

custom_ticks = pd.to_datetime([f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15", f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])  # show only month/day

ax.set_ylabel("Thermal release (mgd)")
ax.set_xlabel(f"Date (Year={yr})")
#%%
yr = 2019
for yr in range(2006, 2023):
    df_ = df_res.loc[f"{yr}-5-20":f"{yr}-9-10", :]

    fig, ax = plt.subplots()
    ax.plot(df_["historical"], color='k', lw=2, ls="-", label="historical")
    ax.plot(df_["rbf"], color='salmon', lw=2, label="rbf")
    ax.plot(df_["rule_based"], color='dodgerblue', lw=2, label="rule_based")
    ax.grid(True, axis='y', lw=0.3, ls="--")
    ax.legend(frameon=False)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks[::3])
    ax.set_xlim([df_.index[0], df_.index[-1]])
    ax.set_ylabel("Thermal release (mgd)")
    ax.set_xlabel("Date")

    ax2 = ax.twinx()
    ax2.plot(df_["Tmax (no_ctrl)"], ls="-", color='grey', label="Tmax (no_ctrl)")
    ax2.axhline(24, lw=1, c="k", ls=":")
    ax2.set_ylabel("$T_{max}$ (°C)")

    plt.tight_layout()
    #clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "compare_with_hist_tr.jpg")
    plt.show()
