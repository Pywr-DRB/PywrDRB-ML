import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_parallel_coords_with_kde(
        ax, df, columns, dict_kde_dfs=None,
        soln_labels=None, objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "-Jrel", "Jadd"],
        ideal_direction='top', fontsize=10, kde_scale=0.12
        ):

    df_subset = df.loc[:, columns].copy()
    num_axes = len(columns)
    right_space = 1
    x_spacing = np.linspace(0, num_axes - 1 + right_space, num_axes + right_space)

    # Auto-compute bounds if not provided
    if objmins is None:
        objmins = df_subset.min().tolist()
    if objmaxs is None:
        objmaxs = df_subset.max().tolist()

    # Normalize objectives
    tops, bottoms = np.array(objmaxs[:num_axes]), np.array(objmins[:num_axes])
    if ideal_direction == 'top':
        df_subset = (df_subset - bottoms) / (tops - bottoms)
    elif ideal_direction == 'bottom':
        df_subset = (bottoms - df_subset) / (bottoms - tops)
    else:
        raise ValueError('ideal_direction must be "top" or "bottom"')

    # Plot background lines
    # colors_brewer = ['#1b9e77', '#d95f02', '#7570b3']
    # if dict_kde_dfs is not None:
    #     for ic, col in enumerate(dict_kde_dfs):
    #         for i in range(dict_kde_dfs[col].shape[0]):
    #             for j in range(num_axes - 1):
    #                 y = [df_subset.iloc[i, j], df_subset.iloc[i, j + 1]]
    #                 x = [x_spacing[j], x_spacing[j + 1]]
    #                 ax.plot(x, y, c=colors_brewer[ic], alpha=0.4, zorder=1, lw=1)
    # else: # Original
    for i in range(df_subset.shape[0]):
        for j in range(num_axes - 1):
            y = [df_subset.iloc[i, j], df_subset.iloc[i, j + 1]]
            x = [x_spacing[j], x_spacing[j + 1]]
            ax.plot(x, y, c='0.8', alpha=0.4, zorder=1, lw=1)

    # Axis lines and ticks
    for j in range(num_axes):
        ax.annotate(str(round(tops[j])), [x_spacing[j], 1.02], ha='center', va='bottom', fontsize=fontsize)
        bottom_label = str(round(bottoms[j]))
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
    colors_brewer = {'no_ctrl': 'red', 'rule_based': 'blue'}
    if soln_labels is not None:
        for soln_label in soln_labels:
            c = colors_brewer.get(soln_label, 'k')
            soln_data = df_subset.loc[df['label'] == soln_label]
            xx, yy = [], []
            for j in range(num_axes - 1):
                x = np.linspace(x_spacing[j], x_spacing[j + 1], 11)
                y = soln_data.iloc[0, j] + (x - x_spacing[j]) * \
                    (soln_data.iloc[0, j + 1] - soln_data.iloc[0, j]) / (x_spacing[j + 1] - x_spacing[j])
                xx += list(x); yy += list(y)
            ax.plot(xx, yy, c='k', lw=2, zorder=5)
            ax.plot(xx, yy, c=c, lw=1, zorder=6, label=soln_label)
    ax.legend(frameon=False, loc="upper right")

    # KDE shading (I seperate this out)
    colors_brewer = ['#1b9e77', '#d95f02', '#7570b3']
    if dict_kde_dfs is not None:
        for i, col in enumerate(dict_kde_dfs):
            results_MC = dict_kde_dfs[col].copy()
            columns_MC = columns
            for o, obj in enumerate(columns_MC):
                scaled_col = f"{obj}_scaled"
                if ideal_direction == 'top':
                    results_MC[scaled_col] = (results_MC[obj] - bottoms[o]) / (tops[o] - bottoms[o])
                else:
                    results_MC[scaled_col] = (bottoms[o] - results_MC[obj]) / (bottoms[o] - tops[o])

            for o, obj in enumerate(columns_MC): #[1:]
                y = np.arange(0, 1, 0.01)
                data = results_MC[f'{obj}_scaled']
                kde = sm.nonparametric.KDEUnivariate(data)
                kde.fit(bw=0.025)
                kde_scale = kde_scale
                x = np.array([kde.evaluate(v)[0] * kde_scale if not np.isnan(kde.evaluate(v)[0]) else 0 for v in y])
                # ax.fill_betweenx(y, x + x_spacing[o + 1], x_spacing[o + 1],
                #                  where=(x > 0.00005), lw=1, alpha=0.6, zorder=4, fc=c, ec='k')
                ax.fill_betweenx(y, x + x_spacing[o], x_spacing[o],
                                 where=(x > 0.00005), lw=1, alpha=0.6, zorder=4, fc=colors_brewer[i%len(colors_brewer)], ec='k')

        # # Debug print
        # print(f"{soln_label} min, mean, max:")
        # for o, obj in enumerate(columns_MC[1:]):
        #     print(f"{obj}: {results_MC[obj].min():.2f}, {results_MC[obj].mean():.2f}, {results_MC[obj].max():.2f}")
        # print()

    ax.set_xlim([x_spacing[0] - 0.3, x_spacing[-1] + 0.3])



#%%
import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

df_highlight = pd.DataFrame()
df_highlight["Jtubr"] = [0, 0.0734]
df_highlight["-Jrel"] = [-0.2375, -0.3185]
df_highlight["Jadd"] = [132.4373, 0.916*132.4373]
df_highlight["label"] = ["no_ctrl", "rule_based"]

# for policy in ['GeneralizedPiecewiseLinearPolicy', 'RegressionPolicy', 'GaussianRBFPolicy', 'CubicRBFPolicy']:
#     #policy = "GaussianRBFPolicy"
#     job_id = "134717"

#     df_ref = clt.borg.read_ref(pn.outputs.get(f"stage1_{policy}_{job_id}/borg.ref"))
#     df_ref = df_ref[['obj3', 'obj1', 'obj2']]
#     df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
#     #df_ref["Jrel"] *= -1
#     df_ref["Jtubr"] *= 3
#     df_ref["label"] = df_ref.index
#     #df_ref["label"][0] = 1

#     df_ref = pd.concat([df_ref, df_highlight])

#     dict_kde_dfs = {
#         0: df_ref[df_ref['Jtubr'] <= 1],
#         1: df_ref[(df_ref['Jtubr'] > 1) & (df_ref['Jtubr'] <= 2)],
#         2: df_ref[df_ref['Jtubr'] > 2],
#         }

#     df = df_ref
#     fig, ax = plt.subplots()
#     plot_parallel_coords_with_kde(
#         ax, df, columns=["Jtubr", "-Jrel", "Jadd"], dict_kde_dfs=dict_kde_dfs,
#         soln_labels=df_highlight["label"].to_list(),
#         objmins=None, objmaxs=None,
#         axes_labels=["Jtubr", "-Jrel", "Jadd"],
#         ideal_direction='bottom', fontsize=10, kde_scale=0.05
#         )
#     ax.set_title(policy)
#     plt.tight_layout()
#     plt.show()
#%% New 111

import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

df_highlight = pd.DataFrame()
df_highlight["Jtubr"] = [0, 0.0734]
df_highlight["-Jrel"] = [-0.2375, -0.3185]
df_highlight["Jadd"] = [1, 0.916]
df_highlight["label"] = ["no_ctrl", "rule_based"]

for policy in ['GeneralizedPiecewiseLinearPolicy']:
    #policy = "GaussianRBFPolicy"
    job_id = "134795111"

    df_ref = clt.borg.read_ref(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/borg.ref"))
    df_ref = df_ref[['obj3', 'obj1', 'obj2']]
    df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
    #df_ref["Jrel"] *= -1
    df_ref["Jtubr"] *= 3
    df_ref["label"] = df_ref.index
    #df_ref["label"][0] = 1

    df_ref = pd.concat([df_ref, df_highlight])

    dict_kde_dfs = {
        0: df_ref[df_ref['Jtubr'] <= 1],
        1: df_ref[(df_ref['Jtubr'] > 1) & (df_ref['Jtubr'] <= 2)],
        2: df_ref[df_ref['Jtubr'] > 2],
        }

    df = df_ref
    fig, ax = plt.subplots()
    plot_parallel_coords_with_kde(
        ax, df, columns=["Jtubr", "-Jrel", "Jadd"], dict_kde_dfs=dict_kde_dfs,
        soln_labels=df_highlight["label"].to_list(),
        objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "-Jrel", "Jadd"],
        ideal_direction='bottom', fontsize=10, kde_scale=0.05
        )
    ax.set_title(policy)
    plt.tight_layout()
    plt.show()

#%% Compare thermal release
sol_idx = 67
df_ref = clt.borg.read_ref(pn.outputs.get(f"stage1_nowcast_{policy}_{job_id}/borg.ref"))
params = df_ref.iloc[sol_idx, :-3]

import joblib
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_mean_thermal_bank_usage_ratio
from stage1_thermal_ctrl_decoupled_withNowcast.lstm_thermal_ctrl_piecewise import eval_func, n_dim, n_steps
from src.policies import GeneralizedPiecewiseLinearPolicy

n_dim = 4  # Number of dimensions for the policy
n_steps = 4  # Number of steps for the piecewise linear policy
def eval_func(*params):
    database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
    # Initialize the thermal control policy with specific parameters
    policy = GeneralizedPiecewiseLinearPolicy(n_dim=n_dim, n_steps=n_steps)
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
            #forecast_T_L_sd = ml_model.forecast_T_L_sd_arr[-1]

            remained_bank_ratio = ml_model.remained_bank_amount/ml_model.thermal_mitigation_bank_size


            #T_L_prev = ml_model.T_L_mu
            #Q_C = ml_model.Q_C[ml_model.t]
            #Q_i = ml_model.Q_i[ml_model.t]
            #cannonsville_storage_pct = ml_model.cannonsville_storage_pct[ml_model.t]
            doc = ml_model.doc[ml_model.t]

            df_t = pd.DataFrame(ml_model.records, index=ml_model.dates)
            df_t = df_t.loc[f"{current_date.year}"]  # Get the data for the current year
            Jadd_t = compute_max_annual_accumulated_degree_days(df_t, col='Tavg_L_mu', threshold=20, return_distribution=False)

            X = np.array([
                minmaxscalers["T_L"].transform(pd.DataFrame([[forecast_T_L_mu]], columns=["T_L"]))[0][0],
                remained_bank_ratio,
                Jadd_t,
                minmaxscalers["doc"].transform(pd.DataFrame([[doc]], columns=["doc"]))[0][0],
                ])

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

    dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
    for date in dates:
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

    Jrel = compute_reliability(df, col="T_L_mu", threshold=24, quantile=0.01, only_summer_period=True, return_distribution=False)
    Jadd = compute_max_annual_accumulated_degree_days(df, col='Tavg_L_mu', threshold=20, only_summer_period=True, return_distribution=False)
    Jtubr = compute_mean_thermal_bank_usage_ratio(df, col='remained_bank_amounts', bank_size=ml_model.thermal_mitigation_bank_size, return_distribution=False, last_date_of_ctrl=(8, 31))

    objs = [-Jrel, Jadd, Jtubr]
    return (objs, df, ml_model)

objs_piecewise, df_piecewise, ml_model_piecewise = eval_func(*params)

#%%

from src.policies import RuleBasedPolicy
def return_dps_func(*params):
    # Initialize the thermal control policy with specific parameters
    policy = RuleBasedPolicy(threshold=24, thermal_release_amount=65)
    # Define the function that will be used for the control algorithm
    def dps_func(model, Q_C, Q_i, cannonsville_storage_pct, current_date):
        # Retrieve the ml_model from the model
        ml_model = model#.ml_model      # Need .ml_model when using the coupled model.
        # Reset the bank amount at the beginning of June
        if current_date.day == 1 and current_date.month == 6:
            ml_model.remained_bank_amount = ml_model.thermal_mitigation_bank_size
        ml_model.update_until(date=current_date)
        # Complete the preparation for thermal control
        # Make a forecast for the current date
        ml_model.forecast(t=ml_model.t, Q_C=None, Q_i=None, cannonsville_storage_pct=None, lead_time=0)
        # Make thermal release decision and record the thermal release
        thermal_release = policy.run(X=ml_model.forecast_T_L_mu_arr)
        thermal_release = min(thermal_release, ml_model.remained_bank_amount)  # Ensure thermal release does not exceed bank size
        return thermal_release
    return dps_func

# Prepare the decision-making function with parameters
params = []
dm_func = return_dps_func(*params)

ml_model = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
ml_model.load_data(database)

dates = pd.date_range(start="1979-01-01", end="2023-12-31", freq='D')
for date in dates:
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

#%%
ml_model_noCtrl = WaterTempLSTMModel(
    model1=pn.models.get() / "TempLSTM/TempLSTM1.yml",
    model2=pn.models.get() / "TempLSTM/TempLSTM2.yml",
    Tavg2Tmax_coefs=pn.models.get() / "TempLSTM/Tavg2Tmax_coefs.json",
    debug=True,
    thermal_mitigation_bank_size=1620 * 3,  # mgd
    )
ml_model_noCtrl.load_data(database)
ml_model_noCtrl.update_until(date=pd.Timestamp('2024-01-01'))  # Update until the end of 2023
df_noCtrl = pd.DataFrame(ml_model_noCtrl.records, index=ml_model.dates)
#%%
df_res = pd.DataFrame(index=df.index)

df_res["historical"] = database["rel_thermal"]
df_res["piecewise"] = df_piecewise["thermal_releases"]
df_res["rule_based"] = df["thermal_releases"]
df_res["Tmax (no_ctrl)"] = df_noCtrl["T_L_mu"]


yr = 2019
for yr in range(2006, 2023):
    df_ = df_res.loc[f"{yr}-5-20":f"{yr}-9-10", :]

    fig, ax = plt.subplots()
    ax.plot(df_["historical"], color='k', lw=2, ls="-", label="historical")
    ax.plot(df_["piecewise"], color='salmon', lw=2, label="piecewise")
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
