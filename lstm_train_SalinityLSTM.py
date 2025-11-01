import pathnavigator
from copy import deepcopy

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.chdir()
from src.model_builder import make_lstm_model, loop_to_train_lstm_models, loop_to_simple_run_lstm_models, loop_to_eval_lstm_models, return_sim_obs_pair
from src.prep_data import data_prep

# from src.model_builder import config_template
config_template = {
    'input_data_file': "data/database/SalinityLSTM_database.csv",
    'x_vars': [],
    'y_vars': [],
    'y_vars_src': [],
    'lag_days': 1,
    'min_date': '1979-01-01',
    'max_date': '2023-12-31',
    'start_date_train': '1979-01-01',
    'end_date_train': '2023-12-31',
    'start_date_val': '2017-01-01',
    'end_date_val': '2017-12-31',
    'start_date_test': '2017-01-01',
    'end_date_test': '2017-12-31',
    'pre_train': True,
    'fine_tune': True,
    'learn_rate_pre': 0.05,
    'learn_rate_fine': 0.05,
    'n_epochs_pre': 50,
    'n_epochs_fine': 350,
    'early_stopping_patience': 50,
    'hidden_units': 16,
    'head_hidden_units': 16,
    'head_n_distr': 1,
    'weight_loss': True,
    'weight_threshold': 80,
    'weight_value': 10,
    'mc_dropout': True,
    'recurrent_dropout_rate': 0,
    'dropout_rate': 0,
    'seq_len': 365,
    'offset': 1.0,
    'seed': 2,
    }
lstm_settings = {
    "model_id": "SalinityLSTM",
    "x_vars": ["Q_Trenton_bc", "Q_Schuylkill_bc", "Q_Trenton_bc_7d_avg", "Q_Schuylkill_bc_7d_avg", "doy"],
    "y_vars": ["saltfront"],
    "y_vars_src": ["saltfront_src"],
    }

#%%
subfolder = "SalinityLSTM"
lstm_config = deepcopy(config_template)
lstm_config.update(lstm_settings)
lstm_config_file = make_lstm_model(subfolder=subfolder, **lstm_config)
_ = data_prep(lstm_config_file, root_dir) # prepare the dataset based on new splits; write to new datafile

#%%
model_ids = ["SalinityLSTM"]
loop_to_train_lstm_models(model_ids, subfolder=subfolder, disable=False, overwrite=True)
lstms = loop_to_simple_run_lstm_models(model_ids, subfolder=subfolder, mode="SalinityLSTM", disable=False, overwrite=True)

#%%
df_metric_train = loop_to_eval_lstm_models(model_ids, subfolder=subfolder, period="train", only_months=None, mode="SalinityLSTM")

print(df_metric_train)
r"""
                 nrmse         r        r2      rmse
model_id                                            
SalinityLSTM  0.095646  0.847963  0.715893  3.410725


Out[4]: '\n                 nrmse         r        r2      rmse\
nmodel_id\nSalinityLSTM  0.084535  0.884838  0.777871  3.014507\n'

                 nrmse         r        r2      rmse
model_id
SalinityLSTM  0.084535  0.884838  0.777871  3.014507
"""
#%%
import pandas as pd
import clt
simple_run = pd.read_csv(pn.models.get("SalinityLSTM/simple_run_SalinityLSTM.csv"), parse_dates=True, index_col=[0])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sim = simple_run["mu_ft"]
obs = simple_run["obs"]
sim, obs = clt.dropna_any(sim, obs)
clt.plots.rmse_over_obs_bins_in_barplot(ax, obs, sim, bins=(60, 90, 5), color="mediumpurple")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.set_ylabel(f"RMSE\n(Overall: {round(clt.metrics.rmse(sim, obs), 2)})")
ax.set_xlabel("$Saltfront$ (RM)")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
sim = simple_run["mu_ft"]
obs = simple_run["obs"]
ax.scatter(obs, sim, s=1)
plt.tight_layout()
plt.show()
#%%
fig, ax = plt.subplots()
sim = simple_run["mu_ft"]
obs = simple_run["obs"]
df = pd.DataFrame({"obs": obs, "sim": sim}, index=simple_run.index)
year = 2007
df = df.loc[f"{year}-01-01":f"{year}-12-31", :]
ax.plot(df["obs"], ls='None', marker='o', color="k", alpha=0.2, ms=3, label="obs")
ax.plot(df["sim"], color='mediumpurple', lw=1, label="sim")
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.legend(frameon=False)
custom_ticks = pd.to_datetime([f"{year}-01-01", f"{year}-04-01", f"{year}-07-01", f"{year}-10-01", f"{year}-12-31"])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])  # show only month/day

ax.set_xlim([df.index[0], df.index[-1]])
ax.set_ylim([60, 88])
ax.set_ylabel("$Saltfront$ (RM)")
ax.set_xlabel(f"Date ({year})")

plt.tight_layout()
plt.show()


#%% Compare across weights 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse_over_obs_bins_in_barplot(
    ax, obs, sim, bins=10, align='center',
    labels=None, colors=None, empty_bin_value=np.nan, **kwargs
):
    """
    Plot RMSE against observed values in bins. If `sim` is a pandas DataFrame,
    draw grouped bars (one bar per column) within each bin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    obs : array-like
        Observed values.
    sim : array-like or pandas.DataFrame
        Simulated values. If DataFrame, columns are separate series to compare.
    bins : int or tuple
        Either number of bins (int), or a tuple (start, end, step).
    align : {'center','edge'}
        Bar alignment (default 'center').
    labels : list of str, optional
        Labels for legend when `sim` is DataFrame. Defaults to sim.columns.
    colors : list of color, optional
        Colors per series when `sim` is DataFrame.
    empty_bin_value : float
        Value to use when a bin has no samples (default np.nan).
    **kwargs :
        Passed to `ax.bar` (e.g., alpha).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'bin_edges': ndarray of bin edges
        - 'bin_centers': ndarray of bin centers
        - 'rmse': ndarray (S x B) for DataFrame input, or (B,) for 1-D input
        - 'handles': list of BarContainers (for legend), else []
    """
    obs = np.asarray(obs)

    # Build bin edges
    if isinstance(bins, tuple) and len(bins) == 3:
        start, end, step = bins
        bin_edges = np.arange(start, end + step, step)
    else:
        bin_edges = np.linspace(np.nanmin(obs), np.nanmax(obs), num=int(bins) + 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Digitize observed into bins
    bin_idx = np.digitize(obs, bin_edges) - 1
    # clamp to [0, n_bins-1]
    bin_idx = np.clip(bin_idx, 0, len(bin_centers) - 1)

    # Helper to compute per-bin RMSE for one simulated series
    def per_bin_rmse(sim_vec):
        sim_vec = np.asarray(sim_vec)
        se = (obs - sim_vec) ** 2
        rmse = np.empty(len(bin_centers), dtype=float)
        for i in range(len(bin_centers)):
            m = bin_idx == i
            if np.any(m):
                rmse[i] = np.sqrt(np.nanmean(se[m]))
            else:
                rmse[i] = empty_bin_value
        return rmse

    handles = []

    if isinstance(sim, pd.DataFrame):
        series_list = [sim[c].to_numpy() for c in sim.columns]
        n_series = len(series_list)
        labels = labels if labels is not None else list(sim.columns)

        # Bar width per series inside each bin (with a small group padding)
        group_pad = 0.1
        bar_widths = (bin_widths * (1 - group_pad)) / max(n_series, 1)

        rmse_mat = np.vstack([per_bin_rmse(sv) for sv in series_list])  # (S, B)

        for j in range(n_series):
            overall_rmse = round(clt.metrics.rmse(series_list[j], obs), 2)
            # Offset each series j around the bin center
            offsets = (-0.5 * (n_series - 1) + j) * bar_widths
            # Compute x positions per bin
            x = bin_centers + offsets

            h = ax.bar(
                x, rmse_mat[j],
                width=bar_widths,
                align='center',
                label=labels[j] + f" (RMSE = {overall_rmse})",
                color=(None if colors is None else colors[j] if j < len(colors) else None),
                **kwargs
            )
            handles.append(h)

        ax.set_xlabel("Observed-value bins")
        ax.set_ylabel("RMSE")
        ax.legend(
            loc='upper center',          # position on top
            bbox_to_anchor=(0.5, 1.18),  # center it horizontally, move slightly above the plot
            ncol=2,                      # two columns
            frameon=False,               # no frame
            fontsize=10                  # optional: adjust font size
        )
        out_rmse = rmse_mat

    else:
        # 1D case
        rmse_vals = per_bin_rmse(sim)
        # default width ~ 90% of bin
        widths = bin_widths * 0.9
        ax.bar(bin_centers, rmse_vals, width=widths, align='center', **kwargs)
        ax.set_xlabel("Observed-value bins")
        ax.set_ylabel("RMSE")
        out_rmse = rmse_vals

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'rmse': out_rmse,
        'handles': handles
    }

import matplotlib.pyplot as plt

sim_df = pd.DataFrame()
for w in [1, 2, 5, 10]:
    simple_run = pd.read_csv(pn.models.get(f"SalinityLSTM_w{w}/simple_run_SalinityLSTM.csv"), parse_dates=True, index_col=[0])
    sim = simple_run["mu_ft"]
    obs = simple_run["obs"]
    sim, obs = clt.dropna_any(sim, obs)
    sim_df[f"weight {w}"] = sim

fig, ax = plt.subplots()

rmse_over_obs_bins_in_barplot(ax, obs, sim_df, bins=(60, 90, 5), colors=["#4477AA", "#EE6677", "#228833", "#CCBB44"])
ax.grid(True, axis='y', lw=0.3, ls="--")
ax.set_ylabel(f"RMSE\n(Overall: {round(clt.metrics.rmse(sim, obs), 2)})")
ax.set_xlabel("$Saltfront$ (RM)")
plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "SalinityLSTM_rmse_barplot_comparising_weights.jpg", dpi=500)
plt.show()







# #%%
# import matplotlib.pyplot as plt
# import clt
# pairs = return_sim_obs_pair(lstms, period="train", only_months=None, mode="SalinityLSTM", disable=False)

# #%%
# sim, obs = pairs['SalinityLSTM']
# sim, obs = clt.dropna_any(sim, obs)
# fig, ax = plt.subplots()
# clt.plots.scatter(ax, x=obs, y=sim, s=1, alpha=0.5,)
# # ax.scatter(x=obs, y=sim, s=1, alpha=0.5)
# # clt.ax.add_45_degree_ref_line(ax)
# # clt.ax.add_linear_regr_line(ax, sim, obs,)
# ax.set_xlabel("Observed (RM)")
# ax.set_ylabel("Predicted (RM)")
# ax.legend()
# plt.tight_layout()
# plt.show()

# #%%
# sim, obs = pairs['SalinityLSTM']
# for y in range(1964, 2024):
#     fig, ax = plt.subplots()
#     ax.plot(obs[f"{y}":f"{y}"], c="k", alpha=0.8, label="Observed")
#     ax.plot(sim[f"{y}":f"{y}"], c="b", alpha=0.8, label="Predicted")
#     ax.set_ylim([40, 100])
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Salt front location (RM)")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()



#%% Check new old 
# import pandas as pd

# new = pd.read_csv(pn.data.raw.get() / "pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])["1979":"2024"][["flow_delTrenton", "flow_outletSchuylkill"]]
# new.columns = ["Q_Trenton_bc", "Q_Schuylkill_bc"]

# old = pd.read_csv("/Users/cl/Downloads/pywrdrb_pub_nhmv10_BC_withObsScaled_flow_and_storage.csv", parse_dates=True, index_col=["date"])["1979":"2024"][["flow_delTrenton", "flow_outletSchuylkill"]]
# old.columns = ["Q_Trenton_bc", "Q_Schuylkill_bc"]

# db = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), parse_dates=True, index_col=["date"])["1979":"2024"][["Q_Trenton_bc", "Q_Schuylkill_bc"]]


# gauge = "Q_Trenton_bc"
# gauge = "Q_Schuylkill_bc"

# df_compare = pd.concat([
#     new[gauge].rename("new"),
#     old[gauge].rename("old"),
#     db[gauge].rename("db")
# ], axis=1).dropna()

# # Correlation matrix
# corr = df_compare.corr()
# print(corr)

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].scatter(df_compare["old"], df_compare["new"], s=5)
# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[0].set_xlabel(f"Old {gauge}")
# ax[0].set_ylabel(f"New {gauge}")
# ax[0].set_title("New vs Old")

# # --- New vs DB ---
# ax[1].scatter(df_compare["db"], df_compare["new"], s=5)
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
# ax[1].set_xlabel(f"DB {gauge}")
# ax[1].set_ylabel(f"New {gauge}")
# ax[1].set_title("New vs DB")

# plt.tight_layout()
# plt.show()







