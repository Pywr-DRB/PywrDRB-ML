import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

def plot_parallel_coords_with_kde(
        ax, df, columns, add_background_grey_lines=True,
        dict_kde_dfs=None, dict_colorlines_dfs=None,
        soln_labels=None, objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "-Jrel", "Jadd"],
        ideal_direction='top',
        fontsize=12, kde_scale=0.12,
        cmap_kdes={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'},
        cmap_lines={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'},
        cmap_highlights={
            'No control': 'k', 'Rule-based': '#E41A1C',
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
        if j == 1:
            top_label = f"{round(tops[j]*(-100), 0)}%"
            bottom_label = f"{round(bottoms[j]*(-100), 0)}%"
        else:
            top_label = str(round(tops[j], 1))
            bottom_label = str(round(bottoms[j], 1))
        
        ax.annotate(top_label, [x_spacing[j], 1.02], ha='center', va='bottom', fontsize=fontsize)
        
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
    #ax.legend(frameon=False, loc="center right")
    line_legend = ax.legend(frameon=False, bbox_to_anchor=(0.72, 0.35), loc="lower left", fontsize=fontsize)

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
                
    # Add KDE legend

    # KDE shading
    if dict_kde_dfs is not None:
        for i, key in enumerate(dict_kde_dfs):
            dff = dict_kde_dfs_scaled[key]
            for o, obj in enumerate(columns):  # [1:]
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
                                 where=(x > 0.00005), lw=1, alpha=0.6, zorder=4,
                                 fc=cmap_kdes[key], ec='k')

        # ---------- KDE legend (second legend) ----------
        kde_handles = []
        kde_labels = ["Standard bank\n(0, 1x]", "Medium bank increase\n(1, 2x]", "High bank increase\n(2, 3x]"]
        for key, color in cmap_kdes.items():
            # You can customize these labels as you like
            label = kde_labels[key] #f"Group {key}"
            kde_handles.append(
                mpatches.Patch(facecolor=color, edgecolor='k', alpha=0.6, label=label)
            )

        kde_legend = ax.legend(handles=kde_handles,
                               bbox_to_anchor=(0.722, -0.1), 
                               loc="lower left",
                               frameon=False, 
                               fontsize=fontsize)

        # re-add the original line legend so both appear
        ax.add_artist(line_legend)
    
    
    ax.set_xlim([x_spacing[0] - 0.3, x_spacing[-1] + 0.3])
    return ax

# Scatter plot
def kde_scatter_plot(ax, df, base_color="#1b9e77", df_ref=None, highlight=True, idx_list=[], fontsize=12):
    # Convert to RGB
    r, g, b = mcolors.to_rgb(base_color)

    # Create a MUCH lighter version (toward white)
    very_light_color = (1 - (1 - r) * 0.05,
                        1 - (1 - g) * 0.05,
                        1 - (1 - b) * 0.05)

    # Build a strong gradient colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "strong_green",
        [very_light_color, base_color]
    )
    
    sc = ax.scatter(
        df["Jadd"],
        df["-Jrel"],
        c=df["Jtubr"],
        cmap=cmap,
        s=50,
        edgecolor="black",
        zorder=13
    )
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Jtubr", fontsize=fontsize)
    
    ax.set_xlabel("Jadd", fontsize=fontsize)
    ax.set_ylabel("Jrel", fontsize=fontsize)
    yticks = ax.get_yticks()
    
    #ax.set_title("Custom Color Gradient: Light → Dark (#1b9e77)")
    ax.set_xlim(30, 105)
    ax.set_ylim(-1.04, -0.3)
    
    yticks = np.linspace(-0.3, -1.0, 8)   # adjust number of ticks as needed
    ax.set_yticks(yticks)
    
    # Example: convert to percent
    ytick_labels = [f"{y * -100:.0f}%" for y in yticks]
    ax.set_yticklabels(ytick_labels)
    #ax.set_yticklabels([f"{y * -100:.0f}%" for y in yticks])
    
    
    # Extract original x/y
    x = df["Jadd"].values
    y = df["-Jrel"].values
    z = df["Jtubr"].values
    labels = df["label"]
    
    
    if df_ref is None:
        x_ref = None
        y_ref = None
        z_ref = None
    else:
        x_ref = df_ref["Jadd"].values
        y_ref = df_ref["-Jrel"].values
        z_ref = df_ref["Jtubr"].values
    
    # --- Normalize ONLY for selecting the closest point ---
    def normalize(arr, arr_ref=None):
        if arr_ref is None:
            arr_ref = arr
        amin, amax = arr_ref.min(), arr_ref.max()
        return (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr)
    
    x_norm = normalize(x, x_ref)
    y_norm = normalize(y, y_ref)
    z_norm = normalize(z, z_ref)
    
    # distance to lower-left corner (0,0) in normalized space
    dist = np.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
    idx_closest = dist.argmin()
    
    # original values of the chosen point
    x_closest = x[idx_closest]
    y_closest = y[idx_closest]
    
    print(df.iloc[idx_closest, -1])
    
    # --- Highlight on the existing figure/axes ---
    if highlight:
        ax.scatter(
            x_closest,
            y_closest,
            s=100,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            zorder=10,
        )
    
    for i, c in idx_list: 
        idx = np.where(labels == i)[0][0]
        x_ = x[idx]
        y_ = y[idx]
        ax.scatter(
            x_,
            y_,
            s=100,
            facecolors="none",
            edgecolors=c,
            linewidths=4,
            zorder=10,
        )
    
    # Ideal point
    # ax.scatter(
    #     31,
    #     -1,
    #     s=200,              # size of the star
    #     marker="*",         # star marker
    #     color="yellow",     # fill color
    #     edgecolors="black", # optional: outline
    #     linewidths=1.2,
    #     zorder=11           # ensure it appears on top
    # )
    
    ax.annotate(
        "*",                   # the star symbol
        xy=(31, -1.03),           # point location
        xytext=(31, -1.06),       # same location (no offset)
        fontsize=28,           # size of the star
        color="yellow",        # star color
        ha="center",
        va="center",
        weight="bold",
        transform=ax.transAxes,
        path_effects=[         # optional: black outline
            path_effects.Stroke(linewidth=1.5, foreground='black'),
            path_effects.Normal()
        ],
        zorder=12
    )
    
    plt.tight_layout()
    return ax

#% Combine the plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(9, 6))

# Grid: 2 rows × 3 columns
gs = gridspec.GridSpec(2, 3, height_ratios=[2.2, 1])

# --- Top large panel (span all 3 columns) ---
ax_top = fig.add_subplot(gs[0, :])
names = ["No control", "Fixed-release\n(baseline)", "historic", #\n(2010-2023)
         "Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", 
         "Enlarged bank", "XXX"]
toC = 103.16

highlight_rows = [
    [0,        -0.303, 1*toC,        names[0]], # "No control"
    [0.8025, -0.5913, 0.7798*toC,   names[1]], # "Rule-based"
    [0.9765,   -0.9981, 0.7351*toC,   names[3]], # "RBF-better Jrel" 28
    [0.9846,   -0.5485, 0.7123*toC,   names[4]], # "RBF-better Jadd" 151
    [1.9905,   -0.9989, 51.6317,   names[5]], # 114
    #[1.4862,   -0.9989, 0.6008*toC,   names[5]], # 91 (only xyz)
    #[2.5005,   -0.9987, 0.4256*toC,   names[6]], # 67 (only xyz)
    #[1.9905,   -0.9989, 0.5005*toC,   names[5]], # 114 (only xy)
    #[2.9883,   -0.9992, 0.3629*toC,   names[6]], # 70 (only xy)
#    [2.8944,   -0.9995, 0.3741*toC,   names[5]], # "RBF-best Jrel" 63
#    [2.9655,   -0.7826, 0.3615*toC,   names[6]], # "RBF-best Jadd" 106
]
df_highlight = pd.DataFrame(highlight_rows, columns=["Jtubr", "-Jrel", "Jadd", "label"])

cmap_highlights={
    names[0]: 'k', 
    names[1]: '#E41A1C',
    names[2]: "blue",
    names[3]: "limegreen",
    names[4]: "cyan", #"aquamarine",
    names[5]: "saddlebrown",
    names[6]: "peru"
    }
ls_highlights={
    names[2]: ":",
    names[5]: "--",
    names[6]: "--"
    }
zorder_highlights={
    names[0]: 0, names[1]: 1,
    names[2]: 2,
    names[3]: 8,
    names[4]: 9,
    names[5]: 4,
    names[6]: 5
    }

policy ="GaussianRBFPolicy"
#job_id = "139181" 
job_id = "143990"

df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))
df_ref = df_ref[['obj3', 'obj1', 'obj2']]
df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
df_ref["Jadd"] /= 0.7984
df_ref["Jadd"] *= toC 
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

plot_parallel_coords_with_kde(
    ax=ax_top, df=df, columns=["Jtubr", "-Jrel", "Jadd"],
    dict_kde_dfs=dict_kde_dfs,
    #dict_colorlines_dfs=dict_colorlines_dfs,
    soln_labels=df_highlight["label"].to_list(),
    objmins=None, objmaxs=None,
    axes_labels=["Thermal\nbank\nusage\n(Jtubr; --)", "Thermal\ncontrol\nreliability\n(Jrel; %)", "Annual\ncumulative\nheat exposure\n(Jadd; °C·day)"],
    ideal_direction='bottom', fontsize=12, kde_scale=0.05,
    cmap_highlights=cmap_highlights,
    ls_highlights=ls_highlights,
    zorder_highlights=zorder_highlights
    )

# --- Bottom three small panels ---
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))
df_ref = df_ref[['obj3', 'obj1', 'obj2']]
df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
df_ref["Jadd"] /= 0.7984
df_ref["Jadd"] *= toC 
df_ref["Jtubr"] *= 3
df_ref["label"] = df_ref.index

cmap_kdes={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}

ax1 = fig.add_subplot(gs[1, 0])
df = df_ref[df_ref["Jtubr"] <=1 ]
kde_scatter_plot(ax1, df, base_color=cmap_kdes[0], highlight=False, idx_list=[(28, "limegreen"), (151, "cyan")])
ax1.axvline(80.44, color="red", ls=":", lw=1)
ax1.axhline(-0.5913, color="red", ls=":", lw=1)

ax2 = fig.add_subplot(gs[1, 1])
df = df_ref[(df_ref["Jtubr"] > 1) & (df_ref["Jtubr"] <= 2)]
kde_scatter_plot(ax2, df, base_color=cmap_kdes[1], highlight=False, idx_list=[(114, "saddlebrown")])
ax2.axvline(80.44, color="red", ls=":", lw=1)
ax2.axhline(-0.5913, color="red", ls=":", lw=1)

ax3 = fig.add_subplot(gs[1, 2])
df = df_ref[(df_ref["Jtubr"] > 2) & (df_ref["Jtubr"] <= 3)]
kde_scatter_plot(ax3, df, base_color=cmap_kdes[2], highlight=False)
ax3.axvline(80.44, color="red", ls=":", lw=1)
ax3.axhline(-0.5913, color="red", ls=":", lw=1)

# Example usage:
#ax_top.set_title()
fontsize = 12
ax1.set_title("Standard bank", fontsize=fontsize)
ax2.set_title("Medium bank increase", fontsize=fontsize)
ax3.set_title("High bank increase", fontsize=fontsize)


ax_top.text(
    -0.1, 1,            # position (x, y) in axis coords
    "a)",
    transform=ax_top.transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="left"
)
axes = [ax1, ax2, ax3]
labels = ["b)", "c)", "d)"]
for ax, label in zip(axes, labels):
    ax.text(
        -0.55, 1.2,            # position (x, y) in axis coords
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )
plt.tight_layout()
plt.show()


#%% release
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])
df_hist = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historic"] = database["rel_thermal"]
df_res["Fixed-release\n(baseline)"] = df_rulebased["thermal_releases"]
df_res["Tmax (No control)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (Rule-based)"] = df_rulebased["T_L_mu"]
df_res["Tmax (historic)"] = df_hist["T_L_mu"] #database["QobsTmax_T_L"]

for i, sol_idx in enumerate([28, 151, 114]):
    df_rbf = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_{sol_idx}.csv", parse_dates=True, index_col=[0])
    df_res[f"{names[3+i]}"] = df_rbf["thermal_releases"]
    df_res[f"Tmax ({names[3+i]})"] = df_rbf["T_L_mu"]
#%%
import matplotlib.gridspec as gridspec

yr = 2020
#for yr in range(1979, 2024):
mgd2m3 = 378541/10**6
df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

colors = {
    'No control': 'k',
    "Fixed-release\n(baseline)": '#E41A1C',
    'historic': "blue",
    names[3]: "limegreen",
    names[4]: "cyan",
    names[5]: "saddlebrown",
    names[6]: "peru"
    }

release_names = ["Fixed-release\n(baseline)"] + names[3:-1] + ['historic']  # All release types you want to plot
n_release_types = len(release_names)

fig = plt.figure(figsize=(8, 1 * (n_release_types + 1)))
gs = gridspec.GridSpec(7, 1, height_ratios=[1, 0.15, 0.3, 0.3, 0.3, 0.3, 0.3], hspace=0)

# Top subplot: Tmax
ax1 = fig.add_subplot(gs[0])
ax1.plot(df_["Tmax (No control)"], ls="-", color=colors['No control'], label="Tmax (No control)")
ax1.plot(df_["Tmax (Rule-based)"], ls="-", color=colors["Fixed-release\n(baseline)"], label="Tmax (baseline)")
#ax1.plot(df_["Tmax (historic)"], ls="-", color=colors['historic'], label="Tmax (historic)")
for rbf_name in names[3:-1]:
    ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color=colors[f"{rbf_name}"], label=f"Tmax ({rbf_name})")
ax1.axhline(24, lw=1, c="k", ls=":")
ax1.set_ylabel("$T_{max}$ (°C)")
ax1.grid(True, axis='y', lw=0.3, ls="--")
#ax1.tick_params(axis='x', direction='in')
ax1.set_ylim([18, 27])
ax1.set_yticks([20, 24])
ax1.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
custom_ticks = pd.to_datetime([
    f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15",
    f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"
])
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])

ax1.text(
    -0.12, 1, "(a)",
    transform=ax1.transAxes,
    ha="left", va="top",
    fontsize=12, fontweight="bold"
)

# Bottom subplots: releases
for i, release_name in enumerate(release_names):
    ax = fig.add_subplot(gs[2+i])
    
    if release_name == 'historic':
        non_zero_mask = df_["historic"] != 0
        x_vals = df_.index[non_zero_mask]
        y_vals = df_["historic"][non_zero_mask]
        if not x_vals.empty:
            markerline, stemlines, baseline = ax.stem(
                x_vals, y_vals*mgd2m3,
                linefmt="k-", markerfmt="ko", basefmt=" ", label="Obs"
            )
            markerline.set_color(colors['historic'])
            stemlines.set_color(colors['historic'])
            plt.setp(stemlines, lw=1, zorder=80)
            plt.setp(markerline, ms=4, zorder=80)
            plt.setp(baseline, visible=False)
            ax.set_xlim(df_.index[0], df_.index[-1])
        else:
            ax.plot([], [], marker='o', color='k', linestyle='None', label="historic")
    else:
        ax.bar(df_.index, df_[release_name]*mgd2m3, width=1.0, color=colors[release_name], label=release_name, alpha=0.6, zorder=4)
    ax.grid(True, axis='y', lw=0.3, ls="--")
    #ax.set_ylabel(f"{release_name}\nThermal release (mgd)")
    ax.set_ylim([0, 200*mgd2m3])
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='x', direction='in')
    ax.set_xticklabels([])
    
    if i == 0:
        ax.text(
            -0.12, 1, "(b)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold"
        )
    
    if i == 2:
        ax.set_ylabel("Thermal release ($10^6 m^3/day$)        ")

ax.set_xlabel(f"Date (Year={yr})")
custom_ticks = pd.to_datetime([
    f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15",
    f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"
])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])

plt.tight_layout()
pn.outputs.mkdir(f"dps_{policy}_{job_id}/figures/RBFs")
clt.fig.savefig(fig, pn.figures.get("attemp1") / f"RBFs_thermal_release_{yr}.jpg")
plt.show()



























#%%


df_highlight = pd.DataFrame()

names = ["No control", "Rule-based", "historic", #\n(2010-2023)
         "RBF-better Jrel", "RBF-better Jadd", "RBF-best Jrel", "RBF-best Jadd"]

names = ["No control", "Fixed-release", "historic", #\n(2010-2023)
         "Standard bank,\nmax Jrel", "Standard bank,\nmin Jadd", "RBF-best Jrel", "RBF-best Jadd"]
                         
# highlight_rows = [
#     #[0,        -0.2018, 1,        names[0]], # "No control"
#     #[0.4681*3, -0.516, 0.7794,   names[1]], # "Rule-based"
#     [0,        -0.2211, 1,        names[0]], # "No control"
#     [1, -0.5577, 0.7834,   names[1]], # "Rule-based"
#     #[1, -0,5236, ]
#     #[0.4994*3, -0.3312, 0.7349,    names[2]], # "historic\n(2010-2023)"
#     #[0.4994*3, -0.4353, 1.2668,    names[2]], # "historic\n(2010-2023)"
#     [1.0095,   -0.7831, 0.78502,   names[3]], # "RBF-better Jrel" 107
#     [1.0197,   -0.5578, 0.772433,   names[4]], # "RBF-better Jadd" 77
#     [2.4696,   -0.9992, 0.496635,   names[5]], # "RBF-best Jrel" 57
#     [3,   -0.9979, 0.39818,   names[6]], # "RBF-best Jadd" 4
# ]

toC = 103.16

highlight_rows = [
    [0,        -0.303, 1*toC,        names[0]], # "No control"
    [0.8025, -0.5913, 0.7798*toC,   names[1]], # "Rule-based"
    [0.9765,   -0.9981, 0.7351*toC,   names[3]], # "RBF-better Jrel" 28
    [0.9846,   -0.5485, 0.7123*toC,   names[4]], # "RBF-better Jadd" 151
    [2.8944,   -0.9995, 0.3741*toC,   names[5]], # "RBF-best Jrel" 63
#    [2.9655,   -0.7826, 0.3615*toC,   names[6]], # "RBF-best Jadd" 106
]
df_highlight = pd.DataFrame(highlight_rows, columns=["Jtubr", "-Jrel", "Jadd", "label"])

cmap_highlights={
    names[0]: 'k', 
    names[1]: '#E41A1C',
    names[2]: "blue",
    names[3]: "limegreen",
    names[4]: "cyan", #"aquamarine",
    names[5]: "saddlebrown",
    names[6]: "peru"
    }
ls_highlights={
    names[2]: ":",
    names[5]: "--",
    names[6]: "--"
    }
zorder_highlights={
    names[0]: 0, names[1]: 1,
    names[2]: 2,
    names[3]: 8,
    names[4]: 9,
    names[5]: 4,
    names[6]: 5
    }

policy ="GaussianRBFPolicy"
#job_id = "139181" 
job_id = "143990"

df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))
df_ref = df_ref[['obj3', 'obj1', 'obj2']]
df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
df_ref["Jadd"] /= 0.7984
df_ref["Jadd"] *= toC 
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
    #dict_colorlines_dfs=dict_colorlines_dfs,
    soln_labels=df_highlight["label"].to_list(),
    objmins=None, objmaxs=None,
    axes_labels=["Thermal\nbank\nusage\n(Jtubr; %)", "Thermal\nsatisfication\nfrequency\n(Jrel; --)", "Annual\ncumulative\nheat exposure\n(Jadd; °C·day)"],
    ideal_direction='bottom', fontsize=10, kde_scale=0.05,
    cmap_highlights=cmap_highlights,
    ls_highlights=ls_highlights,
    zorder_highlights=zorder_highlights
    )
#ax.set_title(policy)
plt.tight_layout()
#clt.fig.savefig(fig, pn.figures.get(f"attemp1") / f"RBFs_tradeoffs.jpg")
plt.show()

#%% Scatter plot
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))
df_ref = df_ref[['obj3', 'obj1', 'obj2']]
df_ref.columns = ["Jtubr", "-Jrel", "Jadd"]
df_ref["Jadd"] /= 0.7984
df_ref["Jadd"] *= toC 
df_ref["Jtubr"] *= 3
df_ref["label"] = df_ref.index

df = df_ref[df_ref["Jtubr"] <=1 ]
# Base color



cmap_kdes={0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
fig, ax = plt.subplots(figsize=(5,4))
df = df_ref[df_ref["Jtubr"] <=1 ]
ax = kde_scatter_plot(ax, df, base_color=cmap_kdes[0])#, df_ref=df_ref)
plt.show()

fig, ax = plt.subplots(figsize=(5,4))
df = df_ref[(df_ref["Jtubr"] > 1) & (df_ref["Jtubr"] <= 2)]
ax = kde_scatter_plot(ax, df, base_color=cmap_kdes[1])#, df_ref=df_ref)
plt.show()

fig, ax = plt.subplots(figsize=(5,4))
df = df_ref[(df_ref["Jtubr"] > 2) & (df_ref["Jtubr"] <= 3)]
ax = kde_scatter_plot(ax, df, base_color=cmap_kdes[2])#, df_ref=df_ref)
plt.show()


#%% Compare thermal release
import joblib
from src.lstm_model import WaterTempLSTMModel
from src.objectives import compute_reliability, compute_max_annual_accumulated_degree_days, compute_max_thermal_bank_usage_ratio, compute_mean_thermal_bank_usage_ratio
from tqdm import tqdm
from src.policies import GaussianRBFPolicy
df_ref = clt.borg.read_ref(pn.outputs.get(f"dps_{policy}_{job_id}/borg.ref"))

database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])
df_hist = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historic"] = database["rel_thermal"]
df_res["Rule-based"] = df_rulebased["thermal_releases"]
df_res["Tmax (No control)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (Rule-based)"] = df_rulebased["T_L_mu"]
df_res["Tmax (hist)"] = df_hist["T_L_mu"]

df_objs = pd.DataFrame()

#for i, sol_idx in enumerate([28, 151, 91, 67]):#63, 106]):
for sol_idx in tqdm(list(df_ref.index)): #enumerate([28, 151, 91, 67]):#63, 106]):
    params = df_ref.iloc[sol_idx, :-3]
    n_dim = 3  # Number of dimensions for the policy
    n_basis = n_dim + 1  # Number of basis functions for the Gaussian RBF policy

    def eval_func(*params):
        database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
        # Initialize the thermal control policy with specific parameters
        policy = GaussianRBFPolicy(n_dim=n_dim, n_basis=n_basis)
        #params = policy.gen_params(seed=42)[0]
        minmaxscalers = joblib.load(pn.get("thermal_ctrl_decoupled") / "minmaxscalers.gz")
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
        for t, date in tqdm(enumerate(dates), desc="Running thermal control policy", disable=True):
            Q_C = None  # Placeholder for controlled release
            Q_i = None  # Placeholder for inflow
            cannonsville_storage_pct = None        

            if date.month in [6, 7, 8]:
                thermal_release = dm_func(ml_model, Q_C, Q_i, cannonsville_storage_pct, date)
            else:
                thermal_release = 0

            # Update data in the ml_model for the next step(s) model update.
            #t = ml_model.t # 
            acc_thermal_release = ml_model.thermal_mitigation_bank_size - ml_model.remained_bank_amount
            ml_model.Q_C[t] += thermal_release
            ml_model.cannonsville_storage_pct[t] = (ml_model.cannonsville_storage_pct[t] * 95700/100 - acc_thermal_release)/ 95700 * 100  # Update the storage percentage based on the thermal release

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

    # df_res[f"{names[3+i]}"] = df_rbf["thermal_releases"]
    # df_res[f"Tmax ({names[3+i]})"] = df_rbf["T_L_mu"]
    # df_objs[f"{names[3+i]}"] = objs
    
    df_res[sol_idx] = df_rbf["thermal_releases"]
    df_res[f"Tmax {sol_idx}"] = df_rbf["T_L_mu"]
    df_objs[f"{sol_idx}"] = objs

    #df_rbf.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_{names[3+i]}_{sol_idx}.csv")
    df_rbf.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_{sol_idx}.csv")

df_res.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_res.csv")
df_objs.to_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / "df_objs.csv")

# 159 [0.9999, -0.3778, 0.742, 0.9665999999999999]
# 24 (better Jadd): Out[95]: [0.9813, -0.356, 0.7558, 0.9606]
# 115 (better Jrel):         [1.0023, -0.456, 0.8904, 0.1869]


#%%
df1 = df_ref[df_ref["Jtubr"] <=1 ] #53 153 10
df2 = df_ref[(df_ref["Jtubr"] > 1) & (df_ref["Jtubr"] <= 2)] #86, 143
df3 = df_ref[(df_ref["Jtubr"] > 2) & (df_ref["Jtubr"] <= 3)] #67 106 24

yr = 2020
#for yr in range(1979, 2024):
df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]
df_.loc[:, df1.index].plot(legend=False, color="g", alpha=0.2, lw=1)
df_.loc[:, df2.index].plot(legend=False, color="orange", alpha=0.2, lw=1)
df_.loc[:, df3.index].plot(legend=False, color="purple", alpha=0.2, lw=1)
#%%
database = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01': '2023-12-31']
df_rulebased = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_rulebased.csv", parse_dates=True, index_col=[0])
df_noCtrl = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_noCtrl.csv", parse_dates=True, index_col=[0])
df_hist = pd.read_csv(pn.data.baseline_ctrl_lstm.get() / "df_hist.csv", parse_dates=True, index_col=[0])

df_res = pd.DataFrame(index=df_rulebased.index)
df_res["historic"] = database["rel_thermal"]
df_res["Rule-based"] = df_rulebased["thermal_releases"]
df_res["Tmax (No control)"] = df_noCtrl["T_L_mu"]
df_res["Tmax (Rule-based)"] = df_rulebased["T_L_mu"]
df_res["Tmax (historic)"] = df_hist["T_L_mu"] #database["QobsTmax_T_L"]

for i, sol_idx in enumerate([28, 151, 63, 106]):
    df_rbf = pd.read_csv(pn.outputs.get(f"dps_{policy}_{job_id}") / f"df_{names[3+i]}_{sol_idx}.csv", parse_dates=True, index_col=[0])
    df_res[f"{names[3+i]}"] = df_rbf["thermal_releases"]
    df_res[f"Tmax ({names[3+i]})"] = df_rbf["T_L_mu"]
#%%
import matplotlib.gridspec as gridspec

yr = 2020
#for yr in range(1979, 2024):
df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

colors = {
    'No control': 'k',
    'Rule-based': '#E41A1C',
    'historic': "blue",
    names[3]: "limegreen",
    names[4]: "cyan",
    names[5]: "saddlebrown",
    names[6]: "peru"
    }

release_names = ['Rule-based'] + names[3:] + ['historic']  # All release types you want to plot
n_release_types = len(release_names)

fig = plt.figure(figsize=(8, 1 * (n_release_types + 1)))
gs = gridspec.GridSpec(8, 1, height_ratios=[1, 0.15, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], hspace=0)

# Top subplot: Tmax
ax1 = fig.add_subplot(gs[0])
ax1.plot(df_["Tmax (No control)"], ls="-", color=colors['No control'], label="Tmax (No control)")
ax1.plot(df_["Tmax (Rule-based)"], ls="-", color=colors['Rule-based'], label="Tmax (Rule-based)")
#ax1.plot(df_["Tmax (historic)"], ls="-", color=colors['historic'], label="Tmax (historic)")
for rbf_name in names[3:]:
    ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color=colors[f"{rbf_name}"], label=f"Tmax ({rbf_name})")
ax1.axhline(24, lw=1, c="k", ls=":")
ax1.set_ylabel("$T_{max}$ (°C)")
ax1.grid(True, axis='y', lw=0.3, ls="--")
#ax1.tick_params(axis='x', direction='in')
ax1.set_ylim([18, 27])
ax1.set_yticks([20, 24])
ax1.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
custom_ticks = pd.to_datetime([
    f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15",
    f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"
])
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])

ax1.text(
    -0.12, 1, "(a)",
    transform=ax1.transAxes,
    ha="left", va="top",
    fontsize=12, fontweight="bold"
)

# Bottom subplots: releases
for i, release_name in enumerate(release_names):
    ax = fig.add_subplot(gs[2+i])
    
    if release_name == 'historic':
        non_zero_mask = df_["historic"] != 0
        x_vals = df_.index[non_zero_mask]
        y_vals = df_["historic"][non_zero_mask]
        if not x_vals.empty:
            markerline, stemlines, baseline = ax.stem(
                x_vals, y_vals,
                linefmt="k-", markerfmt="ko", basefmt=" ", label="Obs"
            )
            markerline.set_color(colors['historic'])
            stemlines.set_color(colors['historic'])
            plt.setp(stemlines, lw=1, zorder=80)
            plt.setp(markerline, ms=4, zorder=80)
            plt.setp(baseline, visible=False)
            ax.set_xlim(df_.index[0], df_.index[-1])
        else:
            ax.plot([], [], marker='o', color='k', linestyle='None', label="historic")
    else:
        ax.bar(df_.index, df_[release_name], width=1.0, color=colors[release_name], label=release_name, alpha=0.6, zorder=4)
    ax.grid(True, axis='y', lw=0.3, ls="--")
    #ax.set_ylabel(f"{release_name}\nThermal release (mgd)")
    ax.set_ylim([0, 160])
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='x', direction='in')
    ax.set_xticklabels([])
    
    if i == 0:
        ax.text(
            -0.12, 1, "(b)",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12, fontweight="bold"
        )
    
    if i == 2:
        ax.set_ylabel("Thermal release (mgd)             ")

ax.set_xlabel(f"Date (Year={yr})")
custom_ticks = pd.to_datetime([
    f"{yr}-06-01", f"{yr}-06-15", f"{yr}-07-01", f"{yr}-07-15",
    f"{yr}-08-01", f"{yr}-08-15", f"{yr}-09-01"
])
ax.set_xticks(custom_ticks)
ax.set_xticklabels([dt.strftime("%m/%d") for dt in custom_ticks])

plt.tight_layout()
pn.outputs.mkdir(f"dps_{policy}_{job_id}/figures/RBFs")
clt.fig.savefig(fig, pn.figures.get("attemp1") / f"RBFs_thermal_release_{yr}.jpg")
#clt.fig.savefig(fig, pn.figures.get(f"attemp1") / f"RBFs_thermal_release_{yr}_with_hist.jpg")
###clt.fig.savefig(fig, pn.outputs.get(f"dps_{policy}_{job_id}/figures/RBFs") / f"RBFs_{yr}.jpg")
plt.show()

#%%
yr = 2023
for yr in range(1979, 2024):
    df_ = df_res.loc[f"{yr}-5-30":f"{yr}-9-01", :]

    colors = {
        'No control': 'k',
        'Rule-based': '#E41A1C',
        'historic': "blue",
        names[3]: "lime",
        names[4]: "aquamarine",
        names[5]: "saddlebrown",
        names[6]: "peru"
        }

    # Create 2-row subplot with height ratio 1:3
    fig, axes = plt.subplots(
        2, 1, figsize=(6, 5),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0},  # 1. Remove vertical space
        sharex=True
    )

    # ----------------- Top subplot -----------------
    ax1 = axes[0]
    ax1.plot(df_["Tmax (No control)"], ls="-", color=colors['No control'], label="Tmax (No control)")
    ax1.plot(df_["Tmax (Rule-based)"], ls="-", color=colors['Rule-based'], label="Tmax (Rule-based)")
    ax1.plot(df_["Tmax (historic)"], ls="-", color=colors['historic'], label="Tmax (historic)")
    for rbf_name in names[3:]:
        ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color=colors[f"{rbf_name}"], label=f"Tmax ({rbf_name})") #, color='dodgerblue'
    ax1.axhline(24, lw=1, c="k", ls=":")
    ax1.set_ylabel("$T_{max}$ (°C)")
    ax1.grid(True, axis='y', lw=0.3, ls="--")
    ax1.tick_params(axis='x', direction='in')  # 2. Inward x-ticks
    ax1.set_ylim([18, 27])
    ax1.set_yticks([20, 24])

    # ----------------- Bottom subplot -----------------
    ax2 = axes[1]
    # 4. historic as markers
    # Filter non-zero values
    non_zero_mask = df_["historic"] != 0
    x_vals = df_.index[non_zero_mask]
    y_vals = df_["historic"][non_zero_mask]

    # Plot only non-zero historic values with stem
    if not x_vals.empty:
        # Plot actual non-zero stems
        markerline, stemlines, baseline = ax2.stem(
            x_vals, y_vals,
            linefmt="k-", markerfmt="ko", basefmt=" ", label="historic"
        )
        markerline.set_color(colors['historic'])       # marker color
        stemlines.set_color(colors['historic'])
        plt.setp(stemlines, lw=1, zorder=80)
        plt.setp(markerline, ms=4, zorder=80)
        plt.setp(baseline, visible=False)
    else:
        # Add a dummy invisible point just to include legend entry
        ax2.plot([], [], marker='o', color='k', linestyle='None', label="historic")

    # 5. rbf and Rule-based as bars
    for rbf_name, zorder in zip(names[3:], [28, 151, 63, 106]):
        ax2.bar(df_.index, df_[f"{rbf_name}"], width=1.0, color=colors[f"{rbf_name}"], label=f"{rbf_name}", alpha=0.6, zorder=4) # color='dodgerblue'
    ax2.bar(df_.index, df_["Rule-based"], width=1.0, color=colors['Rule-based'], label="Rule-based", alpha=0.6, zorder=70)

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

    pos1 = axes[1].get_position()
    axes[1].set_position([pos1.x0, pos1.y0 - 0.07, pos1.width, pos1.height])  # Move second subplot down for hspace
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for external legend
    pn.outputs.mkdir(f"dps_{policy}_{job_id}/figures/RBFs")
    clt.fig.savefig(fig, pn.outputs.get(f"dps_{policy}_{job_id}/figures/RBFs") / f"RBFs_{yr}.jpg")
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
        ax1.plot(df_["Tmax (No control)"], ls="-", color='red', label="Tmax (No control)")
        ax1.plot(df_["Tmax (Rule-based)"], ls="-", color='chocolate', label="Tmax (Rule-based)")
        ax1.plot(df_[f"Tmax ({rbf_name})"], ls="-", color='dodgerblue', label=f"Tmax ({rbf_name})")
        ax1.axhline(24, lw=1, c="k", ls=":")
        ax1.set_ylabel("$T_{max}$ (°C)")
        ax1.grid(True, axis='y', lw=0.3, ls="--")
        ax1.tick_params(axis='x', direction='in')  # 2. Inward x-ticks
        ax1.set_ylim([18, 27])
        ax1.set_yticks([20, 24])

        # ----------------- Bottom subplot -----------------
        ax2 = axes[1]
        # 4. historic as markers
        # Filter non-zero values
        non_zero_mask = df_["historic"] != 0
        x_vals = df_.index[non_zero_mask]
        y_vals = df_["historic"][non_zero_mask]

        # Plot only non-zero historic values with stem
        if not x_vals.empty:
            # Plot actual non-zero stems
            markerline, stemlines, baseline = ax2.stem(
                x_vals, y_vals,
                linefmt="k-", markerfmt="ko", basefmt=" ", label="historic"
            )
            plt.setp(stemlines, lw=1, zorder=0)
            plt.setp(markerline, ms=4, zorder=0)
            plt.setp(baseline, visible=False)
        else:
            # Add a dummy invisible point just to include legend entry
            ax2.plot([], [], marker='o', color='k', linestyle='None', label="historic")

        # 5. rbf and Rule-based as bars
        ax2.bar(df_.index, df_[f"{rbf_name}"], width=1.0, color='dodgerblue', label=f"{rbf_name}", alpha=0.6, zorder=4)
        ax2.bar(df_.index, df_["Rule-based"], width=1.0, color='chocolate', label="Rule-based", alpha=0.6, zorder=2)

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

#%% Plot annual historic releases
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
# Top subplot: Tmax (No control)
ax.plot(df_["Tmax (No control)"], ls="-", color='grey', label="Tmax (No control)")
ax.axhline(24, lw=1, c="k", ls=":")
ax.set_ylabel("$T_{max}$ (°C)")
ax.legend(frameon=False)
ax.grid(True, axis='y', lw=0.3, ls="--")

ax = axes[1]
# Bottom subplot: Thermal releases
ax.plot(df_["historic"], color='k', lw=2, ls="-", label="historic")
ax.plot(df_["rbf"], color='salmon', lw=2, label="rbf")
ax.plot(df_["Rule-based"], color='dodgerblue', lw=2, label="Rule-based")
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
    ax.plot(df_["historic"], color='k', lw=2, ls="-", label="historic")
    ax.plot(df_["rbf"], color='salmon', lw=2, label="rbf")
    ax.plot(df_["Rule-based"], color='dodgerblue', lw=2, label="Rule-based")
    ax.grid(True, axis='y', lw=0.3, ls="--")
    ax.legend(frameon=False)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks[::3])
    ax.set_xlim([df_.index[0], df_.index[-1]])
    ax.set_ylabel("Thermal release (mgd)")
    ax.set_xlabel("Date")

    ax2 = ax.twinx()
    ax2.plot(df_["Tmax (No control)"], ls="-", color='grey', label="Tmax (No control)")
    ax2.axhline(24, lw=1, c="k", ls=":")
    ax2.set_ylabel("$T_{max}$ (°C)")

    plt.tight_layout()
    #clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "compare_with_hist_tr.jpg")
    plt.show()
