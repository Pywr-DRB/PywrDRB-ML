import pandas as pd
import numpy as np
import pathnavigator

if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
elif pathnavigator.os_name == 'Darwin':
    root_dir = rf"/Users/{pathnavigator.user}/Documents/GitHub/PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")

pn = pathnavigator.create(root_dir)
pn.add_to_sys_path()
pn.chdir()

db_TempLSTM = pd.read_csv(pn.data.database.get("TempLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']
db_SalinityLSTM = pd.read_csv(pn.data.database.get("SalinityLSTM_database.csv"), index_col=0, parse_dates=True)['1979-01-01':'2023-12-31']



df_obs = pd.DataFrame()
df_obs["$Saltfront$"] = db_SalinityLSTM["saltfront"]
df_obs.loc[db_SalinityLSTM["saltfront_src"] != "obs", "$Saltfront$"] = np.nan

df_obs["$T_C$"] = db_TempLSTM["QobsTavg_T_C"]
df_obs.loc[db_TempLSTM["tavg_water_cannonsville_src"] != "obs", "$T_C$"] = np.nan
df_obs["$T_i$"] = db_TempLSTM["QobsTavg_T_i"]
df_obs.loc[db_TempLSTM["tavg_water_src"] != "obs", "$T_i$"] = np.nan
df_obs["$T_{avg}$"] = db_TempLSTM["QbcTavg_T_L"]
df_obs.loc[db_TempLSTM["tavg_water_lordville_src"] != "obs", "$T_{avg}$"] = np.nan
df_obs["$T_{max}$"] = db_TempLSTM["QbcTmax_T_L"]
df_obs.loc[db_TempLSTM["tmmx_water_lordville_src"] != "obs", "$T_{max}$"] = np.nan

#%%
import clt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(8, 4))

colors = {
    "$T_C$": "tab:blue",
    "$T_i$": "tab:orange",
    "$T_{avg}$": "tab:green",
    "$T_{max}$": "tab:red",
    "$Saltfront$": "tab:purple"
}

bar_height = 0.8
y_positions = np.arange(len(colors))

for i, col in enumerate(colors.keys()):
    mask = ~df_obs[col].isna()
    if mask.any():
        # find contiguous True segments
        mask_shift = mask != mask.shift(1, fill_value=False)
        segment_starts = df_obs.index[mask_shift & mask]
        segment_ends = df_obs.index[mask_shift & ~mask]
        # handle trailing True
        if len(segment_starts) > len(segment_ends):
            segment_ends = segment_ends.append(pd.Index([df_obs.index[-1]]))
        for start, end in zip(segment_starts, segment_ends):
            ax.barh(i, (end - start).days, left=start, height=bar_height,
                    color=colors[col], align='center')

# Formatting
ax.set_yticks(y_positions)
ax.set_yticklabels(list(colors.keys()), fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.4)

# Legend outside to the right
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
ax.legend(handles=legend_elements, title="Variables", frameon=False,
          loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=12, title_fontsize=12)

plt.tight_layout()
clt.fig.savefig(fig, filename=pn.figures.get("attemp1") / "data_availability.jpg")
plt.show()