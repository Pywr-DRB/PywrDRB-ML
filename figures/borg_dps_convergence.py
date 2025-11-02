#%% Plot DPS convergence
import matplotlib.pyplot as plt
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

policy = "GaussianRBFPolicy"
job_id = "143990"
folder = f"dps_GaussianRBFPolicy_{job_id}"  #f"stage1_nowcast_{policy}_{job_id}"
df_met = clt.borg.read_metric_files(pn.outputs.get(folder) / "metrics")

#%%
fig, ax = plt.subplots()
clt.borg.plot_convergence_across_seeds(ax, pn.outputs.get(folder) / "metrics", frequency=500)

# Legend
# Get legend handles and labels
handles, labels = ax.get_legend_handles_labels()

# Sort the legend entries numerically by seed number
sorted_pairs = sorted(zip(labels, handles), key=lambda x: int(x[0].split()[-1]))
sorted_labels, sorted_handles = zip(*sorted_pairs)

# Apply sorted legend
ax.legend(sorted_handles, sorted_labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

ax.set_xlabel("Number of evaluations")
ax.set_xlim([0, 50000])
plt.tight_layout()  # ensures everything fits nicely
clt.fig.savefig(fig, filename=pn.figures.get("attemp1")/"borg_dps_convergence.jpg")
plt.show()