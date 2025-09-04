#%% Plot DPS convergence
import matplotlib.pyplot as plt
import pathnavigator
if pathnavigator.os_name == 'Windows':
    root_dir = rf"C:\Users\{pathnavigator.user}\Documents\GitHub\PywrDRB-ML"
else:
    root_dir = pathnavigator.expanduser("~/Github/PywrDRB-ML")
pn = pathnavigator.create(root_dir)
pn.chdir()
import clt

policy = "GaussianRBFPolicy"
job_id = "138158" # "138146" with storage dynamics
folder = f"dps_GaussianRBFPolicy_{job_id}"  #f"stage1_nowcast_{policy}_{job_id}"
df_met = clt.borg.read_metric_files(pn.outputs.get(folder) / "metrics")

#%%
fig, ax = plt.subplots()
clt.borg.plot_convergence_across_seeds(ax, pn.outputs.get(folder) / "metrics", frequency=500)

# Legend
handles, labels = ax.get_legend_handles_labels()
handles = handles[:1] + handles[2:] + [handles[1]]
labels  = labels[:1]  + labels[2:]  + [labels[1]]
ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

ax.set_xlabel("Number of evaluations")
ax.set_xlim([0, 100000])
plt.tight_layout()  # ensures everything fits nicely
clt.fig.savefig(fig, filename=pn.figures.get("attemp1")/"borg_dps_convergence.jpg")
plt.show()