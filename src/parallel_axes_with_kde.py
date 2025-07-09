import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_parallel_coords_with_kde(
        ax, df, columns, dict_results_disagg_MC=None,
        soln_labels=None, objmins=None, objmaxs=None,
        axes_labels=["Jtubr", "Jrel", "Jadd"],
        ideal_direction='top', fontsize=10, kde_scale=0.12
        ):

    df_subset = df.loc[:, columns].copy()
    num_axes = len(columns)
    x_spacing = np.linspace(0, num_axes - 1, num_axes)

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
    colors_brewer = {'s3/soln212': '#1b9e77', 'statusquo/soln0': '#d95f02', 'other': '#7570b3'}
    if soln_labels is not None:
        for soln_label in soln_labels:
            c = colors_brewer.get(soln_label, 'blue')
            soln_data = df_subset.loc[df['label'] == soln_label]
            xx, yy = [], []
            for j in range(num_axes - 1):
                x = np.linspace(x_spacing[j], x_spacing[j + 1], 11)
                y = soln_data.iloc[0, j] + (x - x_spacing[j]) * \
                    (soln_data.iloc[0, j + 1] - soln_data.iloc[0, j]) / (x_spacing[j + 1] - x_spacing[j])
                xx += list(x); yy += list(y)
            ax.plot(xx, yy, c='k', lw=3, zorder=5)
            ax.plot(xx, yy, c=c, lw=2, zorder=6)

            # KDE shading
            if dict_results_disagg_MC is not None:
                results_MC = dict_results_disagg_MC[soln_label].copy()
                columns_MC = columns
                for o, obj in enumerate(columns_MC):
                    scaled_col = f"{obj}_scaled"
                    if ideal_direction == 'top':
                        results_MC[scaled_col] = (results_MC[obj] - bottoms[o]) / (tops[o] - bottoms[o])
                    else:
                        results_MC[scaled_col] = (bottoms[o] - results_MC[obj]) / (bottoms[o] - tops[o])

                for o, obj in enumerate(columns_MC[1:]):
                    y = np.arange(0, 1, 0.01)
                    data = results_MC[f'{obj}_scaled']
                    kde = sm.nonparametric.KDEUnivariate(data)
                    kde.fit(bw=0.025)
                    kde_scale = kde_scale
                    x = np.array([kde.evaluate(v)[0] * kde_scale if not np.isnan(kde.evaluate(v)[0]) else 0 for v in y])
                    ax.fill_betweenx(y, x + x_spacing[o + 1], x_spacing[o + 1],
                                     where=(x > 0.00005), lw=1, alpha=0.6, zorder=4, fc=c, ec='k')

                # Debug print
                print(f"{soln_label} min, mean, max:")
                for o, obj in enumerate(columns_MC[1:]):
                    print(f"{obj}: {results_MC[obj].min():.2f}, {results_MC[obj].mean():.2f}, {results_MC[obj].max():.2f}")
                print()

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
df_ref = clt.borg.read_ref(pn.outputs.get("stage1_GaussianRBFPolicy_134717/borg.ref"))
df_ref = df_ref[['obj3', 'obj1', 'obj2']]
df_ref.columns = ["Jtubr", "Jrel", "Jadd"]
df_ref["Jrel"] *= -1
df_ref["label"] = np.nan
df_ref["label"][0] = 1

df = df_ref
fig, ax = plt.subplots()
plot_parallel_coords_with_kde(
    ax, df, columns=["Jtubr", "Jrel", "Jadd"], dict_results_disagg_MC=None,
    soln_labels=[1], objmins=None, objmaxs=None,
    axes_labels=["Jtubr", "Jrel", "Jadd"],
    ideal_direction='bottom', fontsize=10, kde_scale=0.12
    )
plt.tight_layout()
plt.show()
#%%
# Define required solution labels
soln_labels = ['s3/soln212', 'statusquo/soln0', 'soln1', 'soln2', 'soln3']

# Create df with necessary columns
df = pd.DataFrame({
    'label': soln_labels,
    'n_partner': np.random.randint(11, 26, size=len(soln_labels)),
    'captured_gain': np.random.uniform(20, 115, size=len(soln_labels)),
    'captured_nonpartner': np.random.uniform(-100, 31, size=len(soln_labels)),
    'cog_wp_p90': np.random.uniform(100, 1000, size=len(soln_labels))
})
df['cog_wp'] = df['cog_wp_p90']

# Create dict_results_disagg_MC with matching keys
dict_results_disagg_MC = {
    label: pd.DataFrame({
        'n_partner': np.random.randint(11, 26, size=100),
        'captured_gain': np.random.uniform(20, 115, size=100),
        'captured_nonpartner': np.random.uniform(-100, 31, size=100),
        'cog_wp_p90': np.random.uniform(100, 1000, size=100),
        'cog_wp': np.random.uniform(100, 1000, size=100)
    })
    for label in soln_labels
}

# fig = plt.figure(figsize=(10, 6))
# gs = fig.add_gridspec(1, 2)
# ax = fig.add_subplot(gs[0, 1])

fig, ax = plt.subplots()
plot_parallel_coords_with_kde(
    ax=ax,
    df=df,  # Your preloaded DataFrame
    dict_results_disagg_MC=dict_results_disagg_MC,  # Your dictionary of MC samples
    columns=['n_partner', 'captured_gain', 'captured_nonpartner'], #, 'cog_wp_p90'
    objmins=[11, 35, -100], objmaxs=[26, 115, 31],
    axes_labels=['Number\nof\npartners', 'Captured\nwater\ngain\n(GL/yr)',
              'Captured\nwater\ngain for\nnon-partners\n(GL/yr)'],
    soln_labels=['s3/soln212'], # , 'statusquo/soln0'
    ideal_direction='top',
    fontsize=10,
    kde_scale=0.12
)

plt.tight_layout()
plt.show()
