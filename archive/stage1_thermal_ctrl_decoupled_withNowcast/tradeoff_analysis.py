#%%
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
job_id = "138146" #"138146"
folder = "dps_GaussianRBFPolicy_138146"  #f"stage1_nowcast_{policy}_{job_id}"
df_met = clt.borg.read_metric_files(pn.outputs.get(folder) / "metrics")
df_ref = clt.borg.read_ref(pn.outputs.get(folder) / "borg.ref")

#%% Plot convergence
job_id = "138146"
policies = ["GaussianRBFPolicy"]#, "RegressionPolicy", "CubicRBFPolicy", "GeneralizedPiecewiseLinearPolicy"]
for policy in policies:
    folder = "dps_GaussianRBFPolicy_138146"  #f"stage1_nowcast_{policy}_{job_id}"
    df_met = clt.borg.read_metric_files(pn.outputs.get(folder) / "metrics")
    df_met.plot()

    fig, ax = plt.subplots()
    clt.borg.plot_convergence_across_seeds(ax, pn.outputs.get(folder) / "metrics", frequency=500)
    ax.set_title(f"Convergence for {policy}")
    plt.show()
#%% Plot Pareto front
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

policy = "RegressionPolicy"
job_id = "134989"
folder = f"stage1_nowcast_{policy}_{job_id}"
df_ref = clt.borg.read_ref(pn.outputs.get(folder) / "borg.ref")

# Get the last three objective columns
obj_columns = [col for col in df_ref.columns if 'obj' in col.lower()][-3:]

# Rename the columns to match obj_names
obj_names = ["-Jrel", "Jadd", "Jtubr"]
rename_dict = dict(zip(obj_columns, obj_names))
df_ref = df_ref.rename(columns=rename_dict)

# Define reference point (adjust these values as needed)
ref_pt_noCtrl = [-0.2375, 1, 0.0]  # Example values for [-Jrel, Jadd, Jtubr]
ref_pt_rulebased = [-0.3185, 0.916, 0.0734]  # Example values for [-Jrel, Jadd, Jtubr]

# 1) 3D Scatter Plot - colored by Jtubr with reference point
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df_ref[obj_names[0]],
                    df_ref[obj_names[1]],
                    df_ref[obj_names[2]],
                    c=df_ref[obj_names[2]],  # Color by Jtubr
                    cmap='viridis',
                    alpha=0.7,
                    label='Pareto Solutions')

# Add reference point
ax.scatter(ref_pt_noCtrl[0], ref_pt_noCtrl[1], ref_pt_noCtrl[2],
          color='red', s=200, marker='*', label='No Ctrl')
ax.scatter(ref_pt_rulebased[0], ref_pt_rulebased[1], ref_pt_rulebased[2],
          color='blue', s=200, marker='*', label='No Ctrl')

ax.set_xlabel(obj_names[0])
ax.set_ylabel(obj_names[1])
ax.set_zlabel(obj_names[2])
ax.set_title('3D Pareto Front (colored by Jtubr)')

ax.view_init(elev=20, azim=60)  # Adjust these angles as needed
ax.legend()
cbar = plt.colorbar(scatter)
cbar.set_label('Jtubr')
plt.show()

#%%
# 2) Parallel Coordinates Plot - colored by Jtubr with reference line
fig, ax = plt.subplots(figsize=(12, 6))

# Normalize the data to [0, 1] for better visualization
# Normalize the data to [0, 1] for better visualization
df_norm = df_ref[obj_names].copy()
ref_norm_noCtrl = ref_pt_noCtrl.copy()
ref_norm_rulebased = ref_pt_rulebased.copy()

for i, col in enumerate(obj_names):
    df_norm[col] = (df_ref[col] - df_ref[col].min()) / (df_ref[col].max() - df_ref[col].min())
    # Normalize both reference points using same scaling
    ref_norm_noCtrl[i] = (ref_pt_noCtrl[i] - df_ref[col].min()) / (df_ref[col].max() - df_ref[col].min())
    ref_norm_rulebased[i] = (ref_pt_rulebased[i] - df_ref[col].min()) / (df_ref[col].max() - df_ref[col].min())

# Create colormap based on Jtubr values
import matplotlib.cm as cm
norm = plt.Normalize(vmin=df_ref[obj_names[2]].min(), vmax=df_ref[obj_names[2]].max())
cmap = cm.get_cmap('viridis')

# Plot parallel coordinates with color based on Jtubr
for i in range(len(df_norm)):
    color = cmap(norm(df_ref[obj_names[2]].iloc[i]))
    ax.plot(range(len(obj_names)), df_norm.iloc[i], alpha=0.7, linewidth=1, color=color)

# Add reference lines
ax.plot(range(len(obj_names)), ref_norm_noCtrl, color='red', linewidth=3,
        marker='*', markersize=10, label='No Ctrl')
ax.plot(range(len(obj_names)), ref_norm_rulebased, color='blue', linewidth=3,
        marker='*', markersize=10, label='Rule-based')

ax.set_xticks(range(len(obj_names)))
ax.set_xticklabels(obj_names, rotation=45)
ax.set_ylabel('Normalized Objective Values')
ax.set_title('Parallel Coordinates Plot - Pareto Front (colored by Jtubr)')
ax.grid(True, alpha=0.3)
ax.legend()

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Jtubr')

plt.tight_layout()
plt.show()
#%%
def create_interactive_3d_plot(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased):
    """
    Create an interactive 3D plot of the Pareto front with reference points.

    Parameters
    ----------
    df_ref : pandas.DataFrame
        DataFrame containing the Pareto front data
    obj_names : list
        List of objective names ["-Jrel", "Jadd", "Jtubr"]
    ref_pt_noCtrl : list
        Reference point for no control case
    ref_pt_rulebased : list
        Reference point for rule-based case
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Enable interactive mode
    plt.ion()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    scatter = ax.scatter(df_ref[obj_names[0]],
                        df_ref[obj_names[1]],
                        df_ref[obj_names[2]],
                        c=df_ref[obj_names[2]],  # Color by Jtubr
                        cmap='viridis',
                        alpha=0.7,
                        s=20,  # Smaller points for better visibility
                        label='Pareto Solutions')

    # Add reference points
    ax.scatter(ref_pt_noCtrl[0], ref_pt_noCtrl[1], ref_pt_noCtrl[2],
              color='red', s=200, marker='*', label='No Ctrl', edgecolors='black')
    ax.scatter(ref_pt_rulebased[0], ref_pt_rulebased[1], ref_pt_rulebased[2],
              color='blue', s=200, marker='*', label='Rule-based', edgecolors='black')

    # Set labels and title
    ax.set_xlabel(obj_names[0], fontsize=12)
    ax.set_ylabel(obj_names[1], fontsize=12)
    ax.set_zlabel(obj_names[2], fontsize=12)
    ax.set_title('Interactive 3D Pareto Front (colored by Jtubr)', fontsize=14)

    # Add legend
    ax.legend(loc='upper left')

    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Jtubr', fontsize=12)

    # Add grid for better visualization
    ax.grid(True, alpha=0.3)

    # Set initial view angle
    ax.view_init(elev=20, azim=60)

    # Add some useful text
    fig.text(0.02, 0.02, 'Click and drag to rotate • Scroll to zoom • Right-click drag to pan',
             fontsize=10, alpha=0.7)

    plt.tight_layout()
    plt.show()

    return fig, ax

def create_interactive_3d_with_controls(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased):
    """
    Create an interactive 3D plot with additional control widgets (requires ipywidgets).
    """
    try:
        from ipywidgets import interact, FloatSlider, IntSlider
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        def update_plot(elevation=20, azimuth=60, alpha=0.7, point_size=20):
            plt.clf()
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(df_ref[obj_names[0]],
                                df_ref[obj_names[1]],
                                df_ref[obj_names[2]],
                                c=df_ref[obj_names[2]],
                                cmap='viridis',
                                alpha=alpha,
                                s=point_size,
                                label='Pareto Solutions')

            ax.scatter(ref_pt_noCtrl[0], ref_pt_noCtrl[1], ref_pt_noCtrl[2],
                      color='red', s=200, marker='*', label='No Ctrl', edgecolors='black')
            ax.scatter(ref_pt_rulebased[0], ref_pt_rulebased[1], ref_pt_rulebased[2],
                      color='blue', s=200, marker='*', label='Rule-based', edgecolors='black')

            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
            ax.set_title('Interactive 3D Pareto Front with Controls')
            ax.legend()
            ax.view_init(elev=elevation, azim=azimuth)

            cbar = plt.colorbar(scatter, shrink=0.8)
            cbar.set_label('Jtubr')
            plt.show()

        # Create interactive widgets
        interact(update_plot,
                elevation=IntSlider(min=-90, max=90, step=5, value=20, description='Elevation:'),
                azimuth=IntSlider(min=0, max=360, step=10, value=60, description='Azimuth:'),
                alpha=FloatSlider(min=0.1, max=1.0, step=0.1, value=0.7, description='Transparency:'),
                point_size=IntSlider(min=5, max=50, step=5, value=20, description='Point Size:'))

    except ImportError:
        print("ipywidgets not available. Using basic interactive plot instead.")
        return create_interactive_3d_plot(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased)

# Usage example:
# Basic interactive plot
fig, ax = create_interactive_3d_with_controls(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased)

#%%
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

def create_plotly_3d_interactive(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased):
    """
    Create an interactive 3D plot using Plotly with enhanced interactivity.

    Parameters
    ----------
    df_ref : pandas.DataFrame
        DataFrame containing the Pareto front data
    obj_names : list
        List of objective names ["-Jrel", "Jadd", "Jtubr"]
    ref_pt_noCtrl : list
        Reference point for no control case
    ref_pt_rulebased : list
        Reference point for rule-based case
    """


    # Create the main scatter plot for Pareto solutions
    fig = go.Figure()

    # Add Pareto solutions colored by Jtubr
    fig.add_trace(go.Scatter3d(
        x=df_ref[obj_names[0]],
        y=df_ref[obj_names[1]],
        z=df_ref[obj_names[2]],
        mode='markers',
        marker=dict(
            size=5,
            color=df_ref[obj_names[2]],  # Color by Jtubr
            colorscale='Viridis',
            colorbar=dict(title="Jtubr"),
            opacity=0.8
        ),
        name='Pareto Solutions',
        hovertemplate=
        f'<b>Pareto Solution</b><br>' +
        f'{obj_names[0]}: %{{x:.4f}}<br>' +
        f'{obj_names[1]}: %{{y:.4f}}<br>' +
        f'{obj_names[2]}: %{{z:.4f}}<br>' +
        '<extra></extra>'
    ))

    # Add reference point - No Control
    fig.add_trace(go.Scatter3d(
        x=[ref_pt_noCtrl[0]],
        y=[ref_pt_noCtrl[1]],
        z=[ref_pt_noCtrl[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='diamond',
            line=dict(color='black', width=2)
        ),
        name='No Control',
        hovertemplate=
        f'<b>No Control</b><br>' +
        f'{obj_names[0]}: {ref_pt_noCtrl[0]:.4f}<br>' +
        f'{obj_names[1]}: {ref_pt_noCtrl[1]:.4f}<br>' +
        f'{obj_names[2]}: {ref_pt_noCtrl[2]:.4f}<br>' +
        '<extra></extra>'
    ))

    # Add reference point - Rule-based
    fig.add_trace(go.Scatter3d(
        x=[ref_pt_rulebased[0]],
        y=[ref_pt_rulebased[1]],
        z=[ref_pt_rulebased[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='blue',
            symbol='diamond',
            line=dict(color='black', width=2)
        ),
        name='Rule-based',
        hovertemplate=
        f'<b>Rule-based</b><br>' +
        f'{obj_names[0]}: {ref_pt_rulebased[0]:.4f}<br>' +
        f'{obj_names[1]}: {ref_pt_rulebased[1]:.4f}<br>' +
        f'{obj_names[2]}: {ref_pt_rulebased[2]:.4f}<br>' +
        '<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': '3D Interactive Pareto Front (Colored by Jtubr)',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title=obj_names[0],
            yaxis_title=obj_names[1],
            zaxis_title=obj_names[2],
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def create_plotly_parallel_coordinates(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased):
    """
    Create an interactive parallel coordinates plot using Plotly.
    """

    # Normalize the data
    df_norm = df_ref[obj_names].copy()
    for col in obj_names:
        df_norm[col] = (df_ref[col] - df_ref[col].min()) / (df_ref[col].max() - df_ref[col].min())

    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=df_ref[obj_names[2]],  # Color by Jtubr
                     colorscale='Viridis',
                     showscale=True,
                     colorbar=dict(title="Jtubr")),
            dimensions=[
                dict(range=[0, 1],
                     tickvals=[0, 0.25, 0.5, 0.75, 1],
                     ticktext=[f'{df_ref[obj_names[0]].min():.3f}',
                              f'{df_ref[obj_names[0]].min() + 0.25*(df_ref[obj_names[0]].max()-df_ref[obj_names[0]].min()):.3f}',
                              f'{df_ref[obj_names[0]].min() + 0.5*(df_ref[obj_names[0]].max()-df_ref[obj_names[0]].min()):.3f}',
                              f'{df_ref[obj_names[0]].min() + 0.75*(df_ref[obj_names[0]].max()-df_ref[obj_names[0]].min()):.3f}',
                              f'{df_ref[obj_names[0]].max():.3f}'],
                     label=obj_names[0], values=df_norm[obj_names[0]]),
                dict(range=[0, 1],
                     tickvals=[0, 0.25, 0.5, 0.75, 1],
                     ticktext=[f'{df_ref[obj_names[1]].min():.1f}',
                              f'{df_ref[obj_names[1]].min() + 0.25*(df_ref[obj_names[1]].max()-df_ref[obj_names[1]].min()):.1f}',
                              f'{df_ref[obj_names[1]].min() + 0.5*(df_ref[obj_names[1]].max()-df_ref[obj_names[1]].min()):.1f}',
                              f'{df_ref[obj_names[1]].min() + 0.75*(df_ref[obj_names[1]].max()-df_ref[obj_names[1]].min()):.1f}',
                              f'{df_ref[obj_names[1]].max():.1f}'],
                     label=obj_names[1], values=df_norm[obj_names[1]]),
                dict(range=[0, 1],
                     tickvals=[0, 0.25, 0.5, 0.75, 1],
                     ticktext=[f'{df_ref[obj_names[2]].min():.3f}',
                              f'{df_ref[obj_names[2]].min() + 0.25*(df_ref[obj_names[2]].max()-df_ref[obj_names[2]].min()):.3f}',
                              f'{df_ref[obj_names[2]].min() + 0.5*(df_ref[obj_names[2]].max()-df_ref[obj_names[2]].min()):.3f}',
                              f'{df_ref[obj_names[2]].min() + 0.75*(df_ref[obj_names[2]].max()-df_ref[obj_names[2]].min()):.3f}',
                              f'{df_ref[obj_names[2]].max():.3f}'],
                     label=obj_names[2], values=df_norm[obj_names[2]])
            ]
        )
    )

    fig.update_layout(
        title={
            'text': 'Interactive Parallel Coordinates Plot - Pareto Front (Colored by Jtubr)',
            'x': 0.5,
            'font': {'size': 16}
        },
        width=1000,
        height=500
    )

    return fig

# Usage - replace your matplotlib functions with these:
# Create interactive 3D plot
fig_3d = create_plotly_3d_interactive(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased)
fig_3d.show()

# Create interactive parallel coordinates plot
fig_parallel = create_plotly_parallel_coordinates(df_ref, obj_names, ref_pt_noCtrl, ref_pt_rulebased)
fig_parallel.show()