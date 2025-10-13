import random
import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np
import plotly.graph_objects as go


def plot_gpfa_traj_3d(trajectories, figsize=(15, 5), linewidth_single_trial=0.5,
                      alpha_single_trial=0.2, linewidth_trial_average=3,
                      title='Latent dynamics extracted by GPFA',
                      num_traj_to_plot=30,
                      view_azim=-5, view_elev=60):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    T = trajectories[0].shape[1]
    color_vals = np.linspace(0, 1, T - 1)
    cmap = cm.viridis
    norm = Normalize(0, T - 1)

    def make_segments_3d(xyz):
        return np.array([[[xyz[0, i], xyz[1, i], xyz[2, i]],
                          [xyz[0, i+1], xyz[1, i+1], xyz[2, i+1]]]
                         for i in range(xyz.shape[1] - 1)])

    # Collect all points for axis scaling
    all_points = []

    if num_traj_to_plot is None:
        num_traj_to_plot = len(trajectories)

    # Plot single trials
    for traj in trajectories[:num_traj_to_plot]:
        segs = make_segments_3d(traj)
        all_points.append(traj.T)
        lc = Line3DCollection(segs, colors=cmap(
            color_vals), linewidth=linewidth_single_trial, alpha=alpha_single_trial)
        ax.add_collection3d(lc)

    # Plot average trajectory
    avg_traj = np.mean(trajectories, axis=0)
    segs_avg = make_segments_3d(avg_traj)
    all_points.append(avg_traj.T)
    lc_avg = Line3DCollection(segs_avg, colors=cmap(
        color_vals), linewidth=linewidth_trial_average)
    ax.add_collection3d(lc_avg)

    # Compute axis limits
    all_points = np.concatenate(all_points, axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)
    pad_z = 0.05 * (z_max - z_min)

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_zlim(z_min - pad_z, z_max + pad_z)

    # Set labels and title
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)
    ax.view_init(elev=view_elev, azim=view_azim)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Time Step')

    plt.tight_layout()
    return fig, ax


def plot_gpfa_traj_3d_slow(trajectories, figsize=(15, 5), linewidth_single_trial=0.75,
                           alpha_single_trial=0.3, linewidth_trial_average=2,
                           title='Latent dynamics extracted by GPFA',
                           num_traj_to_plot=30,
                           view_azim=-5, view_elev=60):
    """
    Plot interactive 3D trajectories from GPFA analysis with temporal color gradients.

    Parameters
    ----------
    trajectories : list of arrays
        List of trajectory arrays, each of shape (3, n_timepoints)
    figsize : tuple
        Figure size (width, height)
    linewidth_single_trial : float
        Line width for individual trial trajectories
    alpha_single_trial : float
        Transparency for individual trial trajectories
    linewidth_trial_average : float
        Line width for average trajectory
    title : str
        Plot title
    view_azim : float
        Azimuth angle for 3D view
    view_elev : float
        Elevation angle for 3D view
    """
    # Enable interactive mode

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Get number of time steps
    T = trajectories[0].shape[1]
    colors = cm.viridis(np.linspace(0, 1, T - 1))

    # Plot each trial with temporal color gradient
    for traj in trajectories[:num_traj_to_plot]:
        for t in range(T - 1):
            ax.plot(traj[0, t:t+2],
                    traj[1, t:t+2],
                    traj[2, t:t+2],
                    color=colors[t],
                    lw=linewidth_single_trial,
                    alpha=alpha_single_trial)

    # Plot trial-averaged trajectory with color gradient
    avg_traj = np.mean(trajectories, axis=0)
    for t in range(T - 1):
        ax.plot(avg_traj[0, t:t+2],
                avg_traj[1, t:t+2],
                avg_traj[2, t:t+2],
                color=colors[t],
                lw=linewidth_trial_average)

    # Add colorbar to show temporal progression
    norm = plt.Normalize(0, T-1)
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Time Step')

    # Set labels and title
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.set_title(title)

    # Manual legend entry
    ax.plot([], [], [], lw=linewidth_trial_average,
            color='C1', label='Trial averaged trajectory')
    ax.legend()

    # # Set viewing angle
    ax.view_init(azim=view_azim, elev=view_elev)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def plot_gpfa_traj_3d_uniform_color(trajectories,
                                    linewidth_single_trial=0.5,
                                    color_single_trial='C0',
                                    alpha_single_trial=0.5,
                                    linewidth_trial_average=2,
                                    color_trial_average='C1',
                                    view_azim=-5,
                                    view_elev=60,
                                    title='Latent dynamics extracted by GPFA'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')

    # Plot single trials
    for single_trial_trajectory in trajectories:
        ax.plot(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)

    # Plot average
    average_trajectory = np.mean(trajectories, axis=0)
    ax.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
            lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')

    ax.legend()
    ax.view_init(azim=view_azim, elev=view_elev)
    plt.tight_layout()
    plt.show()
    return fig, ax


def _set_equal_aspect_3d(ax, X, Y, Z):
    # Equal aspect box around data
    x_range = np.nanmax(X) - np.nanmin(X)
    y_range = np.nanmax(Y) - np.nanmin(Y)
    z_range = np.nanmax(Z) - np.nanmin(Z)
    max_range = max(x_range, y_range, z_range)
    x_mid = (np.nanmax(X) + np.nanmin(X)) / 2
    y_mid = (np.nanmax(Y) + np.nanmin(Y)) / 2
    z_mid = (np.nanmax(Z) + np.nanmin(Z)) / 2
    ax.set_xlim(x_mid - max_range/2, x_mid + max_range/2)
    ax.set_ylim(y_mid - max_range/2, y_mid + max_range/2)
    ax.set_zlim(z_mid - max_range/2, z_mid + max_range/2)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def plot_gpfa_traj_3d_plotly(trajectories,
                             alpha_single_trial=0.1,
                             linewidth_single_trial=2,
                             linewidth_avg=4,
                             stride=2,
                             colorscale='Viridis',
                             num_trials_to_plot=30,
                             title='Latent dynamics extracted by GPFA'):
    """
    Interactive 3D Plotly plot of GPFA trajectories with color gradient over time.
    """

    T = trajectories[0].shape[1]
    color_vals = np.linspace(0, 1, T // stride)

    fig = go.Figure()

    # Plot individual trials
    for traj in trajectories[:num_trials_to_plot]:
        x, y, z = traj[:, ::stride]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                width=linewidth_single_trial,
                color=color_vals,
                colorscale=colorscale
            ),
            opacity=alpha_single_trial,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Plot average trajectory
    avg = np.mean(trajectories, axis=0)
    x_avg, y_avg, z_avg = avg[:, ::stride]
    fig.add_trace(go.Scatter3d(
        x=x_avg, y=y_avg, z=z_avg,
        mode='lines',
        line=dict(
            width=linewidth_avg,
            color=color_vals,
            colorscale=colorscale
        ),
        name='Trial averaged trajectory',
        hoverinfo='skip'
    ))

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Dim 1',
            yaxis_title='Dim 2',
            zaxis_title='Dim 3'
        ),
        title=title,
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    plt.show()
    return fig


def plot_gpfa_traj_3d_timecolored_average(
    trajectories,
    linewidth_single_trial=0.5,
    color_single_trial='C0',
    alpha_single_trial=0.2,
    linewidth_trial_average=3,
    cmap_average='viridis',
    show_colorbar=True,
    start_end_markers=True,
    start_marker='o',
    end_marker='X',
    marker_size=30,
    view_azim=-5,
    view_elev=60,
    title='Latent dynamics extracted by GPFA'
):
    """
    trajectories: iterable of arrays shaped (D, T), where D >= 3.
    Only the first 3 dimensions are plotted.
    """

    # --- Convert to numpy and clip to first 3 dims
    trajs = []
    for tr in trajectories:
        arr = np.asarray(tr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"Each trajectory must be 2D (dims x time); got {arr.shape}.")
        if arr.shape[0] < 3:
            raise ValueError(
                f"Trajectory must have at least 3 dims; got {arr.shape[0]}.")
        trajs.append(arr[:3])  # take first 3 dims

    if len(trajs) == 0:
        raise ValueError("No trajectories provided.")

    # --- Figure setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')

    # --- Single trials (uniform color)
    for tr in trajs:
        ax.plot(tr[0], tr[1], tr[2],
                lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)

    # --- NaN-padding for averaging
    max_T = max(tr.shape[1] for tr in trajs)
    padded = np.full((len(trajs), 3, max_T), np.nan)
    for i, tr in enumerate(trajs):
        T = tr.shape[1]
        padded[i, :, :T] = tr

    # Average across trials, ignoring NaNs
    average_trajectory = np.nanmean(padded, axis=0)  # (3, max_T)
    pts = average_trajectory.T                       # (max_T, 3)

    # Build line segments
    segs = np.stack([pts[:-1], pts[1:]], axis=1)   # (max_T-1, 2, 3)
    t = np.linspace(0.0, 1.0, max_T)

    lc = Line3DCollection(segs, cmap=cmap_average,
                          norm=Normalize(vmin=0.0, vmax=1.0))
    lc.set_array(t[:-1])
    lc.set_linewidth(linewidth_trial_average)
    ax.add_collection3d(lc)

    # Start/end markers
    if start_end_markers:
        ax.scatter(*pts[0],  s=marker_size, marker=start_marker, label='start')
        ax.scatter(*pts[-1], s=marker_size, marker=end_marker,  label='end')

    if show_colorbar:
        cb = fig.colorbar(lc, ax=ax, pad=0.1)
        cb.set_label('time (normalized)')

    # Axis limits
    all_xyz = np.concatenate([tr.transpose(1, 0)
                             for tr in trajs] + [pts], axis=0)
    ax.set_xlim(np.nanmin(all_xyz[:, 0]), np.nanmax(all_xyz[:, 0]))
    ax.set_ylim(np.nanmin(all_xyz[:, 1]), np.nanmax(all_xyz[:, 1]))
    ax.set_zlim(np.nanmin(all_xyz[:, 2]), np.nanmax(all_xyz[:, 2]))

    if start_end_markers:
        ax.legend(loc='best')
    ax.view_init(azim=view_azim, elev=view_elev)

    plt.tight_layout()
    plt.show()
    return fig, ax
