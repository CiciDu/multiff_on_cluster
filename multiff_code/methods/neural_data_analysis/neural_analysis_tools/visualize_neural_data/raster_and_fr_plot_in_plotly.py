import logging
import numpy as np
import plotly.graph_objects as go

from neural_data_analysis.neural_analysis_tools.get_neural_data import neural_data_processing
from neural_data_analysis.neural_analysis_tools.visualize_neural_data import get_colors_utils
from visualization.plotly_tools import plotly_for_time_series
from visualization.plotly_tools.plotly_for_time_series import plot_blocks_to_show_ff_visible_segments_in_fig_time_series
from visualization.plotly_tools import plotly_plot_class


def create_firing_rate_plot_for_one_duration_in_plotly(
        spikes_df, reference_time, start_time, end_time,
        bin_width=0.02, bins_per_aggregate=1,
        max_clusters_to_plot=None, rel_hover_time=None,
        show_visible_segments=False, visible_segments_info=None,
        smoothing_window=15, smoothing_method='gaussian'):

    filtered_spikes, selected_clusters = _select_clusters_and_spikes_in_time_window(
        spikes_df, start_time, end_time, max_clusters_to_plot)
    if filtered_spikes.empty or len(selected_clusters) == 0:
        return go.Figure()

    filtered_spikes = filtered_spikes.copy()
    filtered_spikes['rel_spike_time'] = filtered_spikes['time'] - \
        reference_time

    rel_start_time = start_time - reference_time
    rel_end_time = end_time - reference_time
    time_bins, binned_df = neural_data_processing.prepare_binned_spikes_df(
        filtered_spikes, bin_width=bin_width,
        spike_time_col='rel_spike_time',
        min_time=rel_start_time, max_time=rel_end_time
    )
    time_array = (time_bins[:-1] + time_bins[1:]) / 2
    fr_df = _prepare_fr_data(binned_df, bin_width,
                             bins_per_aggregate, time_array=time_array,
                             smoothing_window=smoothing_window,
                             smoothing_method=smoothing_method)

    selected_cluster_cols = [f"cluster_{c}" for c in selected_clusters]
    if 'time' not in fr_df.columns or not set(selected_cluster_cols).intersection(fr_df.columns):
        logging.warning(
            "Prepared firing rate DataFrame missing expected columns.")
        return go.Figure()

    fig = go.Figure()

    for c in selected_clusters:
        cluster_col = f"cluster_{c}"
        if cluster_col in fr_df.columns:
            fig.add_trace(go.Scatter(
                x=fr_df['time'],
                y=fr_df[cluster_col],
                mode='lines',
                name=f'Cluster {c}',
                legendgroup=f'cluster-{c}',
                line=dict(color=get_colors_utils._color_for_cluster(
                    c), width=1.5, shape='linear'),
                opacity=0.8,
                customdata=[c] * len(fr_df),
                hovertemplate='<b>Cluster %{customdata}</b><br>'
                              'Time: %{x:.3f}s<br>'
                              'Firing Rate: %{y:.2f} Hz<extra></extra>'
            ))

    if len(selected_cluster_cols) > 1:
        mean_fr = fr_df[selected_cluster_cols].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=fr_df['time'],
            y=mean_fr,
            mode='lines',
            name='Population Mean',
            line=dict(color='#d62728', width=3, dash='solid'),
            opacity=0.9,
            hovertemplate='<b>Population Mean</b><br>'
                          'Time: %{x:.3f}s<br>'
                          'Firing Rate: %{y:.2f} Hz<extra></extra>'
        ))

    y_min = 0
    y_max = fr_df[selected_cluster_cols].max().max()
    y_max = (y_max * 1.1) if y_max != 0 else 1

    _add_reference_lines(fig, y_min, y_max, rel_hover_time)
    fig = _add_firefly_segments(
        fig, y_min, y_max, show_visible_segments, visible_segments_info)
    fig.update_layout(_common_layout('Firing Rate Over Time',
                      'Time (s) relative to reference', 'Firing Rate (Hz)', [y_min, y_max]))

    return fig


def create_raster_plot_for_one_duration_in_plotly(
    spikes_df, reference_time, start_time, end_time,
    max_clusters_to_plot=None, rel_hover_time=None,
    show_visible_segments=False, visible_segments_info=None
):
    filtered_spikes, selected_clusters = _select_clusters_and_spikes_in_time_window(
        spikes_df, start_time, end_time, max_clusters_to_plot
    )
    if filtered_spikes.empty:
        return go.Figure()

    filtered_spikes = filtered_spikes.copy()
    filtered_spikes['rel_spike_time'] = filtered_spikes['time'] - \
        reference_time

    fig = go.Figure()

    # Use stable color per cluster ID
    for cluster_id in selected_clusters:
        group = filtered_spikes[filtered_spikes['cluster'] == cluster_id]
        if not group.empty:
            c = get_colors_utils._color_for_cluster(
                cluster_id)  # <- stable, normalized color
            fig.add_trace(go.Scatter(
                x=group['rel_spike_time'],
                y=[cluster_id] * len(group),
                mode='markers',
                marker=dict(
                    size=4,
                    color=c,
                    opacity=0.8,
                    # thin halo for visibility
                    line=dict(width=0.5, color='white')
                ),
                name=f'Cluster {cluster_id}',
                legendgroup=f'cluster-{cluster_id}',
                showlegend=False,  # keep raster legend clean; toggle to True if you want a legend
                customdata=[cluster_id] * len(group),
                hovertemplate="%{customdata}<extra></extra>"
            ))

    y_min, y_max = filtered_spikes['cluster'].min(
    ), filtered_spikes['cluster'].max()
    y_range = y_max - y_min
    if y_range == 0:
        y_min -= 0.5
        y_max += 0.5
    else:
        y_min -= y_range * 0.1
        y_max += y_range * 0.1

    _add_reference_lines(fig, y_min, y_max, rel_hover_time)
    fig = _add_firefly_segments(
        fig, y_min, y_max, show_visible_segments, visible_segments_info)
    fig.update_layout(_common_layout(
        'Neural Raster Plot',
        'Time (s) relative to reference',
        'Neuron ID',
        [y_min, y_max]
    ))

    return fig


def _add_reference_lines(fig, y_min, y_max, rel_hover_time=None):
    # We decided to not add a line at 0 to avoid distraction.
    # _add_vertical_line(fig, 0, y_min, y_max,
    #                    color=plotly_for_time_series.GUIDE_LINE_COLOR,
    #                    name='Reference Time', showlegend=False, width=2, dash='solid')
    if rel_hover_time is not None:
        _add_vertical_line(fig, rel_hover_time, y_min, y_max,
                           color=plotly_for_time_series.HOVER_LINE_COLOR,
                           dash='dash', showlegend=False, name='Hover Time', width=2)


def _add_firefly_segments(fig, y_min, y_max, show_visible_segments, visible_segments_info):
    if show_visible_segments and visible_segments_info is not None:
        stops_near_ff_row = visible_segments_info['stops_near_ff_row']
        fig = plot_blocks_to_show_ff_visible_segments_in_fig_time_series(
            fig, visible_segments_info['ff_info'], visible_segments_info['monkey_information'],
            stops_near_ff_row,
            unique_ff_indices=[stops_near_ff_row.cur_ff_index,
                               stops_near_ff_row.nxt_ff_index],
            time_or_distance='time', y_range_for_v_line=[y_min, y_max],
            varying_colors=[plotly_plot_class.PlotlyPlotter.cur_ff_color,
                            plotly_plot_class.PlotlyPlotter.nxt_ff_color],
            ff_names=['cur ff', 'nxt ff'],
            block_opacity=0.1,
            show_annotation=False
        )
    return fig


def _common_layout(title, xaxis_title, yaxis_title, y_range):
    return dict(
        hovermode='closest',
        hoverdistance=-1,
        title=dict(text=title, font=dict(size=14, color='#2c3e50'), x=0.5),
        xaxis=dict(title=xaxis_title, showgrid=True, zeroline=True),
        yaxis=dict(title=yaxis_title, range=y_range),
        height=300,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=30, t=50, b=60),
    )


def _select_clusters_and_spikes_in_time_window(spikes_df, start_time, end_time, max_clusters_to_plot=None):
    """
    Select spikes within a time window and limit clusters.

    Args:
        spikes_df: DataFrame with spike data containing 'cluster' and 'time'.
        start_time: Start time of the window.
        end_time: End time of the window.
        max_clusters_to_plot: Max number of clusters to include.

    Returns:
        Tuple of (filtered spikes DataFrame, array of selected clusters).
    """
    filtered_spikes = spikes_df[
        (spikes_df['time'] >= start_time) &
        (spikes_df['time'] <= end_time)
    ].copy()

    unique_clusters = filtered_spikes['cluster'].unique()
    if max_clusters_to_plot and len(unique_clusters) > max_clusters_to_plot:
        selected_clusters = np.random.choice(
            unique_clusters, max_clusters_to_plot, replace=False)
    else:
        selected_clusters = unique_clusters
    selected_clusters = np.sort(selected_clusters)

    filtered_spikes = filtered_spikes[filtered_spikes['cluster'].isin(
        selected_clusters)].copy()
    return filtered_spikes, selected_clusters


def _add_vertical_line(fig, x_val, y_min, y_max, color, width=2, dash=None, name='', showlegend=True):
    """
    Helper to add vertical lines to plotly figures.

    Args:
        fig: Plotly figure object.
        x_val: X coordinate for the vertical line.
        y_min: Minimum y-value for line span.
        y_max: Maximum y-value for line span.
        color: Line color.
        width: Line width.
        dash: Line dash style ('dash', 'dot', etc.).
        name: Legend name.
        showlegend: Whether to show in legend.
    """

    fig.add_trace(go.Scatter(
        x=[x_val, x_val],
        y=[y_min, y_max],
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        showlegend=showlegend,
        hoverinfo='skip'
    ))


def _prepare_fr_data(binned_df, bin_width, bins_per_aggregate, time_array=None, max_time=None,
                     smoothing_window=None, smoothing_method='gaussian'):
    """
    Aggregate binned spike counts into firing rates averaged over bins_per_aggregate bins.

    Args:
        binned_df: DataFrame where each column is cluster spike counts per bin, and index corresponds to bins.
        bin_width: Width of each bin in seconds.
        bins_per_aggregate: Number of bins to average (downsample factor).
        time_array: Optional array of times corresponding to bins.
        max_time: Optional max time to filter data.
        smoothing_window: Optional window size for smoothing (number of bins). If None, no smoothing applied.
        smoothing_method: Smoothing method ('gaussian', 'uniform', 'exponential'). Default is 'gaussian'.

    Returns:
        DataFrame with averaged firing rates in Hz and 'time' column.
    """
    df = binned_df.copy()

    # Assign time column
    if time_array is None:
        df['time'] = df.index * bin_width
    else:
        if len(time_array) != len(df):
            logging.warning(
                f"Length of time_array ({len(time_array)}) does not match binned_df ({len(df)}), adjusting time_array.")
            df['time'] = np.linspace(
                time_array[0], time_array[-1], num=len(df), endpoint=False)
        else:
            df['time'] = time_array

    # Filter by max_time if specified
    if max_time is not None:
        df = df[df['time'] <= max_time]

    # Group bins to aggregate over bins_per_aggregate
    df['agg_bin'] = np.arange(len(df)) // bins_per_aggregate

    # Prepare aggregation dict: average all spike count columns plus time
    agg_dict = {col: 'mean' for col in df.columns if col not in ['agg_bin']}
    df_agg = df.groupby('agg_bin').agg(agg_dict).reset_index(drop=True)

    # Convert spike counts to firing rates in Hz by dividing by bin width
    # Note: When averaging over bins_per_aggregate bins, the effective bin width is bin_width * bins_per_aggregate
    effective_bin_width = bin_width * bins_per_aggregate

    # Convert all cluster columns to firing rates (Hz)
    cluster_cols = [
        col for col in df_agg.columns if col.startswith('cluster_')]
    for col in cluster_cols:
        df_agg[col] = df_agg[col] / effective_bin_width

    # Apply smoothing if requested
    if smoothing_window is not None and smoothing_window > 1:
        df_agg = _apply_smoothing(
            df_agg, cluster_cols, smoothing_window, smoothing_method)

    return df_agg


def _apply_smoothing(df, cluster_cols, window_size, method='gaussian', pad_mode='reflect'):
    """
    Apply smoothing to firing rate data.

    Args:
        df: DataFrame with firing rate data (already binned)
        cluster_cols: List[str] of columns to smooth
        window_size: int, width in bins (Gaussian covers ~±3σ, so FWHM≈0.392*window_size)
        method: 'gaussian' (acausal), 'uniform' (acausal), 'exponential' (causal EMA)
        pad_mode: np.pad mode for acausal methods ('reflect', 'edge', ...)

    Returns:
        DataFrame with smoothed firing rates
    """
    import numpy as np
    import pandas as pd

    ws = int(max(1, round(window_size)))
    df_smoothed = df.copy()

    if method == 'gaussian':
        # sigma chosen so 6σ ≈ window_size  (≈99% mass)
        sigma = ws / 6.0
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # odd
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= kernel.sum()

    elif method == 'uniform':
        kernel_size = ws if ws % 2 == 1 else ws + 1
        kernel = np.ones(kernel_size) / kernel_size

    elif method == 'exponential':
        # causal EMA; span relates to alpha via alpha = 2/(span+1)
        span = ws
        for col in cluster_cols:
            df_smoothed[col] = (
                df[col]
                .ewm(span=span, adjust=False, min_periods=1)  # causal
                .mean()
            )
        return df_smoothed

    else:
        raise ValueError("method must be 'gaussian', 'uniform', or 'exponential'")

    # Apply centered (acausal) convolution for gaussian/uniform
    half = kernel.size // 2
    for col in cluster_cols:
        # choose how to treat NaNs: e.g., fillna(0) or interpolate if needed
        vals = df[col].values
        padded = np.pad(vals, (half, half), mode=pad_mode)
        smoothed = np.convolve(padded, kernel, mode='valid')
        df_smoothed[col] = smoothed

    return df_smoothed
