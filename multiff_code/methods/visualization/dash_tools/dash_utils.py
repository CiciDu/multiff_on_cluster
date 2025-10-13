from null_behaviors import curvature_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go
import matplotlib
import os
import socket
from contextlib import closing

plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)

# drop down: https://dash.plotly.com/dash-core-components/dropdown

# input: https://dash.plotly.com/dash-core-components/input

# callback_context: https://dash.plotly.com/advanced-callbacks


def put_down_the_refreshing_buttons_for_ref_point_and_curv_of_traj(ids=['update_ref_point', 'update_curv_of_traj']):
    return html.Div([
                    html.Button('Update reference point', id=ids[0], n_clicks=0,
                                style={'width': '50%', 'background-color': '#7fa982', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'}),
                    html.Button('Update curvature of trajectory', id=ids[1], n_clicks=0,
                                style={'width': '50%', 'background-color': '#90a8b8', 'padding': '0px 10px 10px 10px', 'margin': '0 0 10px 0'}),
                    ], style=dict(display='flex'))


def put_down_the_previous_plot_and_next_plot_button(ids=['previous_plot_button', 'next_plot_button']):
    return html.Div(children=[
                    html.Button('Previous plot', ids[0], n_clicks=0,
                                style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'}),
                    html.Button('Next plot', ids[1], n_clicks=0,
                                style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'}),
                    ], style=dict(display='flex'))


def put_down_the_dropdown_menu_for_ref_point_mode(ref_point_mode, id='ref_point_mode'):
    return html.Div(children=[
        html.Label(['Reference Point From Stopping Point'], style={
                   'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(
            id=id,
            options=['distance', 'time'],
            value=ref_point_mode,
            searchable=False,
            multi=False,
            # placeholder="Select a mode to determine the reference point",
        )
    ],
        style={'width': '40%', 'padding': '10px 10px 10px 10px', 'background-color': '#9FD4A3'})  # light green


def put_down_the_input_for_ref_point_descr(ref_point_value, id="ref_point_value"):
    return html.Div(
        dcc.Input(id=id,
                  type="number",
                  placeholder="value",
                  debounce=True,
                  value=ref_point_value),
        # make it look better
        style={'width': '25%', 'padding': '10px 10px 10px 10px',
               'background-color': '#9FD4A3'}
    )


def put_down_the_dropdown_menu_for_curv_of_traj_mode(curv_of_traj_mode, label='Curvature of Trajectory', id='curv_of_traj_mode'):
    return html.Div(children=[
        html.Label([label], style={
                   'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(
            id=id,
            options=['time', 'distance', 'now to stop'],
            value=curv_of_traj_mode,
            searchable=False,
            multi=False,
            # placeholder="Select a mode to calculate curvature of trajectory",
        )
    ],
        style={'width': '40%', 'padding': '10px 10px 10px 10px', 'background-color': '#B5D3E7'})  # light blue


def put_down_the_input_for_window_lower_end_and_upper_end(window_for_curv_of_traj, ids=['window_lower_end', 'window_upper_end']):
    if window_for_curv_of_traj is None:
        lower_end = None
        upper_end = None
    else:
        lower_end = window_for_curv_of_traj[0]
        upper_end = window_for_curv_of_traj[1]
    return html.Div(
        [dcc.Input(id=ids[0], type="number", placeholder="window lower end", debounce=True, value=lower_end),
         dcc.Input(id=ids[1], type="number", placeholder="window upper end",
                   debounce=True, value=upper_end),
         ],
        # make it look better
        style={'width': '25%', 'padding': '10px 10px 10px 10px',
               'background-color': '#B5D3E7'}
    )


def put_down_time_series_plot(fig_time_series, id='curv_of_traj_vs_time', padding='0 0 10 0'):
    # it looks like changing padding does not work well.
    return html.Div([
        dcc.Graph(
                    id=id,
                    figure=fig_time_series),
    ], style={'width': '60%', 'padding': padding,  # 'display': 'inline-block',
              })


def put_down_empty_plot_that_takes_no_space(id='empty_plot'):
    empty_plot = go.Figure()

    # Set the height and width of the plot to zero
    empty_plot.update_layout(height=10, width=10)

    return html.Div([
        dcc.Graph(
                    id=id,
                    figure=empty_plot),
        # 'display': 'inline-block',
    ], style={'height': '0px', 'width': '0px', 'padding': '0 0 0 0',
              })


def put_down_both_fig_time_series_separately(fig_time_series_cm, fig_time_series_s, hoverdata_column):
    if hoverdata_column == 'rel_time':
        # fig_time_series_s.update_layout(showlegend=False)
        return html.Div([put_down_time_series_plot(fig_time_series_cm, id='time_series_plot_top'),
                        put_down_time_series_plot(fig_time_series_s, id='time_series_bottom')])

    else:
        # fig_time_series_cm.update_layout(showlegend=False)
        return html.Div([put_down_time_series_plot(fig_time_series_s, id='time_series_plot_top'),
                        put_down_time_series_plot(fig_time_series_cm, id='time_series_bottom')])


# def _put_down_correlation_plots_in_dash(fig_scatter_or_reg, fig_scatter_or_reg2=None, id=None, id_2='correlation_plot_2'):
#     if fig_scatter_or_reg2 is None:
#         correlation_plots_in_dash = plotly_for_correlation.put_down_correlation_plot(fig_scatter_or_reg)
#     else:
#         correlation_plots_in_dash = html.Div([html.Button('Refresh shuffled plot on the right', id='refresh_correlation_plot_2', n_clicks=0,
#                                                     style={'margin': '10px 10px 10px 10px', 'background-color': '#FFC0CB'}),
#                                                 html.Div([plotly_for_correlation.put_down_correlation_plot(fig_scatter_or_reg, id=id, width='50%'),
#                                                         plotly_for_correlation.put_down_correlation_plot(fig_scatter_or_reg2, id=id_2, width='50%')
#                                                         ], style=dict(display='flex')),
#                                             ])
#     return correlation_plots_in_dash


def put_down_monkey_plot(fig, monkey_hoverdata_value, id='monkey_plot'):
    return html.Div([
        dcc.Graph(
                    id=id,
                    figure=fig,
                    # ['Original-Present', 0]}]}
                    hoverData={'points': [
                        {'customdata': monkey_hoverdata_value}]}
                    )
        #  inline-block helps designers create boxes that automatically wrap text and other content to give them space and set them apart when beside other content.
        # 'display': 'inline-block',
    ],  style={'width': '60%', 'padding': '0 0 0 0',
               })


def find_hoverdata_value_upper_bound(row, hoverdata_column):
    if hoverdata_column == 'rel_time':
        hoverdata_value_upper_bound = row.time_before_stop - row.stop_time - 0.1
    elif hoverdata_column == 'rel_distance':
        hoverdata_value_upper_bound = row.distance_before_stop - row.stop_cum_distance - 0.1
    else:
        raise ValueError('hoverdata_column must be rel_time or rel_distance')
    return hoverdata_value_upper_bound


def find_hoverdata_multi_columns(hoverdata_column):
    if hoverdata_column == 'rel_time':
        hoverdata_multi_columns = ['rel_time', 'rel_distance']
    elif hoverdata_column == 'rel_distance':
        hoverdata_multi_columns = ['rel_distance', 'rel_time']
    else:
        hoverdata_multi_columns = [hoverdata_column]
    return hoverdata_multi_columns


def show_a_static_plot(fig_time_series):
    fig_time_series = go.Figure(fig_time_series)
    fig_time_series.update_layout(
        yaxis=dict(range=['null', 'null'],),
        yaxis2=dict(range=['null', 'null'],),)
    fig_time_series.show()
    return


def update_marked_traj_portion_in_monkey_plot(fig, traj_portion, hoverdata_multi_columns=['rel_time']):
    fig.update_traces(overwrite=True, marker={"size": 9, "opacity": 1, "color": 'purple'},
                      x=traj_portion['monkey_x'].values, y=traj_portion['monkey_y'].values,
                      customdata=traj_portion[hoverdata_multi_columns].values,
                      selector=dict(name="trajectory before stop"))

    return fig


def update_fig_time_series_combd_plot_based_on_monkey_hoverdata(fig_time_series_combd, current_hoverdata_value_s, current_hoverdata_value_cm):
    fig_time_series_combd.update_traces(overwrite=True, marker={"opacity": 0.4}, selector=dict(name='Monkey trajectory hover position'),
                                        x=[current_hoverdata_value_s, current_hoverdata_value_s])
    fig_time_series_combd.update_traces(overwrite=True, marker={"opacity": 0.4}, selector=dict(name='time_series_cm_' + 'Monkey trajectory hover position'),
                                        x=[current_hoverdata_value_cm, current_hoverdata_value_cm])
    return fig_time_series_combd


# def make_the_update_functions_for_updating_window(app):
#     @app.callback(
#         Output('window_lower_end', 'value'),
#         Output('window_upper_end', 'value'),
#         Input('curv_of_traj_mode', 'value'),
#         prevent_initial_call=True
#     )
#     def update_window(value):
#         return 0, 0

#     @app.callback(
#         Output('window_upper_end', 'value', allow_duplicate=True),
#         Input('window_lower_end', 'value'),
#         State('window_upper_end', 'value'),
#         prevent_initial_call=True
#     )
#     def update_window2(lower_end, upper_end):
#         if upper_end is not None:
#             if upper_end != 0:
#                 return lower_end
#         raise PreventUpdate
#     return


def create_error_message_display(id_prefix='main_plots_'):
    return html.Div([
        dcc.Loading(
            id=id_prefix+"error_message",
            type="default",
            children='Updated successfully',
            style={'padding': '10px 10px 10px 10px', 'color': 'purple'}
        )],
        style={'width': '75%', 'padding': '10px 10px 10px 10px',
               'background-color': 'orange'}
    )


def print_other_messages(id_prefix='main_plots_', other_messages=''):
    return html.Div([
        dcc.Loading(
            id=id_prefix+"other_messages",
            type="default",
            children=other_messages,
            style={'padding': '10px 10px 10px 10px', 'color': 'purple'}
        )],
        style={'width': '90%', 'padding': '10px 10px 10px 10px',
               'margin': '10px 0 10px 0',
               'background-color': '#9FD4A3'}
    )


def _find_traj_portion_for_traj_curv(trajectory_df, curv_of_traj_current_row):
    traj_portion = trajectory_df[(trajectory_df.point_index >= curv_of_traj_current_row['min_point_index'].item()) & (
        trajectory_df.point_index <= curv_of_traj_current_row['max_point_index'].item())]
    traj_length = traj_portion['rel_distance'].iloc[-1] - \
        traj_portion['rel_distance'].iloc[0]
    return traj_portion, traj_length


def _find_nxt_ff_curv_df(current_plotly_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    nxt_ff_curv_df = _find_ff_curv_df(current_plotly_key_comp['row'].nxt_ff_index, current_plotly_key_comp,
                                      ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new)
    return nxt_ff_curv_df


def _find_cur_ff_curv_df(current_plotly_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    cur_ff_curv_df = _find_ff_curv_df(current_plotly_key_comp['row'].cur_ff_index, current_plotly_key_comp,
                                      ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=ff_caught_T_new)
    return cur_ff_curv_df


def _find_ff_curv_df(ff_index, current_plotly_key_comp, ff_dataframe, monkey_information, curv_of_traj_df, ff_caught_T_new=None):
    duration_to_plot = current_plotly_key_comp['duration_to_plot']
    row = current_plotly_key_comp['row']
    ff_curv_df = curvature_utils.find_curvature_df_for_ff_in_duration(
        ff_dataframe, ff_index, duration_to_plot, monkey_information, curv_of_traj_df,  ff_caught_T_new=ff_caught_T_new, clean=True)
    ff_curv_df['rel_time'] = ff_curv_df['time'] - row.stop_time
    ff_curv_df = ff_curv_df.merge(
        monkey_information[['point_index', 'cum_distance']], on='point_index', how='left')
    ff_curv_df['rel_distance'] = np.round(
        ff_curv_df['cum_distance'] - row.stop_cum_distance, 2)
    ff_curv_df['cntr_arc_curv'] = ff_curv_df['cntr_arc_curv'] * \
        180/np.pi * 100  # convert to degree/cm
    ff_curv_df['opt_arc_curv'] = ff_curv_df['opt_arc_curv'] * \
        180/np.pi * 100  # convert to degree/cm
    ff_curv_df_sub = ff_curv_df[['point_index', 'rel_time',
                                 'rel_distance', 'cntr_arc_curv', 'opt_arc_curv']].copy()
    return ff_curv_df_sub


def put_down_firing_rate_plot(fig_firing_rate, id='firing_rate_plot', width='50%', height=300):
    """Create a firing rate plot component for Dash."""
    if id is None:
        id = 'firing_rate_plot'

    # Update the figure height if provided
    if height is not None:
        fig_firing_rate.update_layout(height=height)

    return html.Div([
        dcc.Graph(
            id=id,
            figure=fig_firing_rate,
        ),
    ], style={'width': width, 'padding': '0 0 10 0'})


def put_down_raster_plot(fig_raster, id='raster_plot', width='50%', height=300):
    """Create a raster plot component for Dash."""
    if id is None:
        id = 'raster_plot'

    # Update the figure height if provided
    if height is not None:
        fig_raster.update_layout(height=height)

    return html.Div([
        dcc.Graph(
            id=id,
            figure=fig_raster,
        ),
    ], style={'width': width, 'padding': '0 0 10 0'})



def _find_open_port(preferred: int | None, max_tries: int = 50) -> int:
    """
    Try preferred first (or $PORT if set and preferred is None), then scan upward.
    Returns an open port number or raises RuntimeError if none found.
    """
    candidates = []
    if preferred is not None:
        candidates.append(int(preferred))
    env_port = os.getenv("PORT")
    if env_port is not None and (preferred is None or int(env_port) != int(preferred)):
        candidates.append(int(env_port))

    # If nothing specified, start from a sensible default (8050) then scan upward.
    if not candidates:
        candidates = [8050]

    tried = set()
    for base in candidates:
        # try base, then base+1 ... base+max_tries
        print(f"Trying port {base} and above up to {base + max_tries}")
        for p in range(base, base + max_tries + 1):
            if p in tried:
                continue
            tried.add(p)
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("0.0.0.0", p))
                    # success: immediately free it; we’ll reuse in run()
                    return p
                except OSError:
                    continue
    raise RuntimeError(
        f"Could not find a free port near {candidates[0]} after {max_tries} attempts."
    )
