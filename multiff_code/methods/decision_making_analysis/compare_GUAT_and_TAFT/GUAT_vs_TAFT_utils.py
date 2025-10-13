from planning_analysis.show_planning import nxt_ff_utils
from planning_analysis.show_planning.cur_vs_nxt_ff import find_cvn_utils
from decision_making_analysis.decision_making import decision_making_utils

import os
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


def add_closest_point_on_trajectory_to_cur_ff(trials_df, monkey_information, ff_real_position_sorted):
    if 'cur_ff_index' not in trials_df.columns:
        trials_df['cur_ff_index'] = trials_df['ff_index']
    trials_df[['cur_ff_x', 'cur_ff_y']
              ] = ff_real_position_sorted[trials_df['cur_ff_index'].values]
    list_of_closest_points = []
    for index, row in trials_df.iterrows():
        start_time = row['first_stop_time'] - 1
        end_time = row['last_stop_time'] + 1
        monkey_sub = monkey_information[(monkey_information['time'] >= start_time) & (
            monkey_information['time'] <= end_time)].copy()
        monkey_sub['distance_to_ff'] = np.sqrt(
            (monkey_sub['monkey_x'] - row['cur_ff_x'])**2 + (monkey_sub['monkey_y'] - row['cur_ff_y'])**2)
        closest_point_index = monkey_sub.loc[monkey_sub['distance_to_ff'].idxmin(
        ), 'point_index']
        list_of_closest_points.append(closest_point_index)
    trials_df['closest_point_index_to_cur_ff'] = list_of_closest_points


def add_stop_point_index(trials_df, monkey_information, ff_real_position_sorted):
    # Note: stop_point_index here is not necessarily a real stop!!! But only the point on the trajectory that is closest to the current ff.
    add_closest_point_on_trajectory_to_cur_ff(
        trials_df, monkey_information, ff_real_position_sorted)
    trials_df['stop_point_index'] = trials_df['closest_point_index_to_cur_ff']
    trials_df['stop_time'] = monkey_information.loc[trials_df['stop_point_index'], 'time'].values


def deal_with_duplicated_stop_point_index(
    GUAT_w_ff_df: pd.DataFrame,
    stop_col: str = "stop_point_index",
    first_col: str = "first_stop_point_index",
    last_col: str = "last_stop_point_index",
    warn: bool = True,
) -> pd.DataFrame:
    """
    De-duplicate rows that share the same `stop_col` by keeping, for each stop,
    the row with the smallest distance (in index units) to either the first or last stop.

    Tie-breakers (in order): min_delta, then delta to first stop, then delta to last stop,
    then first occurrence.

    Parameters
    ----------
    GUAT_w_ff_df : pd.DataFrame
        Must include columns: stop_col, first_col, last_col.
    stop_col : str
        Column with the (possibly duplicated) stop indices to resolve.
    first_col, last_col : str
        Columns with the first/last stop indices for computing deltas.
    warn : bool
        If True, prints a note about the behavior.

    Returns
    -------
    pd.DataFrame
        Same columns as input, with at most one row per `stop_col`.
    """
    required = {stop_col, first_col, last_col}
    missing = required - set(GUAT_w_ff_df.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    df = GUAT_w_ff_df.copy()

    # Identify duplicates by STOP ONLY (matches your comment/intent).
    dup_mask = df.duplicated(subset=[stop_col], keep=False)
    if not dup_mask.any():
        return df.sort_values(by=stop_col, kind="stable").reset_index(drop=True)

    # Split into duplicated and unique parts
    dups = df.loc[dup_mask].copy()
    base = df.loc[~dup_mask].copy()

    # Ensure numeric for distance computation; NaNs become inf so they never win
    for c in (stop_col, first_col, last_col):
        dups[c] = pd.to_numeric(dups[c], errors="coerce")

    dups["delta_first"] = (dups[first_col] - dups[stop_col]).abs()
    dups["delta_last"] = (dups[last_col] - dups[stop_col]).abs()
    dups[["delta_first", "delta_last"]] = dups[[
        "delta_first", "delta_last"]].fillna(np.inf)
    dups["min_delta"] = dups[["delta_first", "delta_last"]].min(axis=1)

    # Pick the best row per stop using deterministic tie-breakers
    best = (
        dups.sort_values(
            by=[stop_col, "min_delta", "delta_first", "delta_last"],
            kind="stable"
        )
        .groupby(stop_col, as_index=False, sort=False)
        .head(1)
    )

    # Recombine; preserve original columns/order
    out = pd.concat([base, best[df.columns]], ignore_index=True)
    out = out.sort_values(by=stop_col, kind="stable").reset_index(drop=True)

    if warn:
        print(
            "Note: multiple stop clusters may map to the same firefly around the same time.\n"
            "We keep the cluster whose stop_point_index is closest (by index distance) to either "
            f"{first_col} or {last_col}. Also, stop_point_index may be just the trajectory point "
            "closest to the current FF, not necessarily a 'true' stop."
        )

    return out


def process_trials_df(trials_df, monkey_information, ff_dataframe, ff_real_position_sorted, stop_period_duration):

    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    processed_df = trials_df[['stop_point_index', 'ff_index']].copy()
    processed_df[['stop_time', 'stop_cum_distance']
                 ] = monkey_information.loc[processed_df.stop_point_index, ['time', 'cum_distance']].values

    processed_df['beginning_time'] = processed_df['stop_time'] - \
        stop_period_duration
    ff_first_and_last_seen_info = nxt_ff_utils.get_ff_first_and_last_seen_info(processed_df['ff_index'].values,
                                                                               processed_df['stop_point_index'].values,
                                                                               processed_df['beginning_time'],
                                                                               processed_df['stop_time'],
                                                                               ff_dataframe_visible,
                                                                               monkey_information)

    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                      'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen',
                      'ff_distance_ff_first_seen', 'ff_distance_ff_last_seen',
                      'ff_angle_ff_first_seen', 'ff_angle_ff_last_seen',
                      'ff_angle_boundary_ff_first_seen', 'ff_angle_boundary_ff_last_seen',
                      'time_ff_first_seen', 'time_ff_last_seen']

    columns_to_add = [
        col for col in columns_to_add if col not in processed_df.columns]
    ff_first_and_last_seen_info = ff_first_and_last_seen_info[columns_to_add + [
        'ff_index', 'stop_point_index']]
    # columns_to_be_renamed_dict = {column: 'NXT_' + column + '_bbas' for column in columns_to_add}
    # ff_first_and_last_seen_info.rename(columns=columns_to_be_renamed_dict, inplace=True)
    processed_df = processed_df.merge(
        ff_first_and_last_seen_info, on=['stop_point_index', 'ff_index'], how='left')

    processed_df['time_since_ff_first_seen'] = processed_df['stop_time'] - \
        processed_df['time_ff_first_seen']
    processed_df['time_since_ff_last_seen'] = processed_df['stop_time'] - \
        processed_df['time_ff_last_seen']

    # also add ff info from the stop point
    ff_info = decision_making_utils.find_many_ff_info_anew(
        processed_df['ff_index'].values, processed_df['stop_point_index'].values,
        ff_real_position_sorted, ff_dataframe_visible, monkey_information)

    more_columns_to_add = ['ff_distance', 'ff_angle',
                           'ff_angle_boundary', 'time_since_last_vis']
    # Note: time_since_last_vis should equal to time_since_ff_last_seen. We shall check it.
    more_columns_to_add = [
        col for col in more_columns_to_add if col not in processed_df.columns]
    ff_info.rename(columns={'point_index': 'stop_point_index'}, inplace=True)
    ff_info = ff_info[more_columns_to_add + ['stop_point_index', 'ff_index']]
    processed_df = processed_df.merge(
        ff_info, on=['stop_point_index', 'ff_index'], how='left')
    return processed_df


def further_make_trials_df(processed_df, monkey_information, ff_real_position_sorted, stop_period_duration, ref_point_mode, ref_point_value):
    processed_df2 = find_cvn_utils.find_ff_info_based_on_ref_point(processed_df, monkey_information, ff_real_position_sorted,
                                                                   ref_point_mode=ref_point_mode, ref_point_value=ref_point_value)

    processed_df2.rename(
        columns={'point_index': 'ref_point_index'}, inplace=True)
    processed_df2['ref_time'] = monkey_information.loc[processed_df2['ref_point_index'], 'time'].values

    processed_df2['stop_time'] = monkey_information.loc[processed_df2['stop_point_index'], 'time'].values
    processed_df2['beginning_time'] = processed_df2['stop_time'] - \
        stop_period_duration
    return processed_df2


def combine_relevant_features(x_features_df, only_cur_ff_df, plan_features_df, drop_columns_w_na=False):
    x_features_df = x_features_df[non_cluster_columns_to_keep_from_x_features_df + ['stop_point_index'] +
                                  [col for col in x_features_df.columns if 'cluster' in col]].copy()
    only_cur_ff_df = only_cur_ff_df[columns_to_keep_from_only_cur_ff_df + [
        'stop_point_index']].copy()
    plan_features_df = plan_features_df[non_cluster_columns_to_keep_from_plan_features_df + ['stop_point_index'] +
                                        [col for col in plan_features_df.columns if 'cluster' in col]].copy()
    x_df = pd.merge(x_features_df, only_cur_ff_df,
                    on='stop_point_index', how='inner')
    x_df = pd.merge(x_df, plan_features_df, on='stop_point_index', how='inner')

    # drop columns with NA and print the names of these columns
    columns_with_na = x_df.columns[x_df.isna().any()].tolist()
    if drop_columns_w_na:
        x_df = x_df.drop(columns=columns_with_na)
        print(
            f'When preparing x_var to predict whether TAFT or GUAT, there are {len(columns_with_na)} columns with NA that are dropped. {x_df.shape[1]} columns are left.')
        print('Columns with NA that are dropped:', np.array(columns_with_na))

    else:
        print(
            f'When preparing x_var to predict whether TAFT or GUAT, there are {len(columns_with_na)} out of {x_df.shape[1]} columns with NA.')

    print('The shape of x_df is:', x_df.shape)

    x_df['dir_from_cur_ff_same_side'] = x_df['dir_from_cur_ff_same_side'].astype(
        int)

    # drop duplicated columns
    x_df = x_df.loc[:, ~x_df.columns.duplicated()]

    return x_df


columns_to_keep_from_only_cur_ff_df = ['opt_arc_curv',
                                       'opt_arc_measure',
                                       'opt_arc_d_heading',
                                       'cntr_arc_curv',
                                       # 'cntr_arc_d_heading', # this is perfectly correlated with cur_ff_angle_at_ref
                                       # 'diff_in_d_heading_to_cur_ff',
                                       # the columns below are repeated later
                                       # 'curv_of_traj',
                                       # 'curv_of_traj_before_stop',
                                       # 'd_heading_of_traj',
                                       # 'd_heading_of_traj',
                                       # 'curv_mean', 'curv_std', 'curv_min', 'curv_Q1', 'curv_median', 'curv_Q3',
                                       # 'curv_max', 'curv_iqr', 'curv_range',
                                       ]


non_cluster_columns_to_keep_from_x_features_df = [
    'cur_ff_angle_boundary_at_ref', 'cur_ff_angle_diff_boundary_at_ref',
    'cur_ff_flash_duration_at_ref',
    'cur_ff_earliest_flash_rel_time_at_ref',
    'cur_ff_latest_flash_rel_time_at_ref']

non_cluster_columns_to_keep_from_plan_features_df = ['d_from_cur_ff_to_nxt_ff',
                                                     'distance_between_stop_and_arena_edge',
                                                     'cum_distance_between_two_stops',
                                                     'time_between_two_stops',
                                                     'nxt_ff_distance_at_ref',
                                                     'nxt_ff_angle_at_ref',
                                                     'cur_ff_distance_at_ref',
                                                     'cur_ff_angle_at_ref',
                                                     'cur_ff_angle_boundary_at_ref',
                                                     'speed_std',
                                                     'ang_speed_std',
                                                     'speed_range',
                                                     'speed_iqr',
                                                     'ang_speed_range',
                                                     'ang_speed_iqr',
                                                     'left_eye_cur_ff_time_perc_5',
                                                     'left_eye_cur_ff_time_perc_10',
                                                     'left_eye_nxt_ff_time_perc_5',
                                                     'left_eye_nxt_ff_time_perc_10',
                                                     'right_eye_cur_ff_time_perc_5',
                                                     'right_eye_cur_ff_time_perc_10',
                                                     'right_eye_nxt_ff_time_perc_5',
                                                     'right_eye_nxt_ff_time_perc_10',
                                                     'LDy_std',
                                                     'LDz_std',
                                                     'RDy_std',
                                                     'RDz_std',
                                                     'LDy_range',
                                                     'LDy_iqr',
                                                     'LDz_range',
                                                     'LDz_iqr',
                                                     'RDy_range',
                                                     'RDy_iqr',
                                                     'RDz_range',
                                                     'RDz_iqr',
                                                     'cur_ff_cluster_50_size',
                                                     'monkey_angle_before_stop',
                                                     #  'NXT_time_ff_last_seen_bbas_rel_to_stop',
                                                     #  'NXT_time_ff_last_seen_bsans_rel_to_stop',
                                                     #  'nxt_ff_last_flash_time_bbas_rel_to_stop',
                                                     #  'nxt_ff_last_flash_time_bsans_rel_to_stop',
                                                     #  'nxt_ff_cluster_last_seen_time_bbas_rel_to_stop',
                                                     #  'nxt_ff_cluster_last_seen_time_bsans_rel_to_stop',
                                                     #  'nxt_ff_cluster_last_flash_time_bbas_rel_to_stop',
                                                     #  'nxt_ff_cluster_last_flash_time_bsans_rel_to_stop',
                                                     'd_heading_of_traj',
                                                     'ref_curv_of_traj',
                                                     # 'angle_from_m_before_stop_to_cur_ff',
                                                     'angle_from_stop_to_nxt_ff',
                                                     # 'angle_from_cur_ff_to_stop',
                                                     'angle_from_cur_ff_to_nxt_ff',
                                                     'curv_mean',
                                                     'curv_std',
                                                     'curv_min',
                                                     'curv_Q1',
                                                     'curv_median',
                                                     'curv_Q3',
                                                     'curv_max',
                                                     #  'curv_iqr', # this will cause perfect correlation
                                                     #  'curv_range', # this will cause perfect correlation
                                                     'curv_of_traj_before_stop',
                                                     # 'dir_from_cur_ff_to_stop', # this has high correlation with dir_from_cur_ff_to_nxt_ff
                                                     'dir_from_cur_ff_to_nxt_ff',
                                                     'dir_from_cur_ff_same_side']
