
from data_wrangling import general_utils
from pattern_discovery import cluster_analysis
from pattern_discovery import monkey_landing_in_ff

import pandas as pd
import numpy as np


def get_all_nxt_ff_df_from_ff_dataframe(
    stops_near_ff_df, ff_dataframe_visible, closest_stop_to_capture_df,
    ff_real_position_sorted, ff_caught_T_new, ff_life_sorted, monkey_information,
    min_time_between_cur_and_nxt_ff_caught_time=0.1,
    min_distance_between_cur_and_nxt_ff=25,
    max_distance_between_cur_and_nxt_ff=500,
    stop_period_duration=2
):
    # Setup and prep
    df = stops_near_ff_df[[
        'cur_ff_index', 'stop_point_index', 'stop_cum_distance',
        'stop_time', 'stop_x', 'stop_y'
    ]].copy()
    df['beginning_time'] = df['stop_time'] - stop_period_duration
    df['data_category_by_vis'] = 'test'
    df['nxt_ff_index'] = df['cur_ff_index'] + 1

    # Filter valid next fireflies
    df = df[df['nxt_ff_index'] < len(ff_caught_T_new)].copy()
    df['nxt_ff_caught_time'] = ff_caught_T_new[df['nxt_ff_index'].values]

    # Add stop info of next firefly
    next_stop_df = closest_stop_to_capture_df.rename(columns={
        'cur_ff_index': 'nxt_ff_index',
        'stop_point_index': 'next_stop_point_index',
        'stop_time': 'next_stop_time'
    })[['nxt_ff_index', 'next_stop_point_index', 'next_stop_time']]
    df = df.merge(next_stop_df, on='nxt_ff_index', how='left')

    # Filter based on minimum time between stops
    df['cur_ff_capture_time'] = ff_caught_T_new[df['cur_ff_index'].values]
    before_time_filter = len(df)
    df = df[
        df['next_stop_time'] -
        df['stop_time'] >= min_time_between_cur_and_nxt_ff_caught_time
    ].copy()
    print(f"{before_time_filter - len(df)} of {before_time_filter} rows removed: time between stops < {min_time_between_cur_and_nxt_ff_caught_time}s")

    # Add positions and distances
    df[['ff_x', 'ff_y']] = ff_real_position_sorted[df['nxt_ff_index'].values]
    df['dist_to_next_stop'] = np.linalg.norm(
        df[['ff_x', 'ff_y']].values -
        monkey_information.loc[df['next_stop_point_index'], [
            'monkey_x', 'monkey_y']].values,
        axis=1
    )
    df['dist_cur_to_nxt'] = np.linalg.norm(
        ff_real_position_sorted[df['cur_ff_index'].values] -
        df[['ff_x', 'ff_y']].values,
        axis=1
    )

    # Filter based on distances
    too_close = df['dist_cur_to_nxt'] <= min_distance_between_cur_and_nxt_ff
    too_far = df['dist_cur_to_nxt'] >= max_distance_between_cur_and_nxt_ff
    df.loc[too_close | too_far, 'data_category_by_vis'] = 'neither'
    print(f"{too_close.sum()} rows removed: distance < {min_distance_between_cur_and_nxt_ff}cm")
    print(f"{too_far.sum()} rows removed: distance > {max_distance_between_cur_and_nxt_ff}cm")

    # Add visibility info
    df = add_nxt_ff_first_and_last_seen_info(
        df, ff_dataframe_visible, monkey_information,
        ff_real_position_sorted, ff_life_sorted
    )
    df['nxt_ff_last_seen_rel_time_bbas'] = (
        df['NXT_time_ff_last_seen_bbas'] - df['beginning_time']
    )

    # Assign unseen as control
    unseen = (df['nxt_ff_last_seen_rel_time_bbas'].isnull()) & (
        df['data_category_by_vis'] != 'neither'
    )
    df.loc[unseen, 'data_category_by_vis'] = 'control'

    # Final output
    keep_cols = [
        'nxt_ff_index', 'cur_ff_index', 'stop_point_index', 'data_category_by_vis',
        'nxt_ff_caught_time', 'next_stop_point_index', 'next_stop_time',
        'dist_to_next_stop', 'dist_cur_to_nxt',
        'nxt_ff_cluster_last_seen_rel_time_bbas',
        'NXT_point_index_ff_first_seen_bbas', 'NXT_point_index_ff_last_seen_bbas',
        'NXT_monkey_angle_ff_first_seen_bbas', 'NXT_monkey_angle_ff_last_seen_bbas',
        'NXT_time_ff_first_seen_bbas', 'NXT_time_ff_last_seen_bbas',
        'NXT_time_ff_first_seen_bsans', 'NXT_time_ff_last_seen_bsans',
        'nxt_ff_cluster_last_seen_time_bbas', 'nxt_ff_cluster_last_seen_time_bsans',
        'nxt_ff_last_seen_rel_time_bbas'
    ]
    df = df[keep_cols].copy()
    df['nxt_ff_index'] = df['nxt_ff_index'].astype(int)

    return df


def add_nxt_ff_first_and_last_seen_info(all_nxt_ff_df, ff_dataframe_visible, monkey_information, ff_real_position_sorted, ff_life_sorted):
    all_nxt_ff_df = all_nxt_ff_df.copy()
    all_nxt_ff_df = add_nxt_ff_first_seen_and_last_seen_info_bbas(
        all_nxt_ff_df, ff_dataframe_visible, monkey_information)
    all_nxt_ff_df = add_nxt_ff_first_seen_and_last_seen_info_bsans(
        all_nxt_ff_df, ff_dataframe_visible, monkey_information)

    all_nxt_ff_df = get_nxt_ff_cluster_last_seen_bbas(
        all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted)
    all_nxt_ff_df = get_nxt_ff_cluster_last_seen_bsans(
        all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted)
    return all_nxt_ff_df





def drop_rows_where_stop_is_not_inside_reward_boundary(closest_stop_to_capture_df):
    original_length = len(closest_stop_to_capture_df)
    outlier_sub_df = closest_stop_to_capture_df[closest_stop_to_capture_df['distance_from_ff_to_stop'] > 25].sort_values(
        by='distance_from_ff_to_stop', ascending=False)
    closest_stop_to_capture_df = closest_stop_to_capture_df[
        closest_stop_to_capture_df['distance_from_ff_to_stop'] <= 25].copy()

    print(f'{original_length - len(closest_stop_to_capture_df)} rows out of {original_length} rows were removed from closest_stop_to_capture_df because the distance between stop and ff center is larger than 25cm, '\
          #   + f'\n which is {round((original_length - len(closest_stop_to_capture_df))/original_length*100, 2)}% of the rows, '\
          #   + f'and the sorted distances from those are {outlier_sub_df["distance_from_ff_to_stop"].values}'
          )
    return closest_stop_to_capture_df


def get_all_captured_ff_first_seen_and_last_seen_info(closest_stop_to_capture_df, stop_period_duration, ff_dataframe_visible, monkey_information, drop_na=False):
    if 'cur_ff_index' in closest_stop_to_capture_df.columns:
        all_ff_index = closest_stop_to_capture_df['cur_ff_index'].values
    else:
        all_ff_index = closest_stop_to_capture_df['ff_index'].values

    if 'stop_point_index' in closest_stop_to_capture_df.columns:
        all_stop_point_index = closest_stop_to_capture_df['stop_point_index'].values
    else:
        all_stop_point_index = closest_stop_to_capture_df['point_index'].values

    if 'stop_time' in closest_stop_to_capture_df.columns:
        all_end_time = closest_stop_to_capture_df['stop_time'].values
    else:
        all_end_time = closest_stop_to_capture_df['time'].values

    all_start_time = all_end_time - stop_period_duration

    ff_info = get_ff_first_and_last_seen_info(all_ff_index, all_stop_point_index, all_start_time,
                                              all_end_time, ff_dataframe_visible, monkey_information, drop_na=drop_na)

    return ff_info


def rename_first_and_last_seen_info_columns(df, prefix='CUR_'):
    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                      'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen',
                      'time_ff_first_seen', 'time_ff_last_seen']
    columns_to_be_renamed = {column: prefix +
                             column + '_bbas' for column in columns_to_add}
    # Note: bbas means "between start and stop"
    df.rename(columns=columns_to_be_renamed, inplace=True)
    return df


def add_nxt_ff_first_seen_and_last_seen_info_bbas(all_nxt_ff_df, ff_dataframe_visible, monkey_information):
    all_nxt_ff_df = _add_stop_or_nxt_ff_first_seen_and_last_seen_info_bbas(
        all_nxt_ff_df, ff_dataframe_visible, monkey_information, cur_or_nxt='nxt')
    return all_nxt_ff_df


def _add_stop_or_nxt_ff_first_seen_and_last_seen_info_bbas(df, ff_dataframe_visible, monkey_information,
                                                           cur_or_nxt='nxt'):
    all_stop_time = df['stop_time'].values
    all_start_time = df['beginning_time'].values
    ff_index_column = cur_or_nxt + '_ff_index'
    nxt_ff_first_and_last_seen_info = get_ff_first_and_last_seen_info(df[ff_index_column].values, df['stop_point_index'].values, all_start_time,
                                                                      all_stop_time, ff_dataframe_visible, monkey_information)

    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                      'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen',
                      'time_ff_first_seen', 'time_ff_last_seen']
    nxt_ff_first_and_last_seen_info = nxt_ff_first_and_last_seen_info[columns_to_add + [
        'stop_point_index']]
    prefix = 'NXT_' if cur_or_nxt == 'nxt' else 'CUR_'
    columns_to_be_renamed_dict = {column: prefix +
                                  column + '_bbas' for column in columns_to_add}
    nxt_ff_first_and_last_seen_info.rename(
        columns=columns_to_be_renamed_dict, inplace=True)
    df = df.merge(nxt_ff_first_and_last_seen_info,
                  on='stop_point_index', how='left')
    return df


def add_nxt_ff_first_seen_and_last_seen_info_bsans(all_nxt_ff_df, ff_dataframe_visible, monkey_information):

    nxt_ff_first_and_last_seen_info = get_ff_first_and_last_seen_info(all_nxt_ff_df['nxt_ff_index'].values, all_nxt_ff_df['stop_point_index'].values, all_nxt_ff_df['stop_time'].values,
                                                                      all_nxt_ff_df['next_stop_time'].values, ff_dataframe_visible, monkey_information)

    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                      'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen',
                      'time_ff_first_seen', 'time_ff_last_seen']
    nxt_ff_first_and_last_seen_info = nxt_ff_first_and_last_seen_info[columns_to_add + [
        'stop_point_index']]
    columns_to_be_renamed_dict = {column: 'NXT_' +
                                  column + '_bsans' for column in columns_to_add}
    nxt_ff_first_and_last_seen_info.rename(
        columns=columns_to_be_renamed_dict, inplace=True)
    all_nxt_ff_df = all_nxt_ff_df.merge(
        nxt_ff_first_and_last_seen_info, on='stop_point_index', how='left')
    return all_nxt_ff_df


def get_nxt_ff_last_seen_rel_time(all_nxt_ff_df, ff_dataframe, stop_period_duration=2):
    # See if nxt ff was visible before stop; if they are not, they will be assigned control rather than test
    all_nxt_ff_df['beginning_time'] = all_nxt_ff_df['stop_time'].values - \
        stop_period_duration
    nxt_ff_last_seen_info = find_first_or_last_ff_sighting_in_stop_period(all_nxt_ff_df, 'nxt_ff_index', ff_dataframe,
                                                                          first_or_last='last')
    nxt_ff_last_seen_info.rename(
        columns={'ff_index': 'nxt_ff_index', 'time': 'nxt_ff_last_seen_time'}, inplace=True)

    all_nxt_ff_df = all_nxt_ff_df.merge(nxt_ff_last_seen_info[[
                                        'nxt_ff_index', 'nxt_ff_last_seen_time']], on='nxt_ff_index', how='left')
    all_nxt_ff_df['nxt_ff_last_seen_rel_time_bbas'] = all_nxt_ff_df['nxt_ff_last_seen_time'] - \
        all_nxt_ff_df['beginning_time']
    return all_nxt_ff_df


def find_first_or_last_ff_sighting_in_stop_period(ff_df, ff_index_column, ff_dataframe,
                                                  first_or_last='last'):
    all_ff_index = ff_df[ff_index_column].values
    all_point_index = ff_df['stop_point_index'].values
    all_start_time = ff_df['beginning_time'].values
    all_end_time = ff_df['stop_time'].values
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    ff_sighting_info = find_first_or_last_ff_sighting(all_ff_index, all_point_index, all_start_time,
                                                      all_end_time, ff_dataframe_visible, first_or_last=first_or_last)
    return ff_sighting_info


def get_nxt_ff_cluster_last_seen_info(all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                      start_time_column='beginning_time', end_time_column='stop_time'):
    # See if nxt ff was visible before stop; if they are not, they will be assigned control rather than test

    if 'nxt_ff_cluster' not in all_nxt_ff_df.columns:
        all_nxt_ff_df[['nxt_ff_x', 'nxt_ff_y']
                      ] = ff_real_position_sorted[all_nxt_ff_df['nxt_ff_index'].values]
        all_nxt_ff_df['nxt_ff_cluster'] = cluster_analysis.find_alive_ff_clusters(all_nxt_ff_df[['nxt_ff_x', 'nxt_ff_y']].values,
                                                                                  ff_real_position_sorted, all_nxt_ff_df[
                                                                                      'beginning_time'].values,
                                                                                  all_nxt_ff_df['next_stop_time'].values,
                                                                                  ff_life_sorted, max_distance=50)
    all_nxt_ff_df['nxt_ff_cluster_size'] = all_nxt_ff_df['nxt_ff_cluster'].apply(
        len)

    all_ff_index = []
    [all_ff_index.extend(array)
     for array in all_nxt_ff_df['nxt_ff_cluster'].tolist()]
    all_ff_index = np.array(all_ff_index)

    all_point_index = np.repeat(
        all_nxt_ff_df['stop_point_index'].values, all_nxt_ff_df['nxt_ff_cluster_size'].values)
    all_end_time = np.repeat(
        all_nxt_ff_df[end_time_column].values, all_nxt_ff_df['nxt_ff_cluster_size'].values)
    all_start_time = np.repeat(
        all_nxt_ff_df[start_time_column].values, all_nxt_ff_df['nxt_ff_cluster_size'].values)
    nxt_ff_last_seen_info = find_first_or_last_ff_sighting(all_ff_index, all_point_index, all_start_time,
                                                           all_end_time, ff_dataframe_visible, first_or_last='last')
    # for each stop_point_index, find the latest last-seen time
    # before sorting, we need to drop rows with NA
    nxt_ff_last_seen_info.dropna(axis=0, inplace=True)
    nxt_ff_last_seen_info.sort_values(by=['stop_point_index', 'time'], ascending=[
                                      True, True], inplace=True)
    nxt_ff_last_seen_info = nxt_ff_last_seen_info.groupby(
        'stop_point_index').last().reset_index(drop=False)
    nxt_ff_last_seen_info = nxt_ff_last_seen_info[[
        'stop_point_index', 'ff_index', 'time']].copy()

    return nxt_ff_last_seen_info


def get_nxt_ff_cluster_last_seen_bbas(all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted):
    # See if nxt ff was visible before stop
    nxt_ff_last_seen_info = get_nxt_ff_cluster_last_seen_info(all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                                              start_time_column='beginning_time', end_time_column='stop_time')

    nxt_ff_last_seen_info.rename(columns={'ff_index': 'ff_index_last_seen_bbas_in_nxt_ff_cluster',
                                          'time': 'nxt_ff_cluster_last_seen_time_bbas'}, inplace=True)
    all_nxt_ff_df = all_nxt_ff_df.merge(
        nxt_ff_last_seen_info, on='stop_point_index', how='left').reset_index(drop=True)

    all_nxt_ff_df['nxt_ff_cluster_last_seen_rel_time_bbas'] = all_nxt_ff_df['nxt_ff_cluster_last_seen_time_bbas'] - \
        all_nxt_ff_df['beginning_time']

    return all_nxt_ff_df


def get_nxt_ff_cluster_last_seen_bsans(all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted):
    # See if nxt ff was visible before stop
    nxt_ff_last_seen_info = get_nxt_ff_cluster_last_seen_info(all_nxt_ff_df, ff_dataframe_visible, ff_real_position_sorted, ff_life_sorted,
                                                              start_time_column='stop_time', end_time_column='next_stop_time')

    nxt_ff_last_seen_info.rename(columns={'ff_index': 'ff_index_last_seen_bsans_in_nxt_ff_cluster',
                                          'time': 'nxt_ff_cluster_last_seen_time_bsans'}, inplace=True)
    all_nxt_ff_df = all_nxt_ff_df.merge(
        nxt_ff_last_seen_info, on='stop_point_index', how='left').reset_index(drop=True)

    all_nxt_ff_df['nxt_ff_cluster_last_seen_rel_time_bsans'] = all_nxt_ff_df['nxt_ff_cluster_last_seen_time_bsans'] - \
        all_nxt_ff_df['beginning_time']

    return all_nxt_ff_df


def _get_nxt_ff_df_or_cur_ff_df(shared_stops_near_ff_df, cur_or_nxt='nxt'):
    shared_columns = ['stop_point_index', 'stop_time', 'stop_cum_distance']
    ff_column = [cur_or_nxt + '_ff_index']
    columns_to_add = ['point_index_ff_first_seen', 'point_index_ff_last_seen',
                      'monkey_angle_ff_first_seen', 'monkey_angle_ff_last_seen',
                      'time_ff_first_seen', 'time_ff_last_seen']
    columns_to_add = [cur_or_nxt.upper() + '_' + column +
                      '_bbas' for column in columns_to_add]
    all_relevant_columns = shared_columns + ff_column + columns_to_add
    ff_df = shared_stops_near_ff_df[all_relevant_columns].copy()
    prefix_len = len(cur_or_nxt + '_')
    columns_to_be_renamed = {
        column: column[prefix_len:-5] for column in columns_to_add}
    columns_to_be_renamed[ff_column[0]] = 'ff_index'
    ff_df.rename(columns=columns_to_be_renamed, inplace=True)
    ff_df.reset_index(drop=True, inplace=True)
    return ff_df


def get_nxt_ff_df_and_cur_ff_df(stops_near_ff_df):

    nxt_ff_df = _get_nxt_ff_df_or_cur_ff_df(
        stops_near_ff_df, cur_or_nxt='nxt')
    cur_ff_df = _get_nxt_ff_df_or_cur_ff_df(
        stops_near_ff_df, cur_or_nxt='cur')

    stops_near_ff_df['earlest_point_index_when_nxt_ff_and_cur_ff_have_both_been_seen_bbas'] = np.stack(
        [nxt_ff_df['point_index_ff_first_seen'].values, cur_ff_df['point_index_ff_first_seen'].values]).max(axis=0)
    return stops_near_ff_df, nxt_ff_df, cur_ff_df


def get_info_for_ff_based_on_stop_period_time_window(stops_near_ff_df, all_ff_index, ff_dataframe_visible, monkey_information, stop_period_duration=2):
    all_stop_point_index, all_stop_time = stops_near_ff_df[
        'stop_point_index'].values, stops_near_ff_df['stop_time'].values
    all_start_time = all_stop_time - stop_period_duration

    ff_info = get_ff_first_and_last_seen_info(all_ff_index, all_stop_point_index, all_start_time,
                                              all_stop_time, ff_dataframe_visible, monkey_information)

    # ff_info['duration_of_ff_unseen'] = ff_info['stop_time'] - ff_info['time_ff_last_seen']
    return ff_info


def get_ff_first_and_last_seen_info(
    all_ff_index, all_stop_point_index, all_start_time, all_end_time,
    ff_dataframe_visible, monkey_information, verbose=True, drop_na=False
):
    # Get first and last seen info
    ff_last_seen_info = find_first_or_last_ff_sighting(
        all_ff_index, all_stop_point_index, all_start_time, all_end_time,
        ff_dataframe_visible, first_or_last='last'
    )
    ff_first_seen_info = find_first_or_last_ff_sighting(
        all_ff_index, all_stop_point_index, all_start_time, all_end_time,
        ff_dataframe_visible, first_or_last='first'
    )

    # Build the ff_info DataFrame
    ff_info = pd.DataFrame({
        'ff_index': all_ff_index,
        'stop_point_index': all_stop_point_index
    })

    for attr in ['point_index', 'monkey_angle', 'ff_distance', 'ff_angle', 'ff_angle_boundary', 'time']:
        ff_info[f'{attr}_ff_first_seen'] = ff_first_seen_info[attr].values
        ff_info[f'{attr}_ff_last_seen'] = ff_last_seen_info[attr].values

    # Handle missing data
    if drop_na and ff_info.isnull().values.any():
        num_null_rows = ff_info.isnull().any(axis=1).sum()
        if num_null_rows > 0:
            print(
                f'Warning: {num_null_rows} of {len(ff_info)} rows in ff_info have null values '
                'because they were not visible in the stop period. They will be dropped.'
            )
            ff_info.dropna(inplace=True)

    # Convert non-null indices to int
    for col in ['point_index_ff_first_seen', 'point_index_ff_last_seen']:
        non_null_mask = ~ff_info[col].isnull()
        ff_info.loc[non_null_mask,
                    col] = ff_info.loc[non_null_mask, col].astype(int)

    # Add corresponding time info
    if 'point_index_ff_first_seen' in ff_info.columns:
        mask = ~ff_info['point_index_ff_first_seen'].isnull()
        ff_info.loc[mask, 'time_ff_first_seen'] = monkey_information.loc[
            ff_info.loc[mask, 'point_index_ff_first_seen'].values, 'time'
        ].values

    if 'point_index_ff_last_seen' in ff_info.columns:
        mask = ~ff_info['point_index_ff_last_seen'].isnull()
        ff_info.loc[mask, 'time_ff_last_seen'] = monkey_information.loc[
            ff_info.loc[mask, 'point_index_ff_last_seen'].values, 'time'
        ].values

    return ff_info


def find_first_or_last_ff_sighting(all_ff_index, all_point_index, all_start_time, all_end_time, ff_dataframe_visible, first_or_last='first'):

    temp_ff_df = pd.DataFrame({'ff_index': all_ff_index,
                               'beginning_time': all_start_time,
                               'end_time': all_end_time,
                               'stop_point_index': all_point_index})
    ff_info = pd.merge(temp_ff_df, ff_dataframe_visible,
                       on='ff_index', how='inner')

    # Filter visibility within window
    ff_info = ff_info[ff_info['time'].between(
        ff_info['beginning_time'], ff_info['end_time'], inclusive='left')].copy()

    ff_info = ff_info.sort_values(by=['stop_point_index', 'ff_index', 'time'])
    # Select first or last occurrence
    if first_or_last == 'first':
        ff_info = ff_info.groupby(['stop_point_index', 'ff_index']).head(
            1).reset_index(drop=True)
    else:
        ff_info = ff_info.groupby(['stop_point_index', 'ff_index']).tail(
            1).reset_index(drop=True)

    # Ensure all input (ff_index, stop_point_index) pairs are preserved
    ff_info = pd.merge(temp_ff_df[['ff_index', 'stop_point_index']], ff_info, on=[
                       'ff_index', 'stop_point_index'], how='left')

    # Optional: Warn about missing entries
    # if ff_info.isnull().values.any():
    #     num_null_rows = len(ff_info[ff_info.isnull().any(axis=1)])
    #     if len(ff_info[ff_info.isnull().any(axis=1)]) > 0:
    #         # show a warning about # rows in df that has null values
    #         print(f'Warning: There are {num_null_rows} rows out of {len(ff_info)} rows in ff_info that have null values.')
    #         ff_info.dropna(axis=0, inplace=True)
    return ff_info


def find_within_n_cm_to_point_info(point_x, point_y, ff_info_in_duration, n_cm=300):
    within_n_cm_to_point_info = ff_info_in_duration.copy()
    within_n_cm_to_point_info['distance_to_point'] = np.linalg.norm(
        [within_n_cm_to_point_info['ff_x'] - point_x, within_n_cm_to_point_info['ff_y'] - point_y], axis=0)
    within_n_cm_to_point_info = ff_info_in_duration[within_n_cm_to_point_info['distance_to_point'] <= n_cm].copy(
    )
    return within_n_cm_to_point_info


def find_if_nxt_ff_cluster_visible_pre_stop(stops_near_ff_df_ctrl, ff_dataframe, ff_real_position_sorted,
                                            max_distance_between_ffs_in_cluster=50, duration_prior_to_stop_time=3):
    ff_dataframe_visible = ff_dataframe[ff_dataframe['visible'] == 1].copy()

    if_nxt_ff_cluster_visible_pre_stop = []
    for index, row in stops_near_ff_df_ctrl.iterrows():
        visible_ff_info_in_duration = ff_dataframe_visible[ff_dataframe_visible['time'].between(
            row['stop_time']-duration_prior_to_stop_time, row['stop_time'])].copy()
        nxt_ff_xy = ff_real_position_sorted[row['nxt_ff_index']]
        nxt_ff_cluster_info = find_within_n_cm_to_point_info(
            nxt_ff_xy[0], nxt_ff_xy[1], visible_ff_info_in_duration, n_cm=max_distance_between_ffs_in_cluster)
        # nxt_ff_cluster_info = nxt_ff_cluster_info[nxt_ff_cluster_info['ff_index'] != row['nxt_ff_index']]
        if len(nxt_ff_cluster_info) > 0:
            if_nxt_ff_cluster_visible_pre_stop.append(True)
        else:
            if_nxt_ff_cluster_visible_pre_stop.append(False)
    if_nxt_ff_cluster_visible_pre_stop = np.array(
        if_nxt_ff_cluster_visible_pre_stop)
    stops_near_ff_df_ctrl['if_nxt_ff_cluster_visible_pre_stop'] = if_nxt_ff_cluster_visible_pre_stop
    return stops_near_ff_df_ctrl


def add_if_nxt_ff_and_nxt_ff_cluster_flash_bbas(df, ff_real_position_sorted, ff_flash_sorted, ff_life_sorted, stop_period_duration=2):

    df['beginning_time'] = df['stop_time'] - stop_period_duration

    flash_on_columns = _get_flash_on_columns(df, ff_flash_sorted,
                                             ff_real_position_sorted,
                                             ff_life_sorted,
                                             duration_start_column='beginning_time',
                                             duration_end_column='stop_time',
                                             )

    flash_on_columns_df = pd.DataFrame(flash_on_columns, index=df.index)
    flash_on_columns_df['nxt_ff_last_flash_time_bbas'] = flash_on_columns_df[
        'nxt_ff_last_flash_time_bbas'].replace(-999, np.nan)
    flash_on_columns_df['nxt_ff_cluster_last_flash_time_bbas'] = flash_on_columns_df[
        'nxt_ff_cluster_last_flash_time_bbas'].replace(-999, np.nan)
    df = pd.concat([df, flash_on_columns_df], axis=1)

    num_if_nxt_ff_cluster_flash_bbas = sum(
        flash_on_columns['if_nxt_ff_cluster_flash_bbas'])
    print(f'Percentage of control rows that have nxt ff cluster flashed on between stop_time - {stop_period_duration} and stop: \
            {round(num_if_nxt_ff_cluster_flash_bbas/len(df)*100, 2)} % of {len(df)} rows.')

    return df


def add_if_nxt_ff_and_nxt_ff_cluster_flash_bsans(df, ff_real_position_sorted, ff_flash_sorted, ff_life_sorted):
    flash_on_columns = _get_flash_on_columns(df, ff_flash_sorted,
                                             ff_real_position_sorted,
                                             ff_life_sorted,
                                             duration_start_column='stop_time',
                                             duration_end_column='next_stop_time',
                                             )

    # for all column names in flash_on_columns, repalce basa with bsans
    flash_on_columns = {key.replace(
        'bbas', 'bsans'): value for key, value in flash_on_columns.items()}

    flash_on_columns_df = pd.DataFrame(flash_on_columns, index=df.index)
    flash_on_columns_df['nxt_ff_last_flash_time_bsans'] = flash_on_columns_df[
        'nxt_ff_last_flash_time_bsans'].replace(-999, np.nan)
    flash_on_columns_df['nxt_ff_cluster_last_flash_time_bsans'] = flash_on_columns_df[
        'nxt_ff_cluster_last_flash_time_bsans'].replace(-999, np.nan)
    df = pd.concat([df, flash_on_columns_df], axis=1)

    num_if_nxt_ff_cluster_flash_bsans = sum(
        flash_on_columns['if_nxt_ff_cluster_flash_bsans'])
    print(f'Percentage of control rows that have nxt ff cluster flashed on between stop time and next stop: \
            {round(num_if_nxt_ff_cluster_flash_bsans/len(df)*100, 2)} % of {len(df)} rows.')

    return df


def _get_flash_on_columns(df, ff_flash_sorted, ff_real_position_sorted, ff_life_sorted,
                          duration_start_column='beginning_time',
                          duration_end_column='stop_time',
                          ):
    if 'nxt_ff_x' not in df.columns:
        df['nxt_ff_x'], df['nxt_ff_y'] = ff_real_position_sorted[df['nxt_ff_index'].values].T
    if 'nxt_ff_cluster' not in df.columns:
        df['nxt_ff_cluster'] = cluster_analysis.find_alive_ff_clusters(df[['nxt_ff_x', 'nxt_ff_y']].values,
                                                                       ff_real_position_sorted,
                                                                       df['beginning_time'].values,
                                                                       df['next_stop_time'].values,
                                                                       ff_life_sorted, max_distance=50)

    flash_on_columns = {'if_nxt_ff_flash_bbas': [],
                        'if_nxt_ff_cluster_flash_bbas': [],
                        'nxt_ff_last_flash_time_bbas': [],
                        'nxt_ff_cluster_last_flash_time_bbas': []

                        }
    for index, row in df.iterrows():
        ff_cluster = row['nxt_ff_cluster']
        if_nxt_ff_flash_bbas = False
        if_nxt_ff_cluster_flash_bbas = False
        nxt_ff_last_flash_time_bbas = -999
        nxt_ff_cluster_last_flash_time_bbas = -999
        for ff_index in ff_cluster:
            ff_flash = ff_flash_sorted[ff_index]
            result = general_utils.find_intersection(
                ff_flash, [row[duration_start_column], row[duration_end_column]])
            if len(result) > 0:
                if_nxt_ff_cluster_flash_bbas = True
                latest_flash_time_before_stop = min(
                    ff_flash[result[-1]][-1], row[duration_end_column])
                nxt_ff_cluster_last_flash_time_bbas = max(
                    nxt_ff_cluster_last_flash_time_bbas, latest_flash_time_before_stop)
                if ff_index == row['nxt_ff_index']:
                    if_nxt_ff_flash_bbas = True
                    nxt_ff_last_flash_time_bbas = max(
                        nxt_ff_last_flash_time_bbas, latest_flash_time_before_stop)
        flash_on_columns['if_nxt_ff_flash_bbas'].append(if_nxt_ff_flash_bbas)
        flash_on_columns['if_nxt_ff_cluster_flash_bbas'].append(
            if_nxt_ff_cluster_flash_bbas)
        flash_on_columns['nxt_ff_last_flash_time_bbas'].append(
            nxt_ff_last_flash_time_bbas)
        flash_on_columns['nxt_ff_cluster_last_flash_time_bbas'].append(
            nxt_ff_cluster_last_flash_time_bbas)
    return flash_on_columns


