from data_wrangling import specific_utils
from pattern_discovery import ff_dataframe_utils
from visualization.matplotlib_tools import plot_behaviors_utils

import os
import numpy as np
import pandas as pd
from math import pi
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def make_ff_dataframe_func(monkey_information, ff_caught_T_new, ff_flash_sorted,
                           ff_real_position_sorted, ff_life_sorted, player="monkey",
                           max_distance=500, ff_radius=10, reward_boundary_radius=25,
                           obs_ff_indices_in_ff_dataframe=None,
                           max_time_since_last_vis=3,
                           ff_in_obs_df=None,
                           to_add_essential_columns=True,
                           to_furnish_ff_dataframe=True,
                           truncate_info_beyond_capture=True,
                           print_progress=True, ):
    """
    Make a dataframe called ff_dataframe that contains various information about all visible or "in-memory" fireflies at each time point


    Parameters
    ----------
    monkey_information: df
        containing the speed, angle, and location of the monkey at various points of time
    ff_caught_T_new: np.array
        containing the time when each captured firefly gets captured
    ff_flash_sorted: list
        containing the time that each firefly flashes on and off
    ff_real_position_sorted: np.array
        containing the real locations of the fireflies
    ff_life_sorted: np.array
        containing the time that each firefly comes into being and gets captured 
        (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    player: str
        "monkey" or "agent" 
    max_distance: num
        the distance beyond which the firefly cannot be considered visible
    ff_radius: num
        the radius of a firefly; the current setting of the game sets it to be 10
    data_folder_name: str, default is None
        the place to store the output as a csv
    print_progress: bool
        whether to print the progress of making ff_dataframe
    obs_ff_indices_in_ff_dataframe: list
        a variable to be passed if the player is "agent"; it contains the correct indices of fireflies 
    ff_in_obs_df: pd.dataframe
        containing the noisy x, y coordinates of fireflies at different points in time, collected from env

    Returns
    -------
    ff_dataframe: pd.dataframe
        containing various information about all visible or "in-memory" fireflies at each time point


    """

    dt = (monkey_information['time'].iloc[-1] -
          monkey_information['time'].iloc[0])/(len(monkey_information)-1)
    if max_time_since_last_vis is None:
        max_memory = 200
    else:
        # we use 1.2 here to make sure that the time range covers the max_time_since_last_vis
        max_memory = int((max_time_since_last_vis/dt)*1.2)
    # max_memory: the numeric value of the variable "memory" for a firefly when it's fully visible

    if player == 'monkey':
        if len(ff_caught_T_new) < 1:
            return ff_dataframe_utils.make_empty_ff_dataframe()
    else:
        if len(ff_caught_T_new) == 0:
            ff_caught_T_new = [monkey_information['time'].max() + 10]

    ff_index = []
    point_index = []
    visible = []
    time_since_last_vis = []
    total_ff_num = len(ff_life_sorted)

    starting_ff = {"monkey": 1, "agent": 0}
    for i in range(starting_ff[player], total_ff_num):
        current_ff_index = i

        visible_indices, in_memory_indices, memory_array, time_since_last_vis_array = ff_dataframe_utils.find_visible_indices_AND_memory_array_AND_time_since_last_vis(
            current_ff_index,
            monkey_information,
            obs_ff_indices_in_ff_dataframe,
            ff_flash_sorted,
            ff_real_position_sorted,
            ff_caught_T_new,
            max_distance,
            player,
            max_memory,
            truncate_info_beyond_capture=truncate_info_beyond_capture,
        )
        # memory_array/in_memory_indices define relevance; skip if empty
        if len(in_memory_indices) > 0:
            num_points_in_memory = len(in_memory_indices)

            # Append the values for this ff efficiently
            ff_index.extend([current_ff_index] * num_points_in_memory)
            point_index.extend(in_memory_indices.tolist())
            m_arr = np.asarray(memory_array)
            visible.extend((m_arr == max_memory).astype(np.int8).tolist())
            time_since_last_vis.extend(np.asarray(
                time_since_last_vis_array).tolist())

        if i % 200 == 0:
            if print_progress:
                print("Making ff_dataframe: ", i, " out of ",
                      total_ff_num, " total number of fireflies ")

    ff_dict = {'ff_index': ff_index, 'point_index': point_index,
               'visible': visible, 'time_since_last_vis': time_since_last_vis}

    ff_dataframe = pd.DataFrame(ff_dict, copy=False)

    if len(ff_dataframe) == 0:
        return ff_dataframe_utils.make_empty_ff_dataframe()

    # Ensure point_index bounds without costly isin
    max_idx = len(monkey_information)
    mask_pi = (ff_dataframe['point_index'] >= 0) & (
        ff_dataframe['point_index'] < max_idx)
    if not mask_pi.all():
        ff_dataframe = ff_dataframe.loc[mask_pi].reset_index(drop=True)

    # Map time via direct numpy indexing
    time_array = monkey_information['time'].to_numpy()
    ff_dataframe['time'] = time_array[ff_dataframe['point_index'].to_numpy()]
    ff_dataframe['target_index'] = np.searchsorted(
        ff_caught_T_new, ff_dataframe['time'])
    ff_dataframe = ff_dataframe[ff_dataframe['target_index'] < len(
        ff_caught_T_new)].copy()

    if ff_in_obs_df is not None:
        ff_dataframe[['ff_x', 'ff_y']
                     ] = ff_real_position_sorted[ff_dataframe['ff_index'].values]
        ff_dataframe = pd.merge(
            ff_dataframe, ff_in_obs_df, how="left", on=["ff_index", "point_index"], sort=False, copy=False
        )
        ff_dataframe.loc[ff_dataframe['ff_x_noisy'].isnull(
        ), 'ff_x_noisy'] = ff_dataframe.loc[ff_dataframe['ff_x_noisy'].isnull(), 'ff_x']
        ff_dataframe.loc[ff_dataframe['ff_y_noisy'].isnull(
        ), 'ff_y_noisy'] = ff_dataframe.loc[ff_dataframe['ff_y_noisy'].isnull(), 'ff_y']

    if to_add_essential_columns:
        add_essential_columns_to_ff_dataframe(
            ff_dataframe, monkey_information, ff_real_position_sorted, ff_radius, reward_boundary_radius)

    if to_furnish_ff_dataframe:
        ff_dataframe = furnish_ff_dataframe(
            ff_dataframe, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted)

    return ff_dataframe


def add_essential_columns_to_ff_dataframe(ff_dataframe, monkey_information, ff_real_position_sorted, ff_radius=10, reward_boundary_radius=25):
    ff_dataframe[['monkey_x', 'monkey_y', 'monkey_angle', 'monkey_angle', 'ang_speed', 'dt', 'cum_distance', 'time', 'monkey_speeddummy']]  \
        = monkey_information.loc[ff_dataframe['point_index'].values, ['monkey_x', 'monkey_y', 'monkey_angle', 'monkey_angle', 'ang_speed', 'dt', 'cum_distance', 'time', 'monkey_speeddummy']].values
    ff_dataframe[['ff_x', 'ff_y']
                 ] = ff_real_position_sorted[ff_dataframe['ff_index'].values]
    ff_dataframe['ff_distance'] = np.linalg.norm(np.array(
        ff_dataframe[['monkey_x', 'monkey_y']])-np.array(ff_dataframe[['ff_x', 'ff_y']]), axis=1)
    ff_dataframe['ff_angle'] = specific_utils.calculate_angles_to_ff_centers(ff_x=ff_dataframe['ff_x'], ff_y=ff_dataframe['ff_y'], mx=ff_dataframe['monkey_x'],
                                                                             my=ff_dataframe['monkey_y'], m_angle=ff_dataframe['monkey_angle'])
    ff_dataframe['ff_angle_boundary'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_dataframe['ff_angle'], distances_to_ff=ff_dataframe['ff_distance'], ff_radius=ff_radius)
    ff_dataframe['abs_ff_angle'] = np.abs(ff_dataframe['ff_angle'])
    ff_dataframe['abs_ff_angle_boundary'] = np.abs(
        ff_dataframe['ff_angle_boundary'])
    ff_dataframe['left_right'] = (
        np.array(ff_dataframe['ff_angle']) > 0).astype(int)
    ff_dataframe['angles_to_reward_boundaries'] = specific_utils.calculate_angles_to_ff_boundaries(
        angles_to_ff=ff_dataframe.ff_angle, distances_to_ff=ff_dataframe.ff_distance, ff_radius=reward_boundary_radius)


def process_ff_dataframe(ff_dataframe, max_distance, max_time_since_last_vis):
    # set ff_index, point_index, target_index, visible, left_right all to be int
    ff_dataframe[['ff_index', 'point_index', 'visible']] = ff_dataframe[[
        'ff_index', 'point_index', 'visible']].astype('int')
    if max_distance is not None:
        ff_dataframe = ff_dataframe[ff_dataframe['ff_distance']
                                    < max_distance + 100]
    if max_time_since_last_vis is not None:
        ff_dataframe = ff_dataframe[ff_dataframe['time_since_last_vis']
                                    <= max_time_since_last_vis]
    return ff_dataframe


def furnish_ff_dataframe(ff_dataframe, ff_real_position_sorted, ff_caught_T_new, ff_life_sorted):

    ff_dataframe['abs_delta_ff_angle'], ff_dataframe['abs_delta_ff_angle_boundary'] = specific_utils.calculate_change_in_abs_ff_angle(current_ff_index=ff_dataframe['ff_index'].values, angles_to_ff=ff_dataframe['ff_angle'].values,
                                                                                                                                      angles_to_boundaries=ff_dataframe['ff_angle_boundary'].values, ff_real_position_sorted=ff_real_position_sorted, monkey_x_array=ff_dataframe[
                                                                                                                                          'monkey_x'].values, monkey_y_array=ff_dataframe['monkey_y'].values,
                                                                                                                                      monkey_angles_array=ff_dataframe['monkey_angle'].values, in_memory_indices=ff_dataframe['point_index'].values)

    # Add some columns (they shall not be saved in csv for the sake of saving space)
    ff_dataframe['target_index'] = np.searchsorted(
        ff_caught_T_new, ff_dataframe['time'])
    ff_dataframe[['target_x', 'target_y']
                 ] = ff_real_position_sorted[ff_dataframe['target_index'].values]
    ff_dataframe['ffdistance2target'] = np.linalg.norm(np.array(
        ff_dataframe[['ff_x', 'ff_y']])-np.array(ff_dataframe[['target_x', 'target_y']]), axis=1)

    # Analyze whether ffangle is decreasing as the monkey moves
    abs_ffangle_decreasing = - np.sign(ff_dataframe['abs_delta_ff_angle'])
    ff_dataframe["abs_ffangle_decreasing"] = abs_ffangle_decreasing

    abs_ffangle_boundary_decreasing = - \
        np.sign(ff_dataframe['abs_delta_ff_angle_boundary'])
    ff_dataframe["abs_ffangle_boundary_decreasing"] = abs_ffangle_boundary_decreasing

    # Analyze whether dw is the same direction as ffangle
    ff_dataframe["IS_TARGET"] = (
        ff_dataframe["ff_index"] == ff_dataframe["target_index"])
    ff_dataframe['ff_index_string'] = ff_dataframe['ff_index'].astype('str')
    dw_same_sign_as_ffangle = np.sign(np.multiply(
        np.array(ff_dataframe["ang_speed"]), np.array(ff_dataframe["ff_angle"])))
    ff_dataframe["dw_same_sign_as_ffangle"] = dw_same_sign_as_ffangle

    dw_same_sign_as_ffangle_boundary = np.sign(np.multiply(np.array(
        ff_dataframe["ang_speed"]), np.array(ff_dataframe["ff_angle_boundary"])))
    ff_dataframe["dw_same_sign_as_ffangle_boundary"] = dw_same_sign_as_ffangle_boundary

    ff_dataframe_utils.add_caught_time_and_whether_caught_to_ff_dataframe(
        ff_dataframe, ff_caught_T_new, ff_life_sorted)

    return ff_dataframe


def make_ff_dataframe_v2_func(duration, monkey_information, ff_caught_T_new, ff_flash_sorted,
                              ff_real_position_sorted, ff_life_sorted, max_distance=500, ff_radius=10,
                              data_folder_name=None, print_progress=True):
    """
    Make a dataframe called ff_dataframe that contains various information about all visible fireflies or fireflies that are invisible but alive and 
    within max distance to the monkey at each time point in the given duration. Here we assume that the player is the monkey. 
    If RL agents are used, the algorithm needs to be modified.


    Difference between ff_dataframe and ff_dataframe_v2:
    the former contains the information of fireflies when they are either visible or in memory (and also alive), while the latter contains 
    the information of fireflies when they are either visible or invisible but alive. Thus, within the same duration, ff_dataframe_v2 
    almost always contains much more information than ff_dataframe. On the other hand, ff_dataframe_v2 only collects information within a given duration,
    but ff_dataframe uses all available data.

    Parameters
    ----------
    duration: list, (2,)
      containing a starting time and and an ending time; only the time points within the duration will be evaluated
    monkey_information: df
      containing the speed, angle, and location of the monkey at various points of time
    ff_caught_T_new: np.array
      containing the time when each captured firefly gets captured
    ff_flash_sorted: list
      containing the time that each firefly flashes on and off
    ff_real_position_sorted: np.array
      containing the real locations of the fireflies
    ff_life_sorted: np.array
      containing the time that each firefly comes into being and gets captured 
      (if the firefly is never captured, then capture time is replaced by the last point of time in data)
    max_distance: num
      the distance beyond which the firefly cannot be considered visible
    ff_radius: num
      the reward boundary of a firefly; the current setting of the game sets it to be 10
    data_folder_name: str, default is None
      the place to store the output as a csv
    print_progress: bool
      whether to print the progress of making ff_dataframe


    Returns
    -------
    ff_dataframe_v2: pd.dataframe
      containing various information about all ff at each time point as long as they are within valid distance;
      this is mostly used to aid the construction of a type of polar plot where all ff (visible or invisible) are shown

    """
    ff_index = []
    point_index = []
    time = []
    target_index = []
    ff_x = []
    ff_y = []
    monkey_x = []
    monkey_y = []
    visible = []
    ff_distance = []
    ff_angle = []
    ff_angle_boundary = []

    # First, find all fireflies that are alive in this duration
    alive_ffs = np.array([index for index, life in enumerate(
        ff_life_sorted) if (life[1] >= duration[0]) and (life[0] < duration[1])])
    for i in alive_ffs:
        # Find the corresponding information in monkey_information in the given duration:
        cum_pos_index, cum_point_index, cum_t, cum_angle, cum_mx, cum_my, cum_speed, cum_speeddummy = plot_behaviors_utils.find_monkey_information_in_the_duration(
            duration, monkey_information)

        # Find distances to ff
        distances_to_ff = np.linalg.norm(
            np.stack([cum_mx, cum_my], axis=1)-ff_real_position_sorted[i], axis=1)
        valid_distance_indices = np.where(distances_to_ff < max_distance)[0]
        angles_to_ff = specific_utils.calculate_angles_to_ff_centers(
            ff_x=ff_real_position_sorted[i, 0], ff_y=ff_real_position_sorted[i, 1], mx=cum_mx, my=cum_my, m_angle=cum_angle)
        angles_to_boundaries = specific_utils.calculate_angles_to_ff_boundaries(
            angles_to_ff=angles_to_ff, distances_to_ff=distances_to_ff, ff_radius=ff_radius)
        # Find the indices of the points where the ff is both within a max_distance and valid angles
        ff_within_range_indices = np.where((np.absolute(
            angles_to_boundaries) <= 2*pi/9) & (distances_to_ff < max_distance))[0]

        # Find indicies of fireflies that have been on at this time point
        ff_flash = ff_flash_sorted[i]
        overall_visible_indices = [index for index in ff_within_range_indices if (len(np.where(
            np.logical_and(ff_flash[:, 0] <= cum_t[index], ff_flash[:, 1] >= cum_t[index]))[0]) > 0)]
        # Also make sure that all these indices are within the ff's lifetime
        corresponding_time = cum_t[overall_visible_indices]
        alive_ff_indices = np.where((corresponding_time >= ff_life_sorted[i, 0]) & (
            corresponding_time <= ff_life_sorted[i, 1]))[0]
        overall_visible_indices = np.array(overall_visible_indices)[
            alive_ff_indices].tolist()

        # Append the values for this ff; Using list operations is faster than np.append here
        ff_index = ff_index + [i] * len(valid_distance_indices)
        point_index = point_index + \
            monkey_information.iloc[cum_pos_index[valid_distance_indices]
                                    ]['point_index'].values.tolist()
        time = time + cum_t[valid_distance_indices].tolist()
        target_index = target_index + \
            np.searchsorted(ff_caught_T_new,
                            cum_t[valid_distance_indices]).tolist()
        ff_x = ff_x + [ff_real_position_sorted[i, 0]] * \
            len(valid_distance_indices)
        ff_y = ff_y + [ff_real_position_sorted[i, 1]] * \
            len(valid_distance_indices)
        monkey_x = monkey_x + cum_mx[valid_distance_indices].tolist()
        monkey_y = monkey_y + cum_my[valid_distance_indices].tolist()
        visible_points_for_this_ff = np.zeros(len(cum_t))
        visible_points_for_this_ff[overall_visible_indices] = 1
        visible = visible + \
            visible_points_for_this_ff[valid_distance_indices].astype(
                'int').tolist()
        ff_distance = ff_distance + \
            distances_to_ff[valid_distance_indices].tolist()
        ff_angle = ff_angle + angles_to_ff[valid_distance_indices].tolist()
        ff_angle_boundary = ff_angle_boundary + \
            angles_to_boundaries[valid_distance_indices].tolist()

        if i % 100 == 0:
            if print_progress:
                print(i, " out of ", len(alive_ffs))

    # Now let's create a dictionary from the lists
    ff_dict = {'ff_index': ff_index, 'point_index': point_index, 'time': time, 'target_index': target_index,
               'ff_x': ff_x, 'ff_y': ff_y, 'monkey_x': monkey_x, 'monkey_y': monkey_y, 'visible': visible,
               'ff_distance': ff_distance, 'ff_angle': ff_angle, 'ff_angle_boundary': ff_angle_boundary}

    ff_dataframe_v2 = pd.DataFrame(ff_dict)

    # Add some columns
    ff_dataframe_v2['target_x'] = ff_real_position_sorted[np.array(
        ff_dataframe_v2['target_index'])][:, 0]
    ff_dataframe_v2['target_y'] = ff_real_position_sorted[np.array(
        ff_dataframe_v2['target_index'])][:, 1]
    ff_dataframe_v2['ffdistance2target'] = np.linalg.norm(np.array(ff_dataframe_v2[[
        'ff_x', 'ff_y']])-np.array(ff_dataframe_v2[['target_x', 'target_y']]), axis=1)
    ff_dataframe_v2['point_index_in_duration'] = ff_dataframe_v2['point_index'] - \
        monkey_information['point_index'].iloc[cum_pos_index[0]]
    ff_dataframe_v2['being_target'] = (
        ff_dataframe_v2['ff_index'] == ff_dataframe_v2['target_index']).astype('int')

    # Also to show whether each ff has been caught;
    # Since when using the agent, ff_caught_T_new does not contain information for all fireflies, we need to fill out the information for the ff not included,
    # To do so, we use the latest time in the environment plus 100s.
    ff_dataframe_utils.add_caught_time_and_whether_caught_to_ff_dataframe(
        ff_dataframe_v2, ff_caught_T_new, ff_life_sorted)

    # if a path is provided, then we will store the dataframe as a csv in the provided path
    if data_folder_name is not None:
        filepath = os.path.join(data_folder_name, 'ff_dataframe_v2.csv')
        os.makedirs(data_folder_name, exist_ok=True)
        ff_dataframe_v2.to_csv(filepath)
    return ff_dataframe_v2
