from decision_making_analysis.cluster_replacement import cluster_replacement_utils
from decision_making_analysis import free_selection, trajectory_info
from null_behaviors import curvature_utils, curv_of_traj_utils
from decision_making_analysis.decision_making import decision_making_class, plot_decision_making
from decision_making_analysis.cluster_replacement import plot_cluster_replacement

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


class GUATHelperClass(decision_making_class.DecisionMaking):

    def process_current_and_alternative_ff_info(self,
                                                num_old_ff_per_row=2,
                                                num_new_ff_per_row=2,
                                                selection_criterion_if_too_many_ff='time_since_last_vis'):

        # The following columns are added so the function further_process_df_related_to_cluster_replacement can be called without errors.
        # But the values in these columns will be updated later
        self.miss_abort_cur_ff_info[['whether_changed',
                                     'whether_intended_target']] = False
        self.miss_abort_nxt_ff_info[['whether_changed',
                                     'whether_intended_target']] = False

        self.miss_abort_cur_ff_info, self.miss_abort_nxt_ff_info = cluster_replacement_utils.further_process_df_related_to_cluster_replacement(self.miss_abort_cur_ff_info, self.miss_abort_nxt_ff_info,
                                                                                                                                               num_old_ff_per_row=num_old_ff_per_row, num_new_ff_per_row=num_new_ff_per_row, selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff)
        self.miss_abort_nxt_ff_info['order'] = self.miss_abort_nxt_ff_info['order'] + \
            num_old_ff_per_row

        self.num_old_ff_per_row = num_old_ff_per_row
        self.num_new_ff_per_row = num_new_ff_per_row
        self.num_ff_per_row = num_old_ff_per_row + num_new_ff_per_row

    def find_input_and_output(self,
                              add_arc_info=False,
                              add_current_curv_of_traj=False,
                              curvature_df=None,
                              curv_of_traj_df=None,
                              window_for_curv_of_traj=[-25, 0],
                              curv_of_traj_mode='distance',
                              truncate_curv_of_traj_by_time_of_capture=False,
                              ff_attributes=[
                                  'ff_distance', 'ff_angle', 'time_since_last_vis', 'time_till_next_visible'],
                              arc_info_to_add=['opt_arc_curv', 'curv_diff']):
        if curv_of_traj_df is None:
            self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
                window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            self.curv_of_traj_df = curv_of_traj_df

        attributes_for_plotting = ['ff_distance',
                                   'ff_angle', 'time_since_last_vis']
        if 'time_till_next_visible' in ff_attributes:
            # self.add_time_till_next_visible()
            attributes_for_plotting.append('time_till_next_visible')

        if curvature_df is not None:
            if self.additional_curvature_df is not None:
                curvature_df = pd.concat(
                    [curvature_df, self.additional_curvature_df], axis=0)
                curvature_df_sub = curvature_df[[
                    'point_index', 'ff_index']].copy()
                curvature_df = curvature_df[~curvature_df_sub.duplicated()]

        self.GUAT_joined_ff_info = pd.concat(
            [self.miss_abort_cur_ff_info, self.miss_abort_nxt_ff_info], axis=0)
        if add_arc_info:
            self.GUAT_joined_ff_info = curvature_utils.add_arc_info_to_df(
                self.GUAT_joined_ff_info, curvature_df, arc_info_to_add=arc_info_to_add, ff_caught_T_new=self.ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))
        self.free_selection_x_df, self.free_selection_x_df_for_plotting, self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(self.GUAT_joined_ff_info, self.monkey_information, ff_attributes=ff_attributes, attributes_for_plotting=attributes_for_plotting,
                                                                                                                                                                                                                   num_ff_per_row=self.num_old_ff_per_row + self.num_new_ff_per_row, add_current_curv_of_traj=add_current_curv_of_traj, ff_caught_T_new=self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)
        self.free_selection_time = self.monkey_information.loc[self.point_index_array, 'time'].values
        self.num_stops = self.num_stops_df.set_index(
            'point_index').loc[self.point_index_array, 'num_stops'].values
        self.y_value = (self.num_stops > 2).astype(int)

    def add_additional_info_to_plot(self, time_range_of_trajectory,
                                    num_time_points_for_trajectory,
                                    ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis', 'time_till_next_visible']):

        if (self.all_available_ff_in_near_future is not None) & ('time_till_next_visible' in ff_attributes):
            self.more_ff_df, self.more_ff_inputs_df_for_plotting = cluster_replacement_utils.find_more_ff_inputs_for_plotting(self.point_index_all, self.sequence_of_obs_ff_indices, self.ff_dataframe, self.ff_real_position_sorted,
                                                                                                                              self.monkey_information, ff_attributes=ff_attributes, all_available_ff_in_near_future=self.all_available_ff_in_near_future)
            _, self.more_traj_points, self.more_traj_stops, _ = trajectory_info.furnish_machine_learning_data_with_trajectory_data_func(self.X_all, self.time_all, self.monkey_information,
                                                                                                                                        trajectory_data_kind=self.trajectory_data_kind, time_range_of_trajectory=time_range_of_trajectory,
                                                                                                                                        num_time_points_for_trajectory=num_time_points_for_trajectory, add_traj_stops=self.add_traj_stops)
            self.more_ff_inputs = self.more_ff_inputs_df_for_plotting.values
            self.more_ff_inputs_to_plot = self.more_ff_inputs[self.indices_test]
            self.more_traj_points_to_plot = self.more_traj_points[self.indices_test]
            self.more_traj_stops_to_plot = self.more_traj_stops[self.indices_test]
        else:
            super().add_additional_info_to_plot(time_range_of_trajectory,
                                                num_time_points_for_trajectory, ff_attributes=ff_attributes)

    def plot_prediction_results(self, selected_cases=None, max_plot_to_make=40, show_direction_of_monkey_on_trajectory=False, show_reward_boundary=False,
                                use_more_ff_inputs=False, use_more_traj_points=False, max_time_since_last_vis=3, predict_num_stops=False, additional_plotting_kwargs={}):

        # if self.polar_plots_kwargs is None:
        self.prepare_to_plot_prediction_results(use_more_ff_inputs=use_more_ff_inputs, use_more_traj_points=use_more_traj_points,
                                                show_direction_of_monkey_on_trajectory=show_direction_of_monkey_on_trajectory)
        for key, value in additional_plotting_kwargs.items():
            self.polar_plots_kwargs[key] = value

        self.polar_plots_kwargs['y_prob'] = None

        self.make_polar_plots_for_cluster_replacement(selected_cases=selected_cases,
                                                      max_plot_to_make=max_plot_to_make,
                                                      show_reward_boundary=show_reward_boundary,
                                                      max_time_since_last_vis=max_time_since_last_vis,
                                                      predict_num_stops=predict_num_stops,
                                                      )

    def make_polar_plots_for_cluster_replacement(self, selected_cases=None, max_plot_to_make=5, show_reward_boundary=False,
                                                 max_time_since_last_vis=3, ff_colormap='Greens',
                                                 predict_num_stops=False, ):

        if selected_cases is not None:
            instance_to_plot = selected_cases[:max_plot_to_make]
        else:
            instance_to_plot = np.arange(self.polar_plots_kwargs['ff_inputs'].shape[0])[
                :max_plot_to_make]

        for i in instance_to_plot:
            self.current_polar_plot_kargs = plot_decision_making.get_current_polar_plot_kargs(i, max_time_since_last_vis=max_time_since_last_vis,
                                                                                              show_reward_boundary=show_reward_boundary, ff_colormap=ff_colormap, **self.polar_plots_kwargs)

            ax = plot_cluster_replacement.make_one_polar_plot_for_cluster_replacement(
                self.num_old_ff_per_row, predict_num_stops=predict_num_stops, **self.current_polar_plot_kargs)

            plt.show()

    def find_input_and_output_for_cluster_replacement(self, num_old_ff_per_row=2, num_new_ff_per_row=2, selection_criterion_if_too_many_ff='time_since_last_vis', sorting_criterion=None,
                                                      add_arc_info=False, add_current_curv_of_traj=False, curvature_df=None, curv_of_traj_df=None, ff_attributes=['ff_distance', 'ff_angle', 'time_since_last_vis'],
                                                      window_for_curv_of_traj=[-25, 0], curv_of_traj_mode='distance', truncate_curv_of_traj_by_time_of_capture=False,
                                                      arc_info_to_add=['opt_arc_curv', 'curv_diff']):
        if curv_of_traj_df is None:
            self.curv_of_traj_df, traj_curv_descr = curv_of_traj_utils.find_curv_of_traj_df_based_on_curv_of_traj_mode(
                window_for_curv_of_traj, self.monkey_information, self.ff_caught_T_new, curv_of_traj_mode=curv_of_traj_mode, truncate_curv_of_traj_by_time_of_capture=truncate_curv_of_traj_by_time_of_capture)
        else:
            self.curv_of_traj_df = curv_of_traj_df

        self.num_old_ff_per_row = num_old_ff_per_row
        self.num_new_ff_per_row = num_new_ff_per_row
        self.old_ff_cluster_df, self.new_ff_cluster_df, self.parallel_old_ff_cluster_df, self.non_chosen_ff_cluster_df = \
            cluster_replacement_utils.find_df_related_to_cluster_replacement(self.replacement_df, self.prior_to_replacement_df, self.non_chosen_df, self.manual_anno,
                                                                             self.ff_dataframe, self.monkey_information, self.ff_real_position_sorted, self.ff_life_sorted, sample_size=None, equal_sample_from_two_cases=True)

        self.joined_old_ff_cluster_df = pd.concat(
            [self.old_ff_cluster_df, self.parallel_old_ff_cluster_df], axis=0)
        self.joined_new_ff_cluster_df = pd.concat(
            [self.new_ff_cluster_df, self.non_chosen_ff_cluster_df], axis=0)

        self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df = cluster_replacement_utils.further_process_df_related_to_cluster_replacement(self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df, num_old_ff_per_row=num_old_ff_per_row, num_new_ff_per_row=num_new_ff_per_row,
                                                                                                                                                   selection_criterion_if_too_many_ff=selection_criterion_if_too_many_ff, sorting_criterion=sorting_criterion)
        self.joined_new_ff_cluster_df['order'] = self.joined_new_ff_cluster_df['order'] + \
            num_old_ff_per_row
        self.joined_cluster_df = pd.concat(
            [self.joined_old_ff_cluster_df, self.joined_new_ff_cluster_df], axis=0)
        if add_arc_info:
            self.joined_cluster_df = curvature_utils.add_arc_info_to_df(
                self.joined_cluster_df, curvature_df, arc_info_to_add=arc_info_to_add, ff_caught_T_new=self.ff_caught_T_new, curv_of_traj_df=curv_of_traj_df)
            ff_attributes = list(set(ff_attributes) | set(arc_info_to_add))
        self.free_selection_x_df, self.free_selection_x_df_for_plotting, self.sequence_of_obs_ff_indices, self.point_index_array, self.pred_var = free_selection.find_free_selection_x_from_info_of_n_ff_per_point(self.joined_cluster_df, self.monkey_information, ff_attributes=ff_attributes,
                                                                                                                                                                                                                   num_ff_per_row=num_old_ff_per_row + num_new_ff_per_row, add_current_curv_of_traj=add_current_curv_of_traj, ff_caught_T_new=self.ff_caught_T_new, curv_of_traj_df=self.curv_of_traj_df)
        self.free_selection_time = self.monkey_information.loc[self.point_index_array, 'time'].values
        # incorporate whether_changed
        self.whether_changed = self.joined_cluster_df[[
            'point_index', 'whether_changed']].drop_duplicates()
        self.y_value = self.whether_changed['whether_changed'].values.astype(
            int).copy()
