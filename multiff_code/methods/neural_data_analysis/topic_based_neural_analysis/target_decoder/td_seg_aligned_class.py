
# Third-party imports
import pandas as pd

# Scientific computing imports

# Local imports
from neural_data_analysis.topic_based_neural_analysis.target_decoder import target_decoder_class
from neural_data_analysis.neural_analysis_tools.gpfa_methods import fit_gpfa_utils, gpfa_helper_class


class TargetDecoderSegmentAlignedClass(target_decoder_class.TargetDecoderClass, gpfa_helper_class.GPFAHelperClass):

    def __init__(self,
                 raw_data_folder_path=None,
                 bin_width=0.05,
                 one_point_index_per_bin=False):

        super().__init__(raw_data_folder_path=raw_data_folder_path,
                         bin_width=bin_width,
                         one_point_index_per_bin=one_point_index_per_bin
                         )

    def prepare_seg_aligned_data(self, align_at_beginning=False):
        self.get_new_seg_info()
        self.prepare_spikes_for_gpfa(align_at_beginning=align_at_beginning)

    def get_concat_data_for_regression(self,
                                       use_lagged_rebinned_behav_data=False,
                                       use_raw_spike_data_instead=False,
                                       apply_pca_on_raw_spike_data=False,
                                       use_lagged_raw_spike_data=False):
        self.get_rebinned_behav_data(
            use_lagged_rebinned_behav_data=use_lagged_rebinned_behav_data)
        self._get_concat_data_for_regression(use_raw_spike_data_instead=use_raw_spike_data_instead,
                                             apply_pca_on_raw_spike_data=apply_pca_on_raw_spike_data,
                                             use_lagged_raw_spike_data=use_lagged_raw_spike_data)

    def get_new_seg_info(self):
        self.new_seg_info = self.y_var[[
            'segment', 'seg_start_time', 'seg_end_time', 'seg_duration']].drop_duplicates()

        # self.new_seg_info = self.single_vis_target_df[[
        #     'segment', 'seg_start_time', 'seg_end_time', 'seg_duration']].drop_duplicates()
        self.new_seg_info.rename(columns={'seg_start_time': 'new_seg_start_time',
                                 'seg_end_time': 'new_seg_end_time', 'seg_duration': 'new_seg_duration'}, inplace=True)

        self.new_seg_info['new_segment'] = pd.factorize(
            self.new_seg_info['segment'])[0]
        self.new_seg_info.sort_values(by='new_segment', inplace=True)

    def prepare_spikes_for_gpfa(self, align_at_beginning=False):
        if not hasattr(self, 'new_seg_info'):
            self.get_new_seg_info()

        gpfa_helper_class.GPFAHelperClass._prepare_spikes_for_gpfa(
            self, self.new_seg_info, align_at_beginning=align_at_beginning)

    def get_rebinned_behav_data(self, use_lagged_rebinned_behav_data=False,
                                use_reduced_behav_data_for_gpfa=True):
        if not hasattr(self, 'new_seg_info'):
            self.get_new_seg_info()

        # we won't rebin the behav data here, but we'll make sure that the new segments match the new_seg_info
        self.use_lagged_rebinned_behav_data = use_lagged_rebinned_behav_data
        self.use_reduced_behav_data_for_gpfa = use_reduced_behav_data_for_gpfa

        if use_lagged_rebinned_behav_data:
            y_var_df = self.y_var_lags_reduced if use_reduced_behav_data_for_gpfa else self.y_var_lags
        else:
            y_var_df = self.y_var_reduced if use_reduced_behav_data_for_gpfa else self.y_var

        self.rebinned_behav_data = y_var_df.copy()
        self.rebinned_behav_data[['segment', 'bin']] = self.y_var[[
            'segment', 'bin']].values.astype(int)

        self.rebinned_behav_data = self.rebinned_behav_data.sort_values(
            by=['segment', 'bin'])

        # also assign new_segment as a continuous array starting from 0 based on segment order
        self.rebinned_behav_data['new_segment'] = pd.factorize(
            self.rebinned_behav_data['segment'])[0]

        if self.align_at_beginning:
            # within each segment, reassign new_bin as a continuous array starting from 0 based on bin order
            self.rebinned_behav_data['new_bin'] = (
                self.rebinned_behav_data.groupby('new_segment').cumcount())
        else:
            self.rebinned_behav_data = fit_gpfa_utils.assign_new_bin_aligned_at_end(
                self.rebinned_behav_data, new_segment_column='new_segment')

        assert set(self.rebinned_behav_data['new_segment'].unique()) == set(
            self.new_seg_info['new_segment'].unique())

    def get_raw_spikes_for_regression(self):
        if self.use_lagged_raw_spike_data:
            x_var_df = self.x_var_lags.copy()
        else:
            x_var_df = self.x_var.copy()

        x_var_df.sort_values(by=['segment', 'bin'], inplace=True)
        assert x_var_df['bin'].equals(self.rebinned_behav_data['bin'])

        # drop all columns in x_var_df that do not start with 'cluster_'
        x_var_df = x_var_df[[
            col for col in x_var_df.columns if col.startswith('cluster_')]]

        x_var_df[['new_bin', 'new_segment']] = self.rebinned_behav_data[[
            'new_bin', 'new_segment']].values
        return x_var_df
