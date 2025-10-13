from pattern_discovery import patterns_and_features_class
from visualization.matplotlib_tools import plot_change_over_time

import os
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class CompareTwoMonkeys():

    def __init__(self):
        self.bruno = patterns_and_features_class.PatternsAndFeatures(
            monkey_name='monkey_Bruno')
        self.schro = patterns_and_features_class.PatternsAndFeatures(
            monkey_name='monkey_Schro')

    def compare_monkeys(self, verbose=True, exists_ok=True):

        self.bruno.combine_or_retrieve_patterns_and_features(
            verbose=verbose, exists_ok=exists_ok)
        self.schro.combine_or_retrieve_patterns_and_features(
            verbose=verbose, exists_ok=exists_ok)
        self.monkey_name = ''
        self.combine_df()

    def combine_df(self, df_names=['agg_pattern_frequencies', 'agg_feature_statistics', 'combd_pattern_frequencies',
                                   'combd_feature_statistics', 'combd_all_trial_features', 'combd_scatter_around_target_df']):
        for df_name in df_names:
            bruno_df = getattr(self.bruno, df_name)
            schro_df = getattr(self.schro, df_name)
            bruno_df['monkey'] = 'Bruno'
            schro_df['monkey'] = 'Schro'
            setattr(self, df_name, pd.concat(
                [bruno_df, schro_df], axis=0).reset_index(drop=True))
            print(f'Made {df_name} with shape {getattr(self, df_name).shape}')

    def plot_feature_statistics(self):
        patterns_and_features_class.PatternsAndFeatures.plot_feature_statistics(
            self, hue='monkey')

    def plot_pattern_frequencies(self):
        patterns_and_features_class.PatternsAndFeatures.plot_pattern_frequencies(
            self, hue='monkey')

    def plot_the_changes_in_pattern_frequencies_over_time(self):
        self.pattern_order = patterns_and_features_class.PatternsAndFeatures.pattern_order
        patterns_and_features_class.PatternsAndFeatures.plot_the_changes_in_pattern_frequencies_over_time(
            self, multiple_monkeys=True)

    def plot_the_changes_in_feature_statistics_over_time(self):
        self.feature_order = patterns_and_features_class.PatternsAndFeatures.feature_order
        patterns_and_features_class.PatternsAndFeatures.plot_the_changes_in_feature_statistics_over_time(
            self, multiple_monkeys=True)

    def plot_the_changes_in_scatter_around_target_over_time(self):
        for y_column_list in [['distance_mean', 'distance_50%'],
                              ['abs_angle_mean', 'abs_angle_50%'],
                              # ['distance_mean', 'distance_Q1', 'distance_median', 'distance_Q3']
                              ]:
            plot_change_over_time.plot_the_changes_over_time_in_wide_df(self.combd_scatter_around_target_df, x='session',
                                                                        y_columns=y_column_list,
                                                                        monkey_name=self.monkey_name,
                                                                        multiple_monkeys=True,
                                                                        title_prefix='Landing position from ff center: '
                                                                        )

    def prepare_to_compare_success_rates(self):
        self.bruno.combine_or_retrieve_patterns_and_features(
            verbose=False, exists_ok=True)
        self.schro.combine_or_retrieve_patterns_and_features(
            verbose=False, exists_ok=True)

        bruno_sub = self.bruno.agg_pattern_frequencies[self.bruno.agg_pattern_frequencies['item'].isin(
            ['ff_capture_rate', 'stop_success_rate'])].copy()
        bruno_sub = bruno_sub[['frequency', 'label', 'rate']].reset_index(
            drop=True).rename(columns={'frequency': 'Total Captured FF'})
        bruno_sub['monkey'] = 'Bruno'

        schro_sub = self.schro.agg_pattern_frequencies[self.schro.agg_pattern_frequencies['item'].isin(
            ['ff_capture_rate', 'stop_success_rate'])].copy()
        schro_sub = schro_sub[['frequency', 'label', 'rate']].reset_index(
            drop=True).rename(columns={'frequency': 'Total Captured FF'})
        schro_sub['monkey'] = 'Schro'

        success_rate_df = pd.concat(
            [bruno_sub, schro_sub], axis=0).reset_index(drop=True)
        success_rate_df = success_rate_df.melt(id_vars=['monkey', 'label'], value_vars=['rate']).sort_values(
            by=['label', 'monkey']).drop(columns=['variable']).reset_index(drop=True)
        self.success_rate_df = success_rate_df.rename(
            columns={'label': 'statistic'})

    def make_plot_to_compare_success_rates(self):
        fig = px.bar(self.success_rate_df, x='statistic', y='value', color='monkey', barmode='group',
                     text='value', title='FF Capture Rate and Stop Success Rate', width=500)  # Adjust the width as needed
        # make the text only 3 decimal
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        # increase y lim a little bit to make sure no text is blocked
        fig.update_yaxes(range=[0, max(self.success_rate_df['value']) + 0.1])
        plt.show()
