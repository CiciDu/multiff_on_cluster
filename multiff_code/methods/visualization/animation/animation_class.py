from data_wrangling import specific_utils, further_processing_class
from visualization.matplotlib_tools import plot_trials
from visualization.animation import animation_utils, animation_func
from data_wrangling import specific_utils

import os
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from functools import partial
from matplotlib import animation
from IPython.display import Video


plt.rcParams["animation.html"] = "html5"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rc('animation', html='jshtml')
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams['animation.embed_limit'] = 2**128
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(suppress=True)


class AnimationClass(further_processing_class.FurtherProcessing):

    def __init__(self, raw_data_folder_path=None):
        super().__init__(raw_data_folder_path=raw_data_folder_path)

    def set_animation_parameters(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None, k=3, static_plot_on_the_left=False, max_num_frames=150, max_duration=30, min_duration=1, rotated=True):
        # Among currentTrial, num_trials, duration, either currentTrial and num_trials must be specified, or duration must be specified
        currentTrial, num_trials, duration = specific_utils.find_currentTrial_or_num_trials_or_duration(
            self.ff_caught_T_new, currentTrial, num_trials, duration)

        # if the duration is too short, then increase the number of trials
        while duration[1] - duration[0] < 0.1:
            num_trials = num_trials+1
            duration = [self.ff_caught_T_new[currentTrial -
                                             num_trials], self.ff_caught_T_new[currentTrial]]

        if static_plot_on_the_left:
            self.fig, self.ax = self._make_static_plot_on_the_left(
                currentTrial=currentTrial, num_trials=num_trials, duration=duration, animation_plot_kwargs=animation_plot_kwargs)
        else:
            self.fig, self.ax = plt.subplots()

        self.currentTrial = currentTrial
        self.num_trials = num_trials
        self.duration = duration
        self.k = k

        self._call_prepare_for_animation_func(currentTrial=currentTrial, num_trials=num_trials, duration=duration, k=k, max_num_frames=max_num_frames,
                                              max_duration=max_duration, min_duration=min_duration, rotated=rotated)

    def call_animation_function(self, with_annotation=False,
                                margin=100, dt=0.0165, plot_time_index=False, fps=None,
                                save_video=True, video_dir=None, file_name=None,
                                **animate_kwargs):
        # dt is to be used to determine the fps(frame per second) of the animation

        # if the key "margin" is in animate_kwargs, then put it into animate_kwargs
        # if not, then put the default value (500) into animate_kwargs
        if 'margin' not in animate_kwargs.keys():
            animate_kwargs['margin'] = margin

        if with_annotation:
            self.annotation_info = animation_utils.make_annotation_info(self.caught_ff_num+1, self.max_point_index, self.n_ff_in_a_row, self.visible_before_last_one_trials, self.disappear_latest_trials,
                                                                        self.ignore_sudden_flash_indices, self.GUAT_indices_df['point_index'].values, self.try_a_few_times_indices)
            animate_func = partial(animation_func.animate_annotated, ax=self.ax, anim_monkey_info=self.anim_monkey_info, ff_dataframe_anim=self.ff_dataframe_anim,
                                   flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, ff_caught_T_new=self.ff_caught_T_new, annotation_info=self.annotation_info,
                                   plot_time_index=plot_time_index, **animate_kwargs)
        else:
            animate_func = partial(animation_func.animate, ax=self.ax, anim_monkey_info=self.anim_monkey_info, ff_dataframe_anim=self.ff_dataframe_anim,
                                   flash_on_ff_dict=self.flash_on_ff_dict, alive_ff_dict=self.alive_ff_dict, believed_ff_dict=self.believed_ff_dict, plot_time_index=plot_time_index, **animate_kwargs)
        self.anim = animation.FuncAnimation(
            self.fig, animate_func, frames=self.num_frames, interval=dt*1000*self.k, repeat=True)

        if save_video:
            self._save_animation(fps, video_dir, file_name)
            try:
                print('Rendering animation ......')
                Video(self.video_path_name, embed=True)
            except Exception as e:
                print('Error rendering animation:', e)


    def _call_prepare_for_animation_func(self, currentTrial=None, num_trials=None, duration=None, k=1, max_num_frames=None,
                                         max_duration=30, min_duration=1, rotated=True):
        self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
            = animation_utils.prepare_for_animation(
                self.ff_dataframe, self.ff_caught_T_new, self.ff_life_sorted, self.ff_believed_position_sorted,
                self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k=k, currentTrial=currentTrial, num_trials=num_trials, duration=duration,
                max_duration=max_duration, min_duration=min_duration, rotated=rotated)
        print("Number of frames is:", self.num_frames)

        # if the number of frames is too large, then reduce k so that the number of frames is reduced
        if max_num_frames is not None:
            while self.num_frames > 150:
                self.k = self.k + 1
                self.num_frames, self.anim_monkey_info, self.flash_on_ff_dict, self.alive_ff_dict, self.believed_ff_dict, self.new_num_trials, self.ff_dataframe_anim \
                    = animation_utils.prepare_for_animation(self.ff_dataframe, self.ff_caught_T_new, self.ff_life_sorted, self.ff_believed_position_sorted,
                                                            self.ff_real_position_sorted, self.ff_flash_sorted, self.monkey_information, k=self.k, currentTrial=currentTrial, num_trials=num_trials,
                                                            duration=duration)
        print("Number of frames for the animation is:", self.num_frames)

    def _make_static_plot_on_the_left(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None):
        self.fig = plt.figure(figsize=(14.5, 7))
        PlotTrials_args = (self.monkey_information, self.ff_dataframe, self.ff_life_sorted, self.ff_real_position_sorted,
                           self.ff_believed_position_sorted, self.cluster_around_target_indices, self.ff_caught_T_new)

        if animation_plot_kwargs is None:
            animation_plot_kwargs = self.animation_plot_kwargs

        ax1 = self.fig.add_subplot(1, 2, 1)
        returned_info = plot_trials.PlotTrials(duration,
                                               fig=self.fig,
                                               axes=ax1,
                                               *PlotTrials_args,
                                               **animation_plot_kwargs,
                                               currentTrial=currentTrial,
                                               num_trials=num_trials,
                                               )
        ax1 = returned_info['axes']
        self.fig.add_axes(ax1)
        self.ax = self.fig.add_subplot(1, 2, 2)
        self.fig.tight_layout()

        return self.fig, self.ax

    def _save_animation(self, fps, video_dir, file_name):
        if fps is None:
            if self.player == 'agent':
                fps = int(4/self.k)  # the real life speed
            else:
                fps = int(62/self.k)

        if video_dir is None:
            video_dir = self.processed_data_folder_path

        if file_name is None:
            try:
                file_name = self.agent_id + \
                    f'__{self.currentTrial-self.num_trials+1}-{self.currentTrial}.mp4'
            except TypeError:
                file_name = self.agent_id + \
                    f'__{self.duration[0]}s_to_{self.duration[1]}s.mp4'

        os.makedirs(video_dir, exist_ok=True)
        self.video_path_name = f"{video_dir}/{file_name}"

        print("Saving animation as:", self.video_path_name)
        writervideo = animation.FFMpegWriter(fps=fps)
        self.anim.save(self.video_path_name, writer=writervideo)
        print("Animation is saved at:", self.video_path_name)

        # save animation as gif
        # self.anim.save(f"{self.processed_data_folder_path}/agent_animation.gif", writer='imagemagick', fps=int(62/self.k)) #SB3
        # self.anim.save(f"{self.processed_data_folder_path}/agent_animation.mp4", writer=writervideo)

    def make_animation(self, currentTrial=None, num_trials=None, duration=None, animation_plot_kwargs=None,
                       save_video=False, video_dir=None, file_name=None,
                       dt=0.0165, k=3, with_annotation=False, plot_time_index=False,
                       static_plot_on_the_left=True, margin=100, max_num_frames=150, max_duration=30, min_duration=1,
                       fps=None, rotated=True, **animate_kwargs):
        if file_name is None:
            if currentTrial is not None:
                file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
            elif duration is not None:
                file_name = f"{self.data_name}_{duration[0]}s-{duration[1]}s.mp4"
            else:
                file_name = f"{self.data_name}_animation.mp4"
        if save_video:
            if video_dir is None:
                video_dir = 'videos'
                print("No video was set, so the default path -- 'videos' -- is used.")

        animate_kwargs['margin'] = margin
        self.set_animation_parameters(currentTrial=currentTrial, num_trials=num_trials, duration=duration, animation_plot_kwargs=animation_plot_kwargs, k=k, rotated=rotated,
                                      static_plot_on_the_left=static_plot_on_the_left, max_num_frames=max_num_frames, max_duration=max_duration, min_duration=min_duration)
        self.call_animation_function(save_video=save_video, video_dir=video_dir, plot_time_index=plot_time_index,
                                     file_name=file_name, dt=dt, with_annotation=with_annotation, fps=fps, **animate_kwargs)

    def make_animation_from_a_category(self, category_name, max_trial_to_plot, sampling_frame_ratio=3, max_duration=30, min_duration=1,
                                       num_trials=3, save_video=True, video_dir=None, additional_kwargs=None, animation_exists_ok=True,
                                       with_annotation=False, dt=0.0165, static_plot_on_the_left=True, max_num_frames=None, **animate_kwargs):
        '''
        Create animations for a given category of trials.
        '''

        # note: if save_video is True, but video_dir is None, then video_dir is set to be the same as self.video_dir eventually
        k = sampling_frame_ratio
        category = self.all_categories[category_name]

        self.video_dir = os.path.join(
            'patterns', category_name, self.monkey_name)
        video_dir = video_dir if video_dir is not None else self.video_dir

        animation_plot_kwargs = self.all_category_animation_kwargs[category_name]
        # instead, video_dir is used later
        animation_plot_kwargs['images_dir'] = None

        if additional_kwargs is not None:
            animation_plot_kwargs.update(additional_kwargs)

        if len(category) == 0:
            print(f"No trials found for category: {category_name}")
            return

        with general_utils.initiate_plot(10, 10, 100):
            for currentTrial in category[:max_trial_to_plot]:
                file_name = f"{self.data_name}_{currentTrial-num_trials+1}-{currentTrial}.mp4"
                video_path_name = os.path.join(self.video_dir, file_name)

                if animation_exists_ok and exists(video_path_name):
                    print(
                        f"Animation for trial {currentTrial} already exists at {video_path_name}. Skipping.")
                    continue

                self.make_animation(
                    currentTrial=currentTrial,
                    num_trials=num_trials,
                    animation_plot_kwargs=animation_plot_kwargs,
                    static_plot_on_the_left=static_plot_on_the_left,
                    save_video=save_video,
                    video_dir=video_dir,
                    file_name=file_name,
                    with_annotation=with_annotation,
                    dt=dt,
                    max_num_frames=max_num_frames,
                    max_duration=max_duration,
                    min_duration=min_duration,
                    **animate_kwargs
                )

    def make_animation_of_chunks(self, df_with_chunks, monkey_information, chunk_numbers=range(10), sampling_frame_ratio=3, additional_kwargs=None, exists_ok=True,
                                 max_duration=30, min_duration=1, dt=0.016, save_video=True, static_plot_on_the_left=True, max_num_frames=None, **animate_kwargs):
        k = sampling_frame_ratio
        self.video_dir = os.path.join(
            'chunks', self.monkey_name, self.data_name)
        animation_plot_kwargs = self.animation_plot_kwargs
        animation_plot_kwargs['images_dir'] = None

        if additional_kwargs is not None:
            animation_plot_kwargs.update(additional_kwargs)

        with general_utils.initiate_plot(10, 10, 100):
            for chunk in chunk_numbers:
                chunk_df = df_with_chunks[df_with_chunks['chunk'] == chunk]
                duration_points = [
                    chunk_df['point_index'].min(), chunk_df['point_index'].max()]
                duration = [monkey_information['time'][duration_points[0]],
                            monkey_information['time'][duration_points[0]]+10]

                file_name = f"chunk{chunk}_{round(duration[0], 1)}s-{round(duration[1], 1)}s.mp4"
                video_path_name = os.path.join(self.video_dir, file_name)
                if exists_ok & exists(os.path.join(self.video_dir, file_name)):
                    print("Animation for the current chunk already exists at",
                          video_path_name, "Moving on to the next trial.")
                    continue
                self.make_animation(duration=duration, animation_plot_kwargs=animation_plot_kwargs, save_video=save_video, file_name=file_name, dt=dt, max_duration=max_duration, min_duration=min_duration,
                                    video_dir=self.video_dir, static_plot_on_the_left=static_plot_on_the_left, max_num_frames=max_num_frames, **animate_kwargs)
