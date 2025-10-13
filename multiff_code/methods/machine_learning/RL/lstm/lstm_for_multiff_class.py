

from machine_learning.RL.env_related import env_for_lstm
from machine_learning.RL.SB3 import rl_for_multiff_utils, rl_for_multiff_class
from machine_learning.RL.lstm import LSTM_functions

import os
import matplotlib.pyplot as plt
import gc
import torch
import pickle
import io
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class LSTMforMultifirefly(rl_for_multiff_class._RLforMultifirefly):

    def __init__(self,
                 overall_folder='RL_models/LSTM_stored_models/all_agents/gen_0/',
                 model_folder_name=None,
                 add_date_to_model_folder_name=False,
                 max_in_memory_time=1,
                 **additional_env_kwargs):

        super().__init__(overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         model_folder_name=model_folder_name,
                         max_in_memory_time=max_in_memory_time,
                         **additional_env_kwargs)

        self.sb3_or_lstm = 'lstm'
        self.env_class = env_for_lstm.EnvForLSTM

    def make_env(self, **additional_env_kwargs):
        self.env_kwargs.update(additional_env_kwargs)
        self.env = self.env_class(**self.env_kwargs)

    def make_agent(self,
                   agent_params_already_set_ok=True,
                   **kwargs):

        if agent_params_already_set_ok & (self.agent_params is not None):
            existing_agent_params = self.agent_params.copy()
        else:
            existing_agent_params = {}

        self.agent_params = {
            "gamma": rl_for_multiff_utils.calculate_model_gamma(self.env.dt),
            "state_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "action_dim": self.env.action_space.shape[0],
            "action_range": 1.0,
            # used 100 before
            "replay_buffer_size": kwargs.get('replay_buffer_size', 500),
            # "replay_buffer": GRU_functions.ReplayBufferGRU(100),  # Uncomment if using GRU
            "hidden_dim": kwargs.get('hidden_dim', 128),
            "soft_q_lr": kwargs.get('soft_q_lr', 0.0015),
            "policy_lr": kwargs.get('policy_lr', 0.003),
            "alpha_lr": kwargs.get('alpha_lr', 0.002),
            "batch_size": kwargs.get('batch_size', 10),
            "update_itr": kwargs.get('update_itr', 1),
            "reward_scale": kwargs.get('reward_scale', 10),  # used 10 before
            "target_entropy": kwargs.get('target_entropy', -2),
            "soft_tau": kwargs.get('soft_tau', 0.015),
            "train_freq": kwargs.get('train_freq', 100),
            "auto_entropy": kwargs.get('auto_entropy', True),
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
        }

        self.agent_params.update(existing_agent_params)
        self.agent_params.update(kwargs)

        # if self.use_env2:
        #     self.agent_params['hidden_dim'] = 800

        self.replay_buffer = LSTM_functions.ReplayBufferLSTM2(
            self.agent_params['replay_buffer_size'])
        self.agent_params['replay_buffer'] = self.replay_buffer
        self.sac_model = LSTM_functions.SAC_Trainer(**self.agent_params)

    def make_initial_env_for_curriculum_training(self, initial_dt=0.1, initial_flash_on_interval=3, initial_angular_terminal_vel=0.64, initial_reward_boundary=75):
        self.make_env()
        self._make_initial_env_for_curriculum_training(initial_dt=initial_dt,
                                                         initial_angular_terminal_vel=initial_angular_terminal_vel)
        self.env.angular_terminal_vel = initial_angular_terminal_vel
        self._update_env_flash_on_interval(
            flash_on_interval=initial_flash_on_interval)
        # optionally relax reward boundary to densify rewards at start
        if initial_reward_boundary is not None:
            self._update_env_reward_boundary(reward_boundary=initial_reward_boundary)

    def _make_agent_for_curriculum_training(self):
        self.make_agent()

    def _update_env_dt(self, dt=0.25):
        self.env.dt = dt
        self.sac_model.gamma = rl_for_multiff_utils.calculate_model_gamma(
            self.env.dt)
        self.env_kwargs_for_curriculum_training['dt'] = dt

    def _update_env_flash_on_interval(self, flash_on_interval=0.3):
        self.env.flash_on_interval = flash_on_interval
        self.env_kwargs_for_curriculum_training['flash_on_interval'] = flash_on_interval

    def _update_env_reward_boundary(self, reward_boundary=25):
        # base_env.MultiFF exposes `reward_boundary`
        self.env.reward_boundary = reward_boundary
        self.env_kwargs_for_curriculum_training['reward_boundary'] = reward_boundary

    def _change_env_after_meeting_reward_threshold(self):
        flash_on_interval = max(
            self.env.flash_on_interval - 0.3, self.env_kwargs['flash_on_interval'])
        self._update_env_flash_on_interval(flash_on_interval=flash_on_interval)
        # self._update_env_dt(dt=max(self.env.dt/2, 0.1))
        self.env.angular_terminal_vel = max(self.env.angular_terminal_vel/4, 0.01)
        self.env.distance2center_cost = max(self.env.distance2center_cost - 0.5, 0)
        # shrink reward boundary towards target in env_kwargs (harder over time)
        if 'reward_boundary' in self.env_kwargs:
            new_rb = max(self.env.reward_boundary - 25 , self.env_kwargs['reward_boundary'])
            self._update_env_reward_boundary(reward_boundary=new_rb)

        print('Current dt:', self.env.dt)
        print('Current gamma:', self.sac_model.gamma)
        print('Current angular_terminal_vel:', self.env.angular_terminal_vel)
        print('Current flash_on_interval:', self.env.flash_on_interval)
        print('Current distance2center_cost:', self.env.distance2center_cost)
        print('Current reward_boundary:', self.env.reward_boundary)

    def _use_while_loop_for_curriculum_training(self, eval_eps_freq=20):
        # while (self.env.dt > self.env_kwargs['dt']) | (self.env.angular_terminal_vel > 0.01) | \
        #     (self.env.flash_on_interval > self.env_kwargs['flash_on_interval']) | (self.env.distance2center_cost > 0):

        while (self.env.flash_on_interval > self.env_kwargs['flash_on_interval']) or \
              ('reward_boundary' in self.env_kwargs_for_curriculum_training and self.env.reward_boundary > self.env_kwargs['reward_boundary']):

            gc.collect()
            # Note: 0.00222 = 0.0035/(pi/2), same as the monkey's threshold
            num_eval_episodes = 1
            reward_threshold = rl_for_multiff_utils.calculate_reward_threshold_for_curriculum_training(
                self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
            print('Current reward_threshold to progress in curriculum training:', reward_threshold)
            # reward_threshold = 1000
            self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                                  reward_threshold_to_stop_on=reward_threshold, dir_name=self.best_model_after_curriculum_dir_name,
                                  env_params_to_save=self.env_kwargs_for_curriculum_training)
            
            self._change_env_after_meeting_reward_threshold()

        # after all condition is met, train the agent once more until it reaches the desired performance
        reward_threshold = rl_for_multiff_utils.calculate_reward_threshold_for_curriculum_training(
            self.env, n_eval_episodes=num_eval_episodes, ff_caught_rate_threshold=0.1)
        self.regular_training(eval_eps_freq=eval_eps_freq, num_eval_episodes=num_eval_episodes,
                              reward_threshold_to_stop_on=reward_threshold, dir_name=self.best_model_after_curriculum_dir_name,
                              env_params_to_save=self.env_kwargs_for_curriculum_training)
        print('reward_threshold:', reward_threshold)

    def _fill_up_replay_buffer_for_best_model_after_curriculum_training(self, eval_eps_freq):
        self.make_env(**self.env_kwargs)
        self._make_env_suitable_for_curriculum_training()
        self.load_best_model_after_curriculum(load_replay_buffer=True)
        self.regular_training(num_train_episodes=int(self.agent_params['replay_buffer_size'] * 1.3), eval_eps_freq=eval_eps_freq, num_eval_episodes=1, dir_name=self.best_model_after_curriculum_dir_name,
                              env_params_to_save=self.env_kwargs_for_curriculum_training)
        # # save replay buffer only, since the best model has been saved already
        # LSTM_functions.save_replay_buffer(self.sac_model, self.best_model_after_curriculum_dir_name)

    def _further_process_best_model_after_curriculum_training(self, eval_eps_freq=20):
        # to ensure that we get the monitor file at the correct place
        self.save_best_model_after_curriculum()
        # _fill_up_replay_buffer_for_best_model_after_curriculum_training(eval_eps_freq)

        # need to remake env to ensure the monitor_dir is in the folder for the specific agent
        self.make_env(**self.env_kwargs)
        self.load_best_model_after_curriculum(load_replay_buffer=True)
        self._restore_env_params_after_curriculum_training()
        flash_on_interval = self.env_kwargs['flash_on_interval']
        self._update_env_flash_on_interval(flash_on_interval=flash_on_interval)
        # restore reward boundary to final target
        if 'reward_boundary' in self.env_kwargs:
            self._update_env_reward_boundary(reward_boundary=self.env_kwargs['reward_boundary'])
        self.env.distance2center_cost = 0
        self.env_kwargs_for_curriculum_training['distance2center_cost'] = 0

    def _restore_env_params_after_curriculum_training(self):

        self._update_env_dt(self, dt=self.env_kwargs['dt'])
        self.env.dv_cost_factor = self.env_kwargs['dv_cost_factor']
        self.env.dw_cost_factor = self.env_kwargs['dw_cost_factor']
        self.env.w_cost_factor = self.env_kwargs['w_cost_factor']
        self.env.distance2center_cost = 0
        # keep reward boundary consistent
        if 'reward_boundary' in self.env_kwargs:
            self._update_env_reward_boundary(reward_boundary=self.env_kwargs['reward_boundary'])

    def _run_current_agent_after_curriculum_training(self):

        # Now, the condition has restored to the original condition. We shall store the trained agent in the current condition.
        self.regular_training()
        self.successful_training = True

    def regular_training(self, num_train_episodes=10000, eval_eps_freq=15, num_eval_episodes=2,
                         print_episode_reward=True, reward_threshold_to_stop_on=None, dir_name=None,
                         env_params_to_save=None):
        max_steps_per_eps = self.env.episode_len

        if dir_name is not None:
            if env_params_to_save is None:
                env_params_to_save = self.env_kwargs
                self.store_env_params(
                    model_folder_name=dir_name, env_params_to_save=env_params_to_save)

        self.sac_model, self.best_avg_reward_record, self.alpha_df = LSTM_functions.train_LSTM_agent(self.env, self.sac_model,
                                                                                                     num_train_episodes=num_train_episodes, eval_eps_freq=eval_eps_freq, max_steps_per_eps=max_steps_per_eps,
                                                                                                     num_eval_episodes=num_eval_episodes, print_episode_reward=print_episode_reward,
                                                                                                     reward_threshold_to_stop_on=reward_threshold_to_stop_on, dir_name=dir_name)
        return self.best_avg_reward_record

    def test_agent(self, num_test_episode=5, max_steps_per_eps=1024, deterministic=True):
        self.env = self.env_class(**self.env_kwargs)
        self.load_agent(load_replay_buffer=False)
        reward_rate = LSTM_functions.evaluate_agent(self.env_for_test, self.sac_model, max_steps_per_eps, num_test_episode,
                                                    deterministic=deterministic)
        print("Average reward per episode: ", reward_rate)
        return reward_rate

    def save_agent(self, whether_save_replay_buffer=True, dir_name=None):

        if dir_name is None:
            dir_name = self.model_folder_name
        LSTM_functions.save_best_model(
            self.sac_model, whether_save_replay_buffer=whether_save_replay_buffer, dir_name=dir_name)

    def load_agent(self, load_replay_buffer=True, dir_name=None, model_name=''):
        # model_name is not really used here, but put here to be consistent with the SB3 version
        try:
            if dir_name is None:
                dir_name = self.model_folder_name

            self.sac_model.load_model(dir_name)

            if load_replay_buffer:
                buffer_path = os.path.join(dir_name, 'buffer.pkl')
                with open(buffer_path, 'rb') as f:
                    self.replay_buffer = CPU_Unpickler(f).load()
                print(
                    f'length of replay buffer: {len(self.sac_model.replay_buffer)}')

        except Exception as e:      
            print(
                f"There's a problem retrieving existing agent or replay_buffer found in {dir_name}. Need to train a new agent. Error message {e}")
            raise ValueError(
                f"There's a problem retrieving existing agent or replay_buffer found in {dir_name}. Need to train a new agent.")

    def save_best_model_after_curriculum(self, whether_save_replay_buffer=True):
        self.save_agent(whether_save_replay_buffer=whether_save_replay_buffer,
                        dir_name=self.best_model_after_curriculum_dir_name)

    def load_best_model_after_curriculum(self, load_replay_buffer=True):
        self.make_agent()
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=self.best_model_after_curriculum_dir_name)

    def make_animation(self, **kwargs):
        super().make_animation(max_num_frames=None, **kwargs)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
