
from reinforcement_learning.agents.feedforward import env_for_sb3
from reinforcement_learning.base_classes import rl_base_class, rl_base_utils
from reinforcement_learning.agents.feedforward import sb3_utils
from reinforcement_learning.base_classes import env_utils

import os
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from os.path import exists
import torch.nn as nn
import gc
import math
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class SB3forMultifirefly(rl_base_class._RLforMultifirefly):

    def __init__(self,
                 overall_folder='multiff_analysis/RL_models/SB3_stored_models/all_agents/env1_relu/',
                 add_date_to_model_folder_name=False,
                 **kwargs):

        super().__init__(overall_folder=overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         **kwargs)

        self.agent_type = 'sb3'
        self.monkey_name = None

        self.env_class = env_for_sb3.EnvForSB3

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.class_instance_env_kwargs,
            **self.additional_env_kwargs
        }

    def load_best_model_postcurriculum(self, load_replay_buffer=True):
        dir_name = self.best_model_postcurriculum_dir
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=dir_name)

    def load_best_model_in_curriculum(self, load_replay_buffer=True):
        dir_name = self.best_model_in_curriculum_dir
        self.load_agent(load_replay_buffer=load_replay_buffer,
                        dir_name=dir_name)

    def make_env(self, monitor_dir=None, **env_kwargs):
        super().make_env(**env_kwargs)

        os.makedirs(self.model_folder_name, exist_ok=True)
        if monitor_dir is None:
            monitor_dir = self.model_folder_name
        self.env = Monitor(self.env, monitor_dir)

    def make_agent(self, **kwargs):

        if self.agent_params is None:
            self.agent_params = {'learning_rate': kwargs.get('learning_rate', 0.0003),
                                 'batch_size': kwargs.get('batch_size', 1024),
                                 'target_update_interval': kwargs.get('target_update_interval', 50),
                                 'buffer_size': kwargs.get('buffer_size', 1000000),
                                 'learning_starts': kwargs.get('learning_starts', 10000),
                                 'train_freq': kwargs.get('train_freq', 10),
                                 # 'train_freq': kwargs.get('train_freq', 1,),
                                 'gradient_steps': kwargs.get('gradient_steps', 10),
                                 'ent_coef': kwargs.get('ent_coef', 'auto'),
                                 'policy_kwargs': kwargs.get('policy_kwargs', dict(activation_fn=nn.ReLU, net_arch=[256, 128])),
                                 'gamma': 0.99,
                                 }
        else:
            self.agent_params.update(kwargs)

        # num_nodes = self.env.env.obs_space_length * 2 + 12
        # print('num_nodes in each layer of the neural network:', num_nodes)

        self.buffer_size = self.agent_params['buffer_size']

        self.sac_model = SAC("MlpPolicy",
                             self.env,
                             **self.agent_params)

        self.agent_params = rl_base_utils.get_agent_params_from_the_current_sac_model(
            self.sac_model)
        print('Made agent with the following params:', self.agent_params)

    def regular_training(self, timesteps=2000000, best_model_save_path=None):
        if best_model_save_path is None:
            best_model_save_path = self.model_folder_name

        stop_train_callback = sb3_utils.StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=15, verbose=1, model_folder_name=self.model_folder_name,
                                                                         overall_folder=self.overall_folder, agent_id=self.agent_id)

        # Note: by adding best_model_save_path, the callback can save the best model after each evaluation
        if best_model_save_path is not None:
            os.makedirs(best_model_save_path, exist_ok=True)
        self.callback = EvalCallback(self.env, eval_freq=15000, callback_after_eval=stop_train_callback, verbose=1,
                                     best_model_save_path=best_model_save_path, n_eval_episodes=3)
        self.write_checkpoint_manifest(best_model_save_path)
        self.sac_model.learn(total_timesteps=int(
            timesteps), callback=self.callback)

    def _make_agent_for_curriculum_training(self):
        print('Making agent for curriculum training...')
        self.make_agent(learning_rate=0.0015,
                        train_freq=10,
                        gradient_steps=1)

    def make_initial_env_for_curriculum_training(self, initial_angular_terminal_vel=0.32,
                                                 initial_distance2center_cost=2,
                                                 initial_flash_on_interval=0.3,
                                                 initial_reward_boundary=75):
        monitor_dir = self.best_model_postcurriculum_dir
        os.makedirs(monitor_dir, exist_ok=True)
        print(f'Making initial env for curriculum training...')
        self.make_env(monitor_dir=monitor_dir, **self.input_env_kwargs)
        self._make_initial_env_for_curriculum_training(initial_angular_terminal_vel=initial_angular_terminal_vel,
                                                       initial_flash_on_interval=initial_flash_on_interval,
                                                       initial_distance2center_cost=initial_distance2center_cost,
                                                       initial_reward_boundary=initial_reward_boundary)

    def _train_till_reaching_reward_threshold(self, n_eval_episodes=1, ff_caught_rate_threshold=0.1):
        reward_threshold = rl_base_utils.calculate_reward_threshold_for_curriculum_training(
            self.env.env, n_eval_episodes=n_eval_episodes, ff_caught_rate_threshold=ff_caught_rate_threshold)
        print('reward_threshold:', reward_threshold)
        stop_train_callback = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold)  # or 10ff/250s
        callback = EvalCallback(
            self.env, eval_freq=15000, callback_after_eval=stop_train_callback, verbose=1, n_eval_episodes=n_eval_episodes,
            best_model_save_path=self.best_model_in_curriculum_dir)
        self.write_checkpoint_manifest(self.best_model_in_curriculum_dir)
        self.sac_model.learn(total_timesteps=1000000, callback=callback)
        if callback.best_mean_reward < reward_threshold:
            raise ValueError(
                "The agent is not able to catch enough ff after training for 1e6 timesteps.")

    def _use_while_loop_for_curriculum_training(self):
        while (self.env.env.dt > self.input_env_kwargs['dt']) | (self.env.env.angular_terminal_vel > self.input_env_kwargs['angular_terminal_vel']):
            gc.collect()
            # Note: 0.00222 = 0.0035/(pi/2), same as the monkey's threshold
            try:
                self._train_till_reaching_reward_threshold()
            except ValueError as e:
                print(f"Error message: {e}")
                break
            self._update_env_after_meeting_reward_threshold()

        # After all the conditions are met, train the agent once again to ensure performance (stop training with no improvement)
        # After curriculum ends, copy best from in-curriculum to post-curriculum
        os.makedirs(self.best_model_postcurriculum_dir, exist_ok=True)
        self.make_env(**self.input_env_kwargs)
        self.load_best_model_in_curriculum(load_replay_buffer=True)
        self.regular_training(
            best_model_save_path=self.best_model_postcurriculum_dir)

        # Now, load the best model in post-curriculum folder and save it to the agent folder
        self.load_best_model_postcurriculum(load_replay_buffer=True)

    def save_agent(self, whether_save_replay_buffer=False, dir_name=None):
        model_name = 'best_model'
        if dir_name is None:
            dir_name = self.model_folder_name

        os.makedirs(dir_name, exist_ok=True)
        self.sac_model.save(os.path.join(dir_name, model_name))
        print('Saved agent:', os.path.join(dir_name, model_name))
        if whether_save_replay_buffer:
            self.sac_model.save_replay_buffer(
                os.path.join(dir_name, 'buffer'))  # I added this
            print('Saved replay buffer:', os.path.join(dir_name, 'buffer'))

        self.write_checkpoint_manifest(dir_name)

    def write_checkpoint_manifest(self, dir_name):
        rl_base_utils.write_checkpoint(dir_name, {
            'algorithm': 'sb3_sac',
            'model_file': f'best_model.zip',
            'replay_buffer': 'buffer',
            'num_timesteps': getattr(self.sac_model, 'num_timesteps', None),
            'env_params_path': 'env_params.txt',
            'env_params': self.current_env_kwargs,
        })

    def load_agent(self, load_replay_buffer=True, keep_current_agent_params=True, dir_name=None, model_name='best_model'):
        manifest = rl_base_utils.read_checkpoint_manifest(dir_name)
        model_file = manifest.get('model_file') if isinstance(
            manifest, dict) else None
        path = os.path.join(dir_name, model_file) if model_file else os.path.join(
            dir_name, model_name + '.zip')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        self.make_env(**manifest['env_params'])
        self.make_agent()
        self.sac_model = self.sac_model.load(path, env=self.env)
        print("Loaded existing agent:", path)

        if load_replay_buffer:
            buffer_name = manifest.get('replay_buffer') if isinstance(
                manifest, dict) else 'buffer'
            path2 = os.path.join(dir_name, buffer_name)
            if os.path.exists(path2):
                self.sac_model.load_replay_buffer(path2)
                print("Loaded existing replay buffer:", path2)
            else:
                print(
                    f"Replay buffer not found at {path2}; proceeding without it.")

        if keep_current_agent_params and (self.agent_params is not None):
            for key, item in self.agent_params.items():
                setattr(self.sac_model, key, item)

        print('Params from agent after loading:')
        print(rl_base_utils.get_agent_params_from_the_current_sac_model(
            self.sac_model))
        self.loaded_agent_dir = dir_name
        return
