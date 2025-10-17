

from reinforcement_learning.env_related import env_for_rnn
from reinforcement_learning.agents.feedforward import rl_base_utils, rl_base_class
from reinforcement_learning.agents.rnn import lstm_utils
from reinforcement_learning.agents.rnn import gru_utils
from reinforcement_learning.env_related import env_utils
from reinforcement_learning.agents.rnn import lstm_class

import os
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch
import numpy as np
import pickle
import io
import copy
plt.rcParams["animation.html"] = "html5"
retrieve_buffer = False
n_steps = 1000
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class GRUforMultifirefly(lstm_class.LSTMforMultifirefly):

    def __init__(self,
                 overall_folder='RL_models/GRU_stored_models/all_agents/gen_0/',
                 model_folder_name=None,
                 add_date_to_model_folder_name=False,
                 max_in_memory_time=1,
                 **additional_env_kwargs):

        super().__init__(overall_folder,
                         add_date_to_model_folder_name=add_date_to_model_folder_name,
                         model_folder_name=model_folder_name,
                         max_in_memory_time=max_in_memory_time,
                         **additional_env_kwargs)

        # keep 'rnn' branch behavior for env handling; mark subtype for clarity
        self.sb3_or_rnn = 'rnn'
        self.rnn_type = 'gru'
        self.env_class = env_for_rnn.EnvForRNN

        self.default_env_kwargs = env_utils.get_env_default_kwargs(
            self.env_class)
        self.input_env_kwargs = {
            **self.default_env_kwargs,
            **self.default_input_env_kwargs,
            **self.additional_env_kwargs
        }

    def make_agent(self,
                   agent_params_already_set_ok=True,
                   **kwargs):

        if not hasattr(self, 'env'):
            self.make_env(**self.input_env_kwargs)

        if agent_params_already_set_ok & (self.agent_params is not None):
            existing_agent_params = self.agent_params.copy()
        else:
            existing_agent_params = {}

        self.agent_params = {
            "gamma": rl_base_utils.calculate_model_gamma(self.env.dt),
            "state_space": self.env.observation_space,
            "action_space": self.env.action_space,
            "action_dim": self.env.action_space.shape[0],
            "action_range": 1.0,
            "replay_buffer_size": kwargs.get('replay_buffer_size', 500),
            "hidden_dim": kwargs.get('hidden_dim', 256),
            "soft_q_lr": kwargs.get('soft_q_lr', 0.0015),
            "policy_lr": kwargs.get('policy_lr', 0.003),
            "alpha_lr": kwargs.get('alpha_lr', 0.002),
            "batch_size": kwargs.get('batch_size', 10),
            "update_itr": kwargs.get('update_itr', 1),
            "reward_scale": kwargs.get('reward_scale', 0.5),
            "target_entropy": kwargs.get('target_entropy', - self.env.action_space.shape[0]),
            "soft_tau": kwargs.get('soft_tau', 0.015),
            "train_freq": kwargs.get('train_freq', 5),
        }

        self.agent_params.update(existing_agent_params)
        self.agent_params.update(kwargs)

        # GRU-specific replay buffer and trainer
        self.replay_buffer = gru_utils.ReplayBufferGRU(
            self.agent_params['replay_buffer_size'])
        self.agent_params['replay_buffer'] = self.replay_buffer

        # Construct GRU trainer (expects positional args)
        ap = self.agent_params
        self.sac_model = gru_utils.GRU_SAC_Trainer(
            replay_buffer=self.replay_buffer,
            state_space=ap['state_space'],
            action_space=ap['action_space'],
            hidden_dim=ap['hidden_dim'],
            action_range=ap['action_range'],
            gamma=ap['gamma'],
            soft_q_lr=ap['soft_q_lr'],
            policy_lr=ap['policy_lr'],
            alpha_lr=ap['alpha_lr'],
            batch_size=ap['batch_size'],
            update_itr=ap['update_itr'],
            reward_scale=ap['reward_scale'],
            target_entropy=ap['target_entropy'],
            soft_tau=ap['soft_tau'],
            train_freq=ap['train_freq'],
        )

    def regular_training(self, num_train_episodes=10000, eval_eps_freq=15, num_eval_episodes=2,
                         print_episode_reward=True, reward_threshold_to_stop_on=None, dir_name=None):
        max_steps_per_eps = self.env.episode_len

        self.sac_model.soft_q_net1.train()
        self.sac_model.soft_q_net2.train()
        self.sac_model.policy_net.train()

        self.eval_rewards = []
        self.best_avg_reward = -9999
        self.best_avg_reward_record = -9999

        self.list_of_epi_rewards = []
        for eps in range(num_train_episodes):
            episode_reward = self._train_gru_episode(
                self.env, self.sac_model, max_steps_per_eps)

            self.list_of_epi_rewards.append(episode_reward)

            if eps % eval_eps_freq == 0 and eps > 0:
                avg_reward = self._evaluate_gru_agent(
                    self.env, self.sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True)

                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    if self.best_avg_reward > self.best_avg_reward_record:
                        self.best_avg_reward_record = self.best_avg_reward
                    if dir_name is not None:
                        self.save_agent(whether_save_replay_buffer=True, dir_name=dir_name)
                        self.write_checkpoint_manifest(dir_name)

                    if reward_threshold_to_stop_on is not None:
                        if self.best_avg_reward_record >= reward_threshold_to_stop_on:
                            break

                self.eval_rewards.append(avg_reward)
                if len(self.eval_rewards) > 100:
                    self.eval_rewards = self.eval_rewards[-100:]

            if print_episode_reward:
                print(f'Episode: {eps}, Episode Reward: {episode_reward}')
                print('============================================================')

        # Keep API parity with LSTM path
        alpha_df = pd.DataFrame({'rewards': self.list_of_epi_rewards})
        return self.sac_model, self.best_avg_reward_record, alpha_df

    def test_agent(self, num_test_episode=5, max_steps_per_eps=512, deterministic=True):
        self.load_latest_agent(load_replay_buffer=False)
        test_env = getattr(self, 'env_for_test', None)
        if test_env is None:
            test_env = self.env
        reward_rate = self._evaluate_gru_agent(test_env, self.sac_model, max_steps_per_eps, num_test_episode,
                                               deterministic=deterministic)
        print("Average reward per episode: ", reward_rate)
        return reward_rate

    def save_agent(self, whether_save_replay_buffer=True, dir_name=None):
        if dir_name is None:
            dir_name = self.model_folder_name
        os.makedirs(dir_name, exist_ok=True)
        self.sac_model.save_model(dir_name)
        self.write_checkpoint_manifest(dir_name)

        if whether_save_replay_buffer:
            buffer_path = os.path.join(dir_name, 'buffer.pkl')
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.sac_model.replay_buffer.buffer, f)

    def write_checkpoint_manifest(self, dir_name):
        rl_base_utils.write_checkpoint(dir_name, {
            'algorithm': 'gru_sac',
            'model_files': ['GRU_q1', 'GRU_q2', 'GRU_policy'],
            'replay_buffer': 'buffer.pkl',
            'env_params': self.current_env_kwargs,
        })

    def load_agent(self, load_replay_buffer=True, dir_name=None):
        manifest = rl_base_utils.read_checkpoint_manifest(dir_name)
        self.make_env(**manifest['env_params'])
        self.make_agent()
        self.sac_model.load_model(dir_name)

        if load_replay_buffer:
            buffer_name = 'buffer.pkl'
            if isinstance(manifest, dict):
                buffer_name = manifest.get('replay_buffer', buffer_name)
            buffer_path = os.path.join(dir_name, buffer_name)
            with open(buffer_path, 'rb') as f:
                loaded_buffer_list = lstm_class.CPU_Unpickler(f).load()

            if isinstance(loaded_buffer_list, list):
                capacity = max(self.agent_params.get(
                    'replay_buffer_size', 500), len(loaded_buffer_list))
                rb = gru_utils.ReplayBufferGRU(capacity)
                rb.buffer = loaded_buffer_list
                rb.position = len(loaded_buffer_list) % capacity
                self.replay_buffer = rb
            else:
                self.replay_buffer = loaded_buffer_list

            self.sac_model.replay_buffer = self.replay_buffer
            print(f'length of replay buffer: {len(self.sac_model.replay_buffer)}')

        print("Loaded existing agent:", dir_name)
        self.loaded_agent_dir = dir_name
        return

    @staticmethod
    def _initialize_gru_hidden(hidden_dim, device):
        return torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device)

    def _train_gru_episode(self, env, sac_model, max_steps_per_eps):
        state, _ = env.reset()
        last_action = env.action_space.sample()
        episode_data = {
            'state': [], 'action': [], 'last_action': [], 'reward': [],
            'next_state': [], 'done': []
        }
        # GRU hidden is a single tensor
        hidden_out = self._initialize_gru_hidden(sac_model.hidden_dim, gru_utils.device)
        ini_hidden_in, ini_hidden_out = None, None

        for step in range(max_steps_per_eps):
            hidden_in = hidden_out
            action, hidden_out = sac_model.policy_net.get_action(
                state, last_action, hidden_in, deterministic=False)
            next_state, reward, done, _, _ = env.step(action)

            if step == 0:
                ini_hidden_in, ini_hidden_out = hidden_in, hidden_out

            episode_data['state'].append(state)
            episode_data['action'].append(action)
            episode_data['last_action'].append(last_action)
            episode_data['reward'].append(reward)
            episode_data['next_state'].append(next_state)
            episode_data['done'].append(done)

            state, last_action = next_state, action

            if step % sac_model.train_freq == 0 and len(sac_model.replay_buffer) > sac_model.batch_size:
                for _ in range(sac_model.update_itr):
                    sac_model.update(batch_size=sac_model.batch_size,
                                     reward_scale=sac_model.reward_scale,
                                     auto_entropy=True,
                                     target_entropy=sac_model.target_entropy,
                                     gamma=sac_model.gamma,
                                     soft_tau=sac_model.soft_tau)

            if done:
                break

        sac_model.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_data['state'], episode_data['action'],
                                     episode_data['last_action'], episode_data['reward'], episode_data['next_state'], episode_data['done'])
        return np.sum(episode_data['reward'])

    def _evaluate_gru_agent(self, env, sac_model, max_steps_per_eps, num_eval_episodes, deterministic=True):
        was_training = sac_model.policy_net.training
        sac_model.policy_net.eval()
        cum_reward = 0
        with torch.no_grad():
            for _ in range(num_eval_episodes):
                state, _ = env.reset()
                last_action = env.action_space.sample()
                hidden_out = self._initialize_gru_hidden(sac_model.hidden_dim, gru_utils.device)
                for _ in range(max_steps_per_eps):
                    hidden_in = hidden_out
                    action, hidden_out = sac_model.policy_net.get_action(
                        state, last_action, hidden_in, deterministic=deterministic)
                    next_state, reward, done, _, _ = env.step(action)
                    cum_reward += reward
                    state, last_action = next_state, action
                    if done:
                        break
        if was_training:
            sac_model.policy_net.train()
        return cum_reward / num_eval_episodes

