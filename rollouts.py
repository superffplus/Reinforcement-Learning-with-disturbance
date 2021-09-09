import numpy as np
import pandas as pd
import scipy.signal as signal
import torch


class Replay:
    def __init__(self, ob_space, act_space, device):
        self.fields = ['state', 'action', 'reward', 'end', 'action_prob', 'value', 'advantage', 'return', 'next_value']
        self.memory = pd.DataFrame(columns=self.fields)
        self.states = []
        self.actions = []
        self.rewards = []
        self.end_masks = []
        self.action_prob = []
        self.values = []

        self.device = device
        self.ob_space = ob_space
        self.act_space = act_space

    def append(self, state, action, reward, done, action_prob=None, value=None):
        """
        Append the trajectory data to the rollout. For the PPO algorithm, it should
        provide action probability and value prediction.\n
        :param state: observation
        :param action: action that the agent takes
        :param reward: reward got from the environment
        :param done: the environment state
        :param action_prob: action probability distribution given by the actor model
        :param value: value prediction given by the value model
        :return:
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.end_masks.append(done)
        if action_prob is not None:
            self.action_prob.append(action_prob)
            self.values.append(value)

    def sample(self, batch_size):
        indices = np.random.choice(self.memory.shape[0], size=batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.fields)

    # def sample(self, batch_size):
    #     indices = np.random.choice(len(self.states), size=batch_size)
    #     return [(self.states[i], self.actions[i], self.rewards[i], self.end_masks[i], self.action_prob[i],
    #              self.values[i], self.advantages[i], self.returns[i]) for i in indices]

    def receive_trajectory_data(self, gamma):
        trajectory_data = pd.DataFrame(columns=['state', 'action', 'reward', 'end', 'action_prob', 'value', 'next_value'])
        for i, action in enumerate(self.actions[:-1]):
            if action > -1:
                state_numpy = self.states[i].view(-1).cpu().numpy()
                action_numpy = self.actions[i].cpu().numpy()
                reward_numpy = self.rewards[i+1].cpu().numpy()
                end_mask_numpy = self.end_masks[i]
                action_prob_numpy = self.action_prob[i].view(-1).detach().cpu().numpy()
                value_numpy = self.values[i].view(-1).detach().cpu().numpy()
                next_value_numpy = self.values[i+1].view(-1).detach().cpu().numpy()
                df = pd.DataFrame([[state_numpy, action_numpy, reward_numpy, end_mask_numpy, action_prob_numpy,
                                    value_numpy, next_value_numpy]], columns=['state', 'action', 'reward', 'end',
                                                                              'action_prob', 'value', 'next_value'])
                trajectory_data = pd.concat([trajectory_data, df])

        trajectory_data['u'] = trajectory_data['reward'] + gamma * trajectory_data['next_value']
        trajectory_data['delta'] = trajectory_data['u'] - trajectory_data['value']
        trajectory_data['advantage'] = signal.lfilter([1., ], [1., -gamma],
                                                      trajectory_data['delta'][::-1])[::-1]
        trajectory_data['return'] = signal.lfilter([1., ], [1., -gamma],
                                                   trajectory_data['reward'][::-1])[::-1]
        self.store(trajectory_data)

    def store(self, df: pd.DataFrame):
        self.memory = pd.concat([self.memory, df[self.fields]], ignore_index=True)
        self.states = []
        self.actions = []
        self.rewards = []
        self.end_masks = []
        self.action_prob = []
        self.values = []
        torch.cuda.empty_cache()
