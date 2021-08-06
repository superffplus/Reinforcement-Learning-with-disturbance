from abc import abstractmethod
import pandas as pd
import numpy as np
import torch
import scipy.signal as signal


class Rollouts:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.end_masks = []
        self.action_prob = []
        self.values = []

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

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.end_masks = []
        self.action_prob = []


class Replay(Rollouts):
    def __init__(self):
        super(Rollouts, self).__init__()
        self.fields = ['state', 'action', 'prob', 'advantage', 'return']
        self.memory = pd.DataFrame(columns=self.fields)

    def store(self, df):
        self.memory = pd.concat([self.memory, df[self.fields]], ignore_index=True)

    def sample(self, batch_size):
        indices = np.random.choice(self.memory.shape[0], size=batch_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.fields)

    def receive_trajectory_data(self, gamma):
        df = pd.DataFrame(np.array([self.states, self.actions, self.end_masks, self.rewards,
                                    self.action_prob, self.values]).reshape(-1, 6),
                          columns=['state', 'action', 'done', 'reward', 'prob', 'value'])
        state_tensor = torch.as_tensor(df['state'], dtype=torch.float32)
        action_tensor = torch.as_tensor(df['action'], dtype=torch.long)
        action_prob_tensor = torch.as_tensor(df['prob'], dtype=torch.float32)
        pi_tensor = torch.gather(action_prob_tensor, 1, action_tensor.unsqueeze(1)).squeeze(1)
        df['log_prob'] = pi_tensor.detach().numpy()
        df['next_value'] = df['value'].shift(-1).fillna(0.)
        df['u'] = df['reward'] + gamma * df['next_value']
        df['delta'] = df['u'] - df['value']

        # This is to compute the discounted advantage function and final return
        # The function is similar to torch.consum() by setting the first and second parameters
        # as [1.,], [1., -gamma]
        # DataFrame[::-1] is to generate the data in reversed order which is similar to the
        # function torch.flip()
        df['advantage'] = signal.lfilter([1.,], [1., -gamma],
                                         df['delta'][::-1])[::-1]
        df['return'] = signal.lfilter([1.,], [1, -gamma],
                                      df['reward'][::-1])[::-1]
        self.store(df)
