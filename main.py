import sys
import argparse
import random
import logging

import torch
import pandas as pd
import numpy as np

import environment
from environment import set_rand_seed
from agent import AdvantageACAgent, EligibilityTraceA2CAgent, PPOAgent, TRPOAgent
from rollouts import Replay

parser = argparse.ArgumentParser(description='provide necessary information for the process')
parser.add_argument('-algo', default='PPO', choices=['A2C', 'EAC', 'PPO', 'TRPO'])
parser.add_argument('-env-mode', default='single', choices=['single', 'all'],
                    help='decide all devices or a single device use the RL based method to make the decision')
parser.add_argument('-strategy', default='nearest', choices=['nearest', 'random'],
                    help='decide offloading strategies of other devices. this will only be used when mode=single')
parser.add_argument('-server-num', default=4, type=int, help='amount of servers')
parser.add_argument('-device-num', default=10, type=int, help='amount of devices')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'cuda'], help='run on cpu or gpu')
parser.add_argument('-decay', default=1e-4, type=float, help='decay of the learning rate')
parser.add_argument('-gamma', type=float, help='coefficient gamma')
parser.add_argument('-alpha', type=float, help='coefficient alpha')
parser.add_argument('-plambda', type=float, help='coefficient lambda')
parser.add_argument('-log', default='INFO', choices=['DEBUG', 'INFO', 'WARNING'])
parser.add_argument('-f', '--filename')
args = parser.parse_args()

# parser the command line parameters
# necessary information for process
algorithm, env_mode, strategy, on_device = args.algo, args.env_mode, args.strategy, args.device
# necessary information for environment
server_num, device_num = args.server_num, args.device_num
# necessary information for algorithm
lr_decay, gamma, alpha, p_lambda = args.decay, args.gamma, args.alpha, args.plambda

# logging module
log_level, filename = args.log, args.filename
numerical_level = getattr(logging, log_level.upper())
if filename is None:
    logging.basicConfig(level=numerical_level, stream=sys.stdout,
                        format='%(asctime)s [%(levelname)s] %(message)s')
else:
    logging.basicConfig(filename=filename, level=numerical_level)

# set parameters
# parameters of environment
env_x_range = 100
env_y_range = 100
env_max_frequency = 400
env_max_task_size = 100
env_max_buffer_size = 20
env_rand_seed = 0

# parameters of network
train_epoch = 10000
single_train_epoch = 40
net_act_hidden = 10
net_value_hidden_1 = 15
net_value_hidden_2 = 7

total_test_epoch = 20

model_dict = {'A2C': AdvantageACAgent, 'EAC': EligibilityTraceA2CAgent, 'PPO': PPOAgent, 'TRPO': TRPOAgent}
rl_model_class = model_dict[algorithm]
net_params = []
if algorithm == 'A2C':
    net_params = [gamma, alpha]
elif algorithm in ['EAC', 'PPO']:
    net_params = [gamma, alpha, p_lambda]
elif algorithm == 'TRPO':
    net_params = []

set_rand_seed(env_rand_seed)
env = environment.ServerEnv(server_num, device_num, env_x_range, env_y_range, mode='all', single_strategy=strategy,
                            max_frequency=env_max_frequency, max_task_size=env_max_task_size,
                            max_queue=env_max_buffer_size)

ob_space_num = 4 * server_num + 4

torch.manual_seed(env_rand_seed)
np.random.seed(env_rand_seed)
if on_device == 'cuda':
    torch.cuda.manual_seed(env_rand_seed)
    torch.cuda.manual_seed_all(env_rand_seed)


def all_mode():
    agent_list, replay_list = [], []
    for i in range(device_num):
        rl_model = rl_model_class(ob_space_num, server_num, net_params, device=on_device)
        agent_list.append(rl_model)
        experience_reply = Replay(ob_space_num, server_num, on_device)
        replay_list.append(experience_reply)

    for epoch in range(train_epoch):
        logging.debug('preparing trace data')
        ob, reward = env.reset()
        last_task_tensors = [torch.zeros(1, 4).to(on_device) for _ in range(device_num)]
        for mini_step in range(single_train_epoch):
            server_state_dataframe, task_state_dataframe = ob
            decision_tensor = torch.ones(device_num, dtype=torch.int64) * -1
            for i in range(device_num):
                server_state_tensor = torch.as_tensor(server_state_dataframe.values, dtype=torch.float32,
                                                      device=on_device)
                if not task_state_dataframe.iloc[i].isnull().any():
                    task_state_tensor = torch.as_tensor(task_state_dataframe.iloc[i], dtype=torch.float32,
                                                        device=on_device).view(-1, 4)
                    last_task_tensors[i] = task_state_tensor
                    environment_tensor = torch.cat((server_state_tensor, task_state_tensor), dim=0).view(-1,
                        4 * server_num + 4)
                    action, action_probs = agent_list[i].step(environment_tensor)
                    decision_tensor[i] = action
                else:
                    environment_tensor = torch.cat((server_state_tensor, last_task_tensors[i]), dim=0).view(-1,
                        4 * server_num + 4)
                    action_probs = torch.zeros(device_num)
                pred_value = agent_list[i].value_model(environment_tensor)
                replay_list[i].append(environment_tensor, decision_tensor[i], reward[i], False, action_probs, pred_value)
            ob, reward, _ = env.step(decision_tensor)

        logging.debug('trace data is ready, start to learn')
        # update the model
        for i in range(device_num):
            replay_list[i].receive_trajectory_data(0.99)
            # prevent overfit due to short of data
            # if epoch < 10:
            #     continue
            agent_list[i].learn(replay_list[i])
        logging.debug('learning is ready, turn to test or next turn')
        if (epoch % 20) == 0:
            test_all_agent(agent_list, epoch)


def single_mode():
    agent = rl_model_class(4*server_num+4, server_num, net_params, device=on_device)
    replay = Replay(ob_space_num, server_num, on_device)

    for epoch in range(train_epoch):
        ob, reward = env.reset()
        for mini_step in range(single_train_epoch):
            server_state, task_state = ob
            decision_tensor = torch.tensor(-1)
            pred_value, action_probs = torch.zeros(1), torch.zeros(server_num)
            server_state_tensor = torch.as_tensor(server_state, dtype=torch.float32, device=on_device)
            task_state_tensor = torch.as_tensor(task_state, dtype=torch.float32, device=on_device)
            environment_tensor = torch.cat((server_state_tensor, task_state_tensor), dim=1)
            if not task_state['size'].isnull().any():
                action, action_probs = agent.step(environment_tensor)
                pred_value = agent.value_model(environment_tensor)
                decision_tensor = action
            replay.append(environment_tensor, decision_tensor, reward, False, action_probs, pred_value)

            ob, reward = env.step(decision_tensor)

        replay.clean_data()
        agent.learn(replay)
        if (epoch % 20) == 0:
            test_single_agent(agent, epoch)


def test_all_agent(agent_list, train_episode):
    # set critic metric to record the performance of the algorithm
    # 'total', 'finished', 'unfinished', 'rejected' indicate total, finished, unfinished, rejected amount of tasks
    # 'reward' indicates the total reward
    critic_init_tensor = torch.zeros([device_num, 6])
    critic = pd.DataFrame(critic_init_tensor, columns=['total', 'finished', 'unfinished', 'rejected', 'reward',
                                                       'pre_rejected'], index=[i for i in range(device_num)])
    for test_epoch in range(total_test_epoch):
        ob, reward = env.reset()
        for mini_epoch in range(single_train_epoch):
            decisions = torch.ones(device_num,  dtype=torch.int64) * -1
            server_state, task_state = ob
            server_state_tensor = torch.as_tensor(server_state.values, dtype=torch.float32, device=on_device)
            for i in range(device_num):
                if not task_state.iloc[i].isnull().any():
                    task_state_tensor = torch.as_tensor(task_state.iloc[i], dtype=torch.float32, device=on_device).view(-1, 4)
                    environment_tensor = torch.cat((server_state_tensor, task_state_tensor), dim=0).view(-1, 4*server_num+4)
                    action, action_probs = agent_list[i].step(environment_tensor)
                    decisions[i] = action
            os, reward, rejected_info = env.step(decisions)
            for i in range(device_num):
                critic.iloc[i]['total'] += (reward[i] != 0)
                critic.iloc[i]['reward'] += reward[i]
                critic.iloc[i]['pre_rejected'] += rejected_info[i]
                if reward[i] == 0:
                    continue
                elif reward[i] == -1.5:
                    critic.iloc[i]['unfinished'] += 1
                elif reward[i] == -1:
                    critic.iloc[i]['rejected'] += 1
                else:
                    critic.iloc[i]['finished'] += 1

    total_task = critic['total'].sum()
    logging.info('train episode %d, mean reward is %.2f, finished rate is %.2f, unfinished rate is %.2f, rejected '
                 'rate is %.2f, pre_rejected rate is %.2f', train_episode, critic['reward'].sum()/total_task,
                 critic['finished'].sum()/total_task, critic['unfinished'].sum()/total_task,
                 critic['rejected'].sum()/total_task, critic['pre_rejected'].sum()/total_task)


def test_single_agent(agent, train_episode):
    critic_init_tensor = torch.zeros(5)
    critic = pd.DataFrame(critic_init_tensor, columns=['total', 'finished', 'unfinished', 'rejected', 'reward'])
    for test_epoch in range(total_test_epoch):
        ob, reward = env.reset()
        for mini_epoch in range(single_train_epoch):
            server_state, task_state = ob
            action = torch.tensor(-1)
            if not task_state['size'].isnull().any():
                server_state_tensor = torch.as_tensor(server_state, dtype=torch.float32, device=on_device)
                task_state_tensor = torch.as_tensor(task_state, dtype=torch.float32, device=on_device)
                environment_tensor = torch.cat((server_state_tensor, task_state_tensor), dim=1)
                action, action_probs = agent.step(environment_tensor)
                critic['total'] += 1
            ob, reward = env.step(action)
            critic['reward'] += reward
            if reward == 0:
                continue
            elif reward == -1.5:
                critic['unfinished'] += 1
            elif reward == -1:
                critic['rejected'] += 1
            else:
                critic['finished'] += 1

    logging.info('train episode %d, mean reward is %.2f, finished rate is %.2f, unfinished rate is %.2f, rejected '
                 'rate is %.2f', train_episode, critic['reward']/critic['total'], critic['finished']/critic['total'],
                 critic['unfinished']/critic['total'], critic['rejected']/critic['total'])


if env_mode == 'all':
    all_mode()
elif env_mode == 'single':
    single_mode()
