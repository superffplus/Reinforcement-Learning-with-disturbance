import sys
import argparse
import random

import torch
import pandas as pd

import environment
import agent

parser = argparse.ArgumentParser(description='provide necessary information for the process')
parser.add_argument('-algo', default='PPO', choices=['A2C', 'EAC', 'PPO', 'TRPO'])
parser.add_argument('-env-mode', default='single', choices=['single', 'all'],
                    help='decide all devices or a single device use the RL based method to make the decision')
parser.add_argument('-strategy', default='nearest', choices=['nearest', 'random'],
                    help='decide offloading strategies of other devices. this will only be used when mode=single')
parser.add_argument('-server-num', default=4, type=int, help='amount of servers')
parser.add_argument('-device-num', default=10, type=int, help='amount of devices')
parser.add_argument('-d', '--device', default='cpu', choices=['cpu', 'gpu'], help='run on cpu or gpu')
parser.add_argument('-decay', default=1e-4, type=float, help='decay of the learning rate')
parser.add_argument('-gamma', type=float, help='coefficient')
args = parser.parse_args()

# parser the command line parameters
# necessary information for process
algorithm, env_mode, strategy, on_device = args.algo, args.env_mode, args.strategy, args.device
# necessary information for environment
server_num, device_num = args.server_num, args.device_num
# necessary information for algorithm
lr_decay, gamma = args.decay, args.gamma

# set parameters
# parameters of environment
env_x_range = 100
env_y_range = 100
env_max_frequency = 10000
env_max_task_size = 100
env_max_buffer_size = 10
env_rand_seed = 0

# parameters of network
net_act_hidden = 10
net_value_hidden_1 = 15
net_value_hidden_2 = 7
