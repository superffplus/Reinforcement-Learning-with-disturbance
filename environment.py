import math

from numpy.random import default_rng
import pandas as pd
import torch
import gym
from gym import spaces


global rng


def set_rand_seed(seed=0):
    global rng
    rng = default_rng(seed)


class Device:
    """
    Define the device class. It should contain the following functions:\n
    * Generate tasks randomly
    * Record the finished tasks information including amounts and delay
    """
    def __init__(self, id, pos_x, pos_y, random_threshold, size, delay):
        self.id = id
        self.delay = delay
        self.rand_threshold = random_threshold  # generate task if the random generated number is less than it

        self.x = pos_x
        self.y = pos_y
        self.size = size

        self.total_tasks = 0   # total tasks
        self.finished_tasks_amount = 0   # normally finished tasks
        self.finished_delay = 0
        self.rejected_tasks_amount = 0    # tasks which are not accepted by servers
        self.unfinished_tasks_amount = 0   # tasks finished out of time

    def generate(self):
        global rng
        if rng.random() > self.rand_threshold:
            return None
        # need to test the degree of dispersion to verify whether the variance is appropriate.
        rand_size = rng.gamma(2, 2, 1)[0] * self.size
        self.total_tasks += 1
        task = Task(rand_size, self, self.delay)
        return task


class Task:
    def __init__(self, size, source: Device, delay):
        self.size = size
        self.remain_size = size   # remaining task size need to be executed
        self.source = source
        self.delay_constraint = delay

    def execute(self, frequency, time_slot):
        """
        Execute the task, check the state whether it is done or out of time\n
        :param frequency: frequency of server
        :param time_slot: time_slot
        :return: if done, return True, else False
        """
        self.remain_size -= frequency * time_slot
        return True if self.remain_size <= 0 else False


class Buffer:
    """
    Task Buffer in servers
    """
    def __init__(self, length):
        self.length = length
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, task: Task):
        if len(self.buffer) >= self.length:
            return False
        else:
            self.buffer.append(task)
            return True

    def execute(self, frequency, time_slot):
        """
        Execute the task and check the state of the task in the top buffer
        :param frequency: CPU frequency
        :param time_slot: time slot
        :return: task if the task in top buffer is done or out of time, else None
        """
        if len(self.buffer) > 0:
            task = self.buffer[0]
            done = task.execute(frequency, time_slot)

            # pop the task if it is finished or out of time
            if done:
                return self.buffer.pop(0)
        return None


class Server:
    def __init__(self, frequency, queue_len, pos_x, pos_y, time_slot=1, trans_coeff=0.01):
        self.f = frequency
        self.queue = Buffer(queue_len)
        self.x, self.y = pos_x, pos_y
        self.time_slot = time_slot
        self.remaining_buffer_time = 0   # record time to process remaining tasks in the buffer
        self.trans_coeff = trans_coeff

    def get_queue_state(self):
        return self.queue.length - len(self.queue)

    def accept_task(self, task: Task):
        """
        Accept tasks from devices.\n
        :param task: task from device
        :return: If the buffer in the server is not full, return None and execution time, else task and 0
        """
        state = self.queue.push(task)
        if not state:
            return task, 0
        self.remaining_buffer_time += math.ceil(task.size/self.f)

        # computing the transpose delay
        distance = (self.x - task.source.x) ** 2 + (self.y - task.source.y) ** 2
        trans_delay = distance ** 0.5 * self.trans_coeff

        # here, the remaining_buffer_time actually is the execution time
        return None, self.remaining_buffer_time + trans_delay

    def execute(self):
        """
        Execute tasks in the buffer\n
        :return: If the task in the top of the buffer is finished or out of time, return task,
        else return None
        """
        task = self.queue.execute(self.f, self.time_slot)
        self.remaining_buffer_time -= self.time_slot
        return task


class ServerEnv(gym.Env):
    def __init__(self, server_num, device_num, x_range, y_range, mode='all', single_strategy='random',
                 max_frequency=1000, max_task_size=100, max_queue=10):
        self.server_num = server_num
        self.device_num = device_num
        self.x_range = x_range
        self.y_range = y_range
        self.max_frequency = max_frequency
        self.max_task_size = max_task_size
        self.max_queue = max_queue

        self.observation_space = (server_num, 4)
        self.action_space = server_num

        # mode = all means all devices use the RL based decision making
        # mode = single means only the single device use the RL based decision making
        self.mode = mode

        # only be used in mode=single
        # single strategy can be 'random' which means others devices offload task randomly,
        # and it can also be 'nearest' which means other devices offload task to the nearest servers.
        self.single_strategy = single_strategy

        self.servers, self.devices, self.listen_queue = None, None, None

        # variable to record the servers' states, including [frequencies, remaining buffer size, location_x, location_y]
        self.server_indexes = [n for n in range(self.server_num)]
        self.server_status = pd.DataFrame(columns=['frequency', 'remain_buffer', 'pos_x', 'pos_y'],
                                          index=self.server_indexes, dtype=float)
        # variable to record the tasks' states, including [size, delay, location_x, location_y]
        # if mode='single', record the task information of the single device only
        self.task_indexes = [n for n in range(self.device_num) if self.mode == 'all' or n == 0]
        self.task_status = pd.DataFrame(columns=['size', 'delay', 'pos_x', 'pos_y'], index=self.task_indexes,
                                        dtype=float)
        self.generated_tasks = []  # store the generated tasks temporarily
        # store the nearest servers of each device. this will only be used where mode='single'
        self.nearest_servers_list = []

    def reset(self):
        global rng
        self.servers = []
        self.devices = []

        frequencies = (rng.random(self.server_num) + 0.1) * self.max_frequency
        queue_lens = (rng.random(self.server_num) + 0.3) * self.max_queue
        servers_x_pos = rng.uniform(-self.x_range, self.x_range, self.server_num)
        servers_y_pos = rng.uniform(-self.y_range, self.y_range, self.server_num)
        # generate servers
        for i in range(self.server_num):
            frequency, server_x_pos, server_y_pos = frequencies[i], servers_x_pos[i], servers_y_pos[i]
            queue_len = math.ceil(queue_lens[i])
            # frequency = random() * self.max_frequency + 0.1 * self.max_frequency
            # queue_len = int(random() * self.max_queue + 0.1 * self.max_queue)
            # x_pos = rd.uniform(-self.x_range, self.x_range)
            # y_pos = rd.uniform(-self.y_range, self.y_range)
            self.servers.append(Server(frequency, queue_len, server_x_pos, server_y_pos))
            self.server_status.iloc[i]['frequency'] = frequency
            self.server_status.iloc[i]['remain_buffer'] = queue_len
            self.server_status.iloc[i]['pos_x'] = server_x_pos
            self.server_status.iloc[i]['pos_y'] = server_y_pos

        thresholds = rng.uniform(0.15, 0.45, self.device_num)
        delays = rng.random(self.device_num) * 100 + 30
        devices_x_pos = rng.uniform(-self.x_range, self.x_range, self.device_num)
        devices_y_pos = rng.uniform(-self.y_range, self.y_range, self.device_num)
        # generate devices
        for i in range(self.device_num):
            threshold, delay, device_x_pos, device_y_pos = thresholds[i], delays[i], devices_x_pos[i], devices_y_pos[i]
            # delay = random() * 10 + 1
            # device_pos_x = rd.uniform(-self.x_range, self.x_range)
            # device_pos_y = rd.uniform(-self.y_range, self.y_range)
            new_device = Device(i, device_x_pos, device_y_pos, threshold, self.max_task_size, delay)
            new_task = new_device.generate()
            self.devices.append(new_device)
            self.generated_tasks.append(new_task)
            if new_task is not None:
                self.task_status.iloc[i]['size'] = new_task.size
                self.task_status.iloc[i]['delay'] = new_task.delay_constraint
                self.task_status.iloc[i]['pos_x'] = new_task.source.x
                self.task_status.iloc[i]['pos_y'] = new_task.source.y

        reward = torch.zeros(self.device_num)
        self.nearest_servers_list = self.get_neatest_server()

        if self.mode == 'all':
            return [self.server_status, self.task_status], reward
        elif self.mode == 'single':
            return [self.server_status, self.task_status.iloc[0]], reward

    def step(self, action):
        if self.mode == 'all':
            return self.run_all_devices(action)
        else:
            return self.run_single_device(action)

    def run_all_devices(self, actions):
        """
        Execute the decision from all devices, and return the state of the environment.\n
        :param actions: decision from all devices. If the corresponding task is None, the decision will be ignored.
        :return: observation of the environment
        """
        # process the tasks generated in the last time slot with the decisions
        fail_assign_indicator = [0 for _ in range(self.device_num)]
        reward = torch.zeros(self.device_num)
        for i in range(self.device_num):
            existing_task = self.generated_tasks[i]
            if existing_task is None:
                continue
            self.check_server_buffer_state(fail_assign_indicator, existing_task, i)
            tasks_rejected_status, execution_time = self.servers[actions[i]].accept_task(existing_task)
            if not tasks_rejected_status:
                # task is accepted
                if execution_time > existing_task.delay_constraint:
                    # reward of unfinished and rejected tasks should be different to identify
                    # change this value should also modify the test function of main.py
                    reward[i] = -1.5
                    existing_task.source.unfinished_tasks_amount += 1
                else:
                    reward[i] = 1 + (existing_task.delay_constraint - execution_time) / existing_task.delay_constraint * 3
                    existing_task.source.finished_tasks_amount += 1
                    existing_task.source.finished_delay += execution_time
            else:
                existing_task.source.rejected_tasks_amount += 1
                reward[i] = -2

        # run the servers and generate new tasks
        for i in range(self.server_num):
            working_server = self.servers[i]
            working_server.execute()
            self.server_status.iloc[i]['remain_buffer'] = working_server.get_queue_state()

        # reset the generated tasks list
        self.task_status = pd.DataFrame(columns=['size', 'delay', 'pos_x', 'pos_y'], index=self.task_indexes)
        self.generated_tasks = []
        for i in range(self.device_num):
            task = self.devices[i].generate()
            self.generated_tasks.append(task)
            if task is not None:
                if i == 0 or self.mode == 'all':
                    self.task_status.iloc[i]['size'] = task.size
                    self.task_status.iloc[i]['delay'] = task.delay_constraint
                    self.task_status.iloc[i]['pos_x'] = task.source.x
                    self.task_status.iloc[i]['pos_y'] = task.source.y

        return [self.server_status, self.task_status], reward, fail_assign_indicator

    def run_single_device(self, action):
        """
        Execute the decision from a single device (default 0th device), and return the state of the environment.\n
        :param action: action from the single device
        :return: observation of the environment, and the reward
        """
        global rng
        task_reject_state, execution_time, reward = None, 0, 0
        for i in range(self.device_num):
            task = self.generated_tasks[i]
            if task is None:
                continue
            if i == 0:
                task_reject_state, execution_time = self.servers[action].accept_task(task)
            elif self.single_strategy == 'random':
                # task_reject_state, execution_time = self.servers[rd.randint(0, self.server_num)].accept_task(task)
                task_reject_state, execution_time = self.servers[rng.integers(self.server_num)].accept_task(task)
            elif self.single_strategy == 'nearest':
                task_reject_state, execution_time = self.servers[self.nearest_servers_list[i]].accept_task(task)

            if not task_reject_state:
                if execution_time > task.delay_constraint:
                    reward = -1
                    task.source.unfinished_tasks_amount += 1
                else:
                    reward = (task.delay_constraint - execution_time) / task.delay_constraint * 3
                    task.source.finished_tasks_amount += 1
                    task.source.finished_delay += execution_time
            else:
                reward = -1
                task.source.rejected_tasks_amount += 1

        # execute
        for i in range(self.server_num):
            working_server = self.servers[i]
            working_server.execute()
            self.server_status.iloc[i]['remain_buffer'] = working_server.get_queue_state()

        self.generated_tasks = []
        self.task_status = pd.DataFrame(columns=['size', 'delay', 'pos_x', 'pos_y'], index=self.task_indexes)
        for i in range(self.device_num):
            new_task = self.devices[i].generate()
            self.generated_tasks.append(new_task)
            if new_task is not None:
                if i == 0:
                    self.task_status.iloc[i]['size'] = new_task.size
                    self.task_status.iloc[i]['delay'] = new_task.delay_constraint

        return [self.server_status, self.task_status], reward

    def get_neatest_server(self):
        nearest_servers = []
        for i in range(self.device_num):
            n_server, n_server_idx = self.servers[0], 0
            pos_x, pos_y = self.devices[i].x, self.devices[i].y
            short_distance = (pos_x - n_server.x) ** 2 + (pos_y - n_server.y) ** 2
            for j in range(self.server_num):
                distance = (pos_x - self.servers[j].x) ** 2 + (pos_y - self.servers[j].y) ** 2
                if distance < short_distance:
                    short_distance = distance
                    n_server_idx = j
            nearest_servers.append(n_server_idx)
        return nearest_servers

    def check_server_buffer_state(self, pre_rejected_indicator, task: Task, device_index):
        rejected_flag = True
        for i in range(self.server_num):
            exec_time = math.ceil(task.size / self.servers[i].f) + self.servers[i].remaining_buffer_time
            if len(self.servers[i].queue) > 0 and exec_time < task.delay_constraint:
                rejected_flag = False
                break
        pre_rejected_indicator[device_index] = int(rejected_flag)
