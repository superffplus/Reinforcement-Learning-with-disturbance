import random
import queue
import math

import pandas as pd
import torch
import gym


class Device:
    """
    Define the device class. It should contain the following functions:\n
    * Generate tasks randomly
    * Record the finished tasks information including amounts and delay
    """
    def __init__(self, id, max_x, max_y, random_threshold, max_size, delay, random_seed=0):
        self.id = id
        self.delay = delay
        self.rand = random
        self.rand.seed(random_seed)
        self.rand_threshold = random_threshold  # generate task if the random generated number is less than it

        self.x = random.uniform(-max_x, max_x)
        self.y = random.uniform(-max_y, max_y)
        self.size = random.random() * max_size + 0.1 * max_size

        self.total_tasks = 0   # total tasks
        self.finished_tasks_amount = 0   # normally finished tasks
        self.finished_delay = 0
        self.rejected_tasks_amount = 0    # tasks which are not accepted by servers
        self.unfinished_tasks_amount = 0   # tasks finished out of time

    def generate(self):
        if self.rand.random() > self.rand_threshold:
            return None
        # need to test the degree of dispersion to verify whether the variance is appropriate.
        rand_size = random.normalvariate(self.size, 1)
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
            done, out_time = task.execute(frequency, time_slot)

            # pop the task if it is finished or out of time
            if done or out_time:
                return self.buffer.pop(0)
        return None


class Server:
    def __init__(self, frequency, queue_len, pos_x, pos_y, time_slot=1, trans_coeff=1):
        self.f = frequency
        self.queue = Buffer(queue_len)
        self.x, self.y = pos_x, pos_y
        self.time_slot = time_slot
        self.remaining_buffer_time = 0   # record time to process remaining tasks in the buffer
        self.trans_coeff = trans_coeff

    def get_queue_state(self):
        return len(self.queue)

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
    def __init__(self, server_num, device_num, x_range, y_range, mode='all',
                 max_frequency=1000, max_task_size=100, max_queue=10, rand_seed=0):
        self.server_num = server_num
        self.device_num = device_num
        self.rand = random
        self.rand.seed(rand_seed)
        self.rand_seed = rand_seed
        self.x_range = x_range
        self.y_range = y_range
        self.max_frequency = max_frequency
        self.max_task_size = max_task_size
        self.max_queue = max_queue

        # mode = all means all devices use the RL based decision making
        # mode = single means only the single device use the RL based decision making
        self.mode = mode

        self.servers, self.devices, self.listen_queue = None, None, None
        self.generated_tasks = []   # store tasks generated right now, and wait the offloading decision

        # variable to record the servers' states, including frequencies, remaining buffer size, location(x, y)
        self.status_dataframe = pd.DataFrame(columns=['frequency', 'remain_buffer', 'pos_x', 'pos_y'])

        self.reset()

    def reset(self):
        self.servers = []
        self.devices = []
        self.generated_tasks = []

        # generate servers
        for i in range(self.server_num):
            frequency = random.random() * self.max_frequency + 0.1 * self.max_frequency
            queue_len = int(random.random() * self.max_queue + 0.1 * self.max_queue)
            x_pos = random.uniform(-self.x_range, self.x_range)
            y_pos = random.uniform(-self.y_range, self.y_range)
            self.servers.append(Server(frequency, queue_len, x_pos, y_pos))
            self.status_dataframe['frequency'][i] = frequency
            self.status_dataframe['remain_buffer'][i] = queue_len
            self.status_dataframe['pos_x'] = x_pos
            self.status_dataframe['pos_y'] = y_pos

        # generate devices
        for i in range(self.device_num):
            generate_seed = random.random()
            delay = random.random() * 10 + 1
            new_device = Device(i, self.x_range, self.y_range, generate_seed, self.max_task_size, delay, self.rand_seed)
            new_task = new_device.generate()
            self.devices.append(new_device)
            self.generated_tasks.append(new_task)

        return [self.status_dataframe, self.generated_tasks]

    def step(self, action):
        if self.mode == 'self':
            return self.run_all_devices(action)

    def run_all_devices(self, actions):
        """
        Execute the decision from all devices, and return the state of the environment.\n
        :param actions: decision from all devices. If the corresponding task is None, the decision will be ignored.
        :return: observation of the environment
        """
        # process the tasks generated in the last time slot with the decisions
        reward = torch.zeros(self.device_num)
        for i in range(self.device_num):
            existing_task = self.generated_tasks[i]
            tasks_rejected_status, execution_time = self.servers[actions[i]].accept_task(existing_task)
            if not tasks_rejected_status:
                # task is accepted
                if execution_time > existing_task.delay_constraint:
                    reward[i] = -1
                    existing_task.source.unfinished_tasks_amount += 1
                else:
                    reward[i] = 1   # can be modified
                    existing_task.source.finished_tasks_amount += 1
                    existing_task.source.finished_delay += execution_time
            else:
                existing_task.source.rejected_tasks_amount += 1

        # reset the generated tasks list
        self.generated_tasks = []

        # run the servers and generate new tasks
        for i in range(self.server_num):
            working_server = self.servers[i]
            working_server.execute()
            self.status_dataframe['remain_buffer'] = working_server.get_queue_state()

        for i in range(self.device_num):
            task = self.devices[i].generate()
            self.generated_tasks.append(task)

        return [self.status_dataframe, self.generated_tasks]
