import gym
import random
import queue
from threading import RLock


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

    def generate(self, trans_coff):
        if self.rand.random() > self.rand_threshold:
            return None
        # need to test the degree of dispersion to verify whether the variance is appropriate.
        rand_size = random.normalvariate(self.size, 1)
        task = Task(rand_size, self, self.delay, trans_coff)
        return task

    def update_state(self, task):
        self.total_tasks += 1
        if task.reject:
            self.rejected_tasks_amount += 1
        elif task.out_of_time:
            self.unfinished_tasks_amount += 1
        else:
            self.finished_tasks_amount += 1
            self.finished_delay = self.finished_delay + task.execute_delay + task.trans_delay


class Task:
    def __init__(self, size, source: Device, delay, trans_coff):
        self.size = size
        self.remain_size = size   # remaining task size need to be executed
        self.source = source
        self.delay_constraint = delay
        self.dst = None    # destination device

        self.done = False    # indicate the state of the task
        self.reject = False    # indicate that the task is rejected by the server
        self.out_of_time = False   # indicate whether the task is out of time or not

        self.execute_delay = 0
        self.trans_delay = 0
        self.trans_coff = trans_coff

    def set_dst(self, destination: Device):
        self.dst = destination
        trans_distance = (self.source.x - destination.x) ** 2 + (self.source.y - destination.y) ** 2
        self.trans_delay = trans_distance ** 0.5 * self.trans_coff

    def execute(self, frequency, time_slot):
        """
        Execute the task, check the state whether it is done or out of time\n
        :param frequency: frequency of server
        :param time_slot: time_slot
        :return: if done, return True, else False
        """
        self.remain_size -= frequency * time_slot
        self.execute_delay += time_slot
        if self.remain_size <= 0:
            self.done = True
        if (self.execute_delay + self.trans_delay) > self.delay_constraint:
            self.out_of_time = True
        return self.done, self.out_of_time


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
            task.reject = True
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


class ListenQueue:
    """
    Listen and pop the executed tasks(including finished, out of time, and rejected) into the queue.\n
    return the tasks in the queue to the source device.
    """
    def __init__(self):
        self.listen_queue = queue.Queue(0)
        self.lock = RLock()

    def receive_tasks(self, task: Task):
        self.lock.acquire()
        try:
            self.listen_queue.put(task)
        except Exception as e:
            print(str(e))
        finally:
            self.lock.release()

    def back_to_device(self):
        """
        :return: If queue is not empty, return the task, else None
        """
        task = None
        self.lock.acquire()
        try:
            if not self.listen_queue.empty():
                task = self.listen_queue.get(block=False)
        except Exception as e:
            print(str(e))
        finally:
            self.lock.release()
        return task


class Server:
    def __init__(self, frequency, queue_len, pos_x, pos_y, time_slot=1):
        self.f = frequency
        self.queue = Buffer(queue_len)
        self.x, self.y = pos_x, pos_y
        self.time_slot = time_slot

    def get_queue_state(self):
        return len(self.queue)

    def accept_task(self, task: Task):
        """
        Accept tasks from devices.\n
        :param task: task from device
        :return: If the buffer in the server is not full, return None, else task
        """
        state = self.queue.push(task)
        if not state:
            task.reject = True
            return task
        return None

    def execute(self):
        """
        Execute tasks in the buffer\n
        :return: If the task in the top of the buffer is finished or out of time, return task,
        else return None
        """
        task = self.queue.execute(self.f, self.time_slot)
        return task


class ServerEnv(gym.Env):
    def __init__(self, server_num, device_num, x_range, y_range,
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

        self.servers, self.devices, self.listen_queue = None, None, None

        self.reset()

    def reset(self):
        self.servers = []
        self.devices = []
        self.listen_queue = ListenQueue()

        # generate servers
        for i in range(self.server_num):
            frequency = random.random() * self.max_frequency + 0.1 * self.max_frequency
            queue_len = int(random.random() * self.max_queue + 0.1 * self.max_queue)
            x_pos = random.uniform(-self.x_range, self.x_range)
            y_pos = random.uniform(-self.y_range, self.y_range)
            self.servers.append(Server(frequency, queue_len, x_pos, y_pos))

        # generate devices
        for i in range(self.device_num):
            generate_seed = random.random()
            delay = random.random() * 10 + 1
            self.devices.append(Device(i, self.x_range, self.y_range, generate_seed, self.max_task_size, delay,
                                       self.rand_seed))

    def step(self, action):
        self.servers
