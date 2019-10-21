import copy
import uuid
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding

class ProductItem():
    def __init__(self, item_type, production_time, expire_time):
        self._id = uuid.uuid1()
        self._item_type = item_type
        # how long it takes to produce
        self._production_time = production_time

        # how long the product stays fresh
        self._expire_time = expire_time

        # age of product, if negative it is still being produced
        self.age = -production_time
    
    def is_done(self):
        return self.age > 0

    def is_fresh(self):
        return self.age < self._expire_time

    def step(self):
        self.age += 1


class Order():
    def __init__(self, item_type):
        self._item_type = item_type
        self.is_done = False
        self.waiting_time = 0

    def step(self):
        if not self.is_done:
            self.waiting_time += 1

class Inventory():
    def __init__(self):
        self._products = []

    def reset(self):
        self._products = []

    def add(self, product):
        for item in product:
            self._products.append(item)
        
    def take(self, product_id):
        for i in range(product_id):
            self._products.pop()

    def products(self):
        return self._products

    def step(self):
        for item in self._products:
            item.step()


class ProducerModel():
    def __init__(self):
        # baking in the oven
        self._production_queue = []

    def _is_busy(self):
        return len(self._production_queue)>0

    def _is_all_ready(self):
        return all([x.is_done() for x in self._production_queue])

    def reset(self):
        self._production_queue = []
        return self._production_queue

    def production_queue(self):
        return self._production_queue

    def start_producing(self, product_type, num_product):
        if self._is_busy():
            return False

        for i in range(num_product):
            self._production_queue.append(ProductItem('type', 5, 100))
        
        return True

    def step(self):
        # update
        for item in self._production_queue:
            item.step()

        if self._is_all_ready(): # check status
            ready_products = self._production_queue.copy()
            self._production_queue = []
            return ready_products

        return []

class ConsumerModel():
    def __init__(self):
        self._order_queue = []

    def reset(self):
        self._order_queue = []
        self._num_new_order = np.random.randint(0,2)

    def order_queue(self):
        return self._order_queue

    def make_orders(self, inventory_products):
        self._num_new_order = np.random.randint(0,3)
        print(self._num_new_order)
        return self._num_new_order

    def step(self):
        for order in self._order_queue:
            order.step()

    def _serve_orders(self, inventory_products):
        """
        split orders and available, remove items from the queue
        """
        n = self.make_orders(inventory_products)

        for i in range(n):
            self._order_queue.append(Order('type'))

        total_order = len(self._order_queue)
        available = len(inventory_products)

        take = available if available<=total_order else total_order

        remain = total_order - take

        for i in range(take):
            self._order_queue.pop()

        assert remain == len(self._order_queue)
        return take, remain



class InventoryTrackingEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(15)
        self.states_space = np.zeros((2,))

        self._producer_model = ProducerModel()
        self._inventory = Inventory()
        self._consumer_model = ConsumerModel()

        self.timestamp = 0
        self.state = dict()
        self.state_history = None
        self.fig = None
        self.axes = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if plt.get_fignums():
            plt.ioff()
            plt.show()
            self.fig = None
            self.axes = None

        self.timestamp = 0
        self._producer_model.reset()
        self._consumer_model.reset()
        self._inventory.reset()

        self.state = {
            "producer": self._producer_model.production_queue(),
            "consumer": copy.deepcopy(self._consumer_model.order_queue()),
            "inventory": self._inventory.products(),
            "num_ready": 0,
            "act": 0,
            "take": 0,
            "add_and_take":0,
            "num_new_orders":0,
        }

        self.state_history={}
        self.acc_list = list()

        return self.state

    def step(self, action):
        assert self.action_space.contains(action)
        self.act = action
 
        self.timestamp += 1

        ready_products = self._producer_model.step()
        self._inventory.add(ready_products)
        curr_products = self._inventory.products()
        consum_products, orderqueue = self._consumer_model._serve_orders(curr_products)
        self._inventory.take(consum_products)
        self._producer_model.start_producing('type',action)
        self._consumer_model.step()
        self._inventory.step()

        self.state["num_ready"] = len(ready_products)
        self.state["take"] = consum_products
        self.state["act"] = action
        self.state["add_and_take"] = len(ready_products)-consum_products
        self.state["num_new_orders"] = len(self._consumer_model.order_queue()) - len(self.state["consumer"]) + consum_products
        self.state["consumer"] = copy.deepcopy(self._consumer_model.order_queue())
        self.state["producer"] = self._producer_model.production_queue()
        self.state["inventory"] = self._inventory.products()

        self.accumulate_state(self.state)
 
        done = self.timestamp>100
 
        return self.state, 0, done, {}

    def render(self, mode='human', close=False):
        if not self.state_history:
            return
        screen_width = 600
        screen_height = 400

        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(4,1)
            plt.ion()

        for axis in self.axes:
            axis.clear()

        self.axes[0].plot(self.state_history["num_products"], label="inventory_products")
        self.axes[0].plot(self.state_history["num_orders"], label="customer_orders")
        self.axes[0].legend(loc="upper right")
        self.axes[0].set_title("inventory")

        self.axes[1].plot(self.state_history["num_new_production"], label="produce")
        self.axes[1].plot(self.state_history["num_new_orders"], label="order")
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_title("step actions")

        self.axes[2].plot(self.state_history["num_production"], label="in baking")
        self.axes[2].plot(self.state_history["num_ready"], label="done baking")
        self.axes[2].legend(loc="upper right")
        self.axes[2].set_title("producer model")

        self.axes[3].plot(self.state_history["num_new_done_orders"], label="done ordering")
        self.axes[3].plot(self.state_history["num_new_add_products"], label="inv diff")
        self.axes[3].legend(loc="upper right")
        self.axes[3].set_title("consumer model")

        plt.subplots_adjust(hspace=0.3)
        plt.draw()
        plt.pause(0.001)
        return np.array(self.fig.canvas.renderer.buffer_rgba())

    def close(self):
        if self.fig:
            plt.close()
            self.fig = None
            self.axes = None

    def accumulate_state(self, state):
        self.acc_list.append(copy.deepcopy(self.state))
        self.state_history["num_products"] = [len(x["inventory"]) for x in self.acc_list]
        self.state_history["num_orders"] = [len(x["consumer"]) for x in self.acc_list]
        self.state_history["num_new_orders"] = [x["num_new_orders"] for x in self.acc_list]
        self.state_history["num_new_production"] = [x["act"] for x in self.acc_list]
        self.state_history["num_ready"] = [x["num_ready"] for x in self.acc_list]
        self.state_history["num_production"] = [len(x["producer"]) for x in self.acc_list]
        self.state_history["num_new_done_orders"] = [x["take"] for x in self.acc_list]
        self.state_history["num_new_add_products"] = [x["add_and_take"] for x in self.acc_list]




class Metric():
    def __init__(self, env):
        self._env = env
        
        self._previous_state = {}
        self._current_state = {}

        self.state_descriptor={}
        self.metric = {}

    def reset(self):
        return self.metric

    def get_state_descriptor(self, previous_state, current_state):
        # state_metric
        self.state_descriptor["num_of_orders"] = len(current_state["order_queue"])
        self.state_descriptor["num_consumptions"] = len(current_state["consumer_queue"])
        self.state_descriptor["num_of_remaining_products"] = len(current_state["products"])

        fresh_states = 0
        for item in current_state["products"]:
            fresh_states += item.age
        self.state_descriptor["fresh_states_of_remaining"] = fresh_states

        waiting_time = 0
        for order in current_state["order_queue"]:
            waiting_time += order.waiting
        self.state_descriptor["sum_of_waiting_time"] = waiting_time

        # step_metric
        self.state_descriptor["num_new_orders"] = len(current_state["order_queue"]) - len(previous_state["order_queue"])
        self.state_descriptor["num_new_products"] = len(current_state["products"]) - len(previous_state["products"])

        return self.state_descriptor

    def calculate_metric_function(self, previous_state, current_state):
        self.get_state_descriptor(preivous_state, current_state)

        self.metric["remaining_products"] = - self.state_descriptor["num_of_remaining_products"]

        # weighted sum from all metrics
        self.metric["episode_metric"] = f1(self.state_descriptor)
        """
        f1 : -num_remaining_products + fresh_states_of_remaining + num_total_cosumption - num_remaining_orders + num_total_production
        """

    def step(self):
        self._current_state = self._env.get_state_summary()

        if self._previous_state:
            self.calculate_metric_function(self._previous_state, self._current_state)
        
        self._previous_state.update(self._current_state)
    
        if is_done:
            self.get_final_metric()

        return self.metric


class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, states):
        isbusy = states["isbusy"]
        isfresh = states["products"]["is_fresh"]
        order = states["oders"]["is_empty"]
        act = f(states, self.action_space)
        return act


def train(agent, env, metric, num_episodes, num_steps):
    for episode in range(num_episodes):

        states = env.reset()
        metric.reset()

        for step in range(num_steps):

            action = agent.act(states)

            states, reward, done, info = env.step(action)
            
            metric.step()

            if done:
                break