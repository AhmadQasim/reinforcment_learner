from collections import Counter
import uuid
import yaml
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from gym.utils import seeding

from gym_baking.envs.consumers.random_consumer import RandomConsumer as Consumer

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


class Inventory():
    def __init__(self):
        self._products = []
        self.state = {'products': self._products}

    def reset(self):
        self._products.clear()
        return self.get_state()

    def add(self, products):
        for item in products:
            self._products.append(item)
        
    def take(self, products):
        for item in products:
            self._products.remove(item)

    def get_state(self):
        self.state["products"] = self._products.copy()
        return self.state

    def step(self):
        for item in self._products:
            item.step()


class ProducerModel():
    def __init__(self, config):
        self.config = config
        self._production_queue = []
        self.state = {}
        self.state["production_queue"] = []
        self.state["is_busy"] = self.is_busy()

    def is_busy(self):
        return len(self._production_queue)>0

    def _is_all_ready(self):
        return all([x.is_done() for x in self._production_queue])

    def reset(self):
        self._production_queue.clear()
        return self.get_state()

    def get_state(self):
        self.state["production_queue"] = self._production_queue.copy()
        self.state["is_busy"] = self.is_busy()
        return self.state

    def get_ready_products(self):
        if self._is_all_ready():
            ready_products = self._production_queue.copy()
            self._production_queue.clear()
            return ready_products
        return []

    def start_producing(self, product_type, num_product):
        if self.is_busy():
            return False

        for i in range(num_product):
            item = ProductItem(
                self.config[product_type]["type"],
                self.config[product_type]["production_time"], 
                self.config[product_type]["expire_time"])
            self._production_queue.append(item)
        
        return True

    def step(self):
        # update
        for item in self._production_queue:
            item.step()


class InventoryManagerEnv(gym.Env):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        self.config = config['product_list']
        self.episode_max_steps = config['episode_max_steps']
        self._validate_config(self.config)
        self.product_list = [x['type'] for x in self.config.values()]

        num_types = len(self.config)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([num_types-1,30]), dtype=np.int64)
        self.observed_product = self.product_list

        self._producer_model = ProducerModel(self.config)
        self._inventory = Inventory()
        self._consumer_model = Consumer(self.config)

        self.timestep = 0
        self.state = dict()
        self.fig = None
        self.axes = None

        self.state_history = {}
        self._metric = Metric(self.config)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.timestep = 0

        self.state["timestep"] = 0
        self.state["action"] = None
        self.state["ready_queue"] = []
        self.state["serve_queue"] = []
        self.state["consumer_state"] = self._consumer_model.reset()
        self.state["producer_state"] = self._producer_model.reset()
        self.state["inventory_state"] = self._inventory.reset()

        self.state_history = {}

        observation = {k:self.state[k] for k in ["producer_state", "inventory_state", "consumer_state"]}

        return observation

    def step(self, action):
        assert self.action_space.contains(action), "invalid action {}".format(action)
        self.act = action
 
        ready_products = self._producer_model.get_ready_products()
        self._inventory.add(ready_products)
        curr_products = self._inventory.get_state()["products"]
        taken_products = self._consumer_model._serve_orders(curr_products, self.timestep)
        self._inventory.take(taken_products)
        self._producer_model.start_producing(action[0], action[1])
        self._producer_model.step()
        self._consumer_model.step()
        self._inventory.step()

        self.state["timestep"] = self.timestep
        self.state["action"] = action
        self.state["ready_queue"] = ready_products
        self.state["serve_queue"] = taken_products
        self.state["consumer_state"] = self._consumer_model.get_state()
        self.state["producer_state"] = self._producer_model.get_state()
        self.state["inventory_state"] = self._inventory.get_state()
        self.get_state_history(self.state)

        self.timestep += 1
        
        observation = {k:self.state[k] for k in ["producer_state", "inventory_state", "consumer_state"]}
        
        done = self.timestep>self.episode_max_steps

        reward, metric_info = self._metric.get_metric(self.state_history, done)

        return observation, reward, done, metric_info

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

        for key in self.product_list:
            self.axes[0].plot(self.state_history["inventory_"+key], label = 'inventory_'+key)
            self.axes[0].plot(self.state_history["order_queue_"+key], label = 'order_queue_'+key)
        self.axes[0].legend(loc="upper right")
        self.axes[0].set_title("inventory")

        for key in self.product_list:
            self.axes[1].plot(self.state_history["action_"+key], label = 'action_'+key)
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_title("step actions")

        self.axes[2].plot(self.state_history["in_production"], label="in production")
        for key in self.product_list:
            self.axes[2].plot(self.state_history['ready_queue_'+key], label = 'ready_product_'+key)
        self.axes[2].legend(loc="upper right")
        self.axes[2].set_title("producer model")

        for key in self.product_list:
            self.axes[3].plot(self.state_history["serve_queue_"+key], label = 'serve_queue_'+key)
        self.axes[3].legend(loc="upper right")
        self.axes[3].set_title("consumer model")

        plt.subplots_adjust(hspace=0.3)
        plt.draw()
        plt.pause(0.001)
        return np.array(self.fig.canvas.renderer.buffer_rgba())

    def get_state_history(self, state):
        in_production = len(state["producer_state"]["production_queue"])
        is_busy = state["producer_state"]["is_busy"]
        ready_count = Counter([x._item_type for x in state["ready_queue"]])
        taken_count = Counter([x._item_type for x in state["serve_queue"]])
        order_count = Counter([x._item_type for x in state["consumer_state"]["order_queue"]])
        inventory_count = Counter([x._item_type for x in state["inventory_state"]["products"]])
        
        self.state_history.setdefault('in_production', []).append(in_production)
        self.state_history.setdefault('is_busy', []).append(is_busy)

        for key in self.config.keys():
            num_request = state["action"][1] if state["action"][0]==key else 0
            self.state_history.setdefault("action_"+self.config[key]['type'], []).append(num_request)

        for key in self.product_list:
            self.state_history.setdefault('inventory_'+key, []).append(inventory_count[key])
            self.state_history.setdefault('order_queue_'+key, []).append(order_count[key])
            self.state_history.setdefault('production_queue_'+key, []).append(order_count[key])
            self.state_history.setdefault('ready_queue_'+key, []).append(ready_count[key])
            self.state_history.setdefault('serve_queue_'+key, []).append(taken_count[key])

    def _validate_config(self, config):
        product_list = [x['type'] for x in config.values()]
        assert len(set(product_list)) == len(config)

    def close(self):
        if self.fig:
            plt.close()
            self.fig = None
            self.axes = None


class Metric():
    def __init__(self, config):
        self.config = config
        self.product_list = [x["type"] for x in self.config.values()]

    def get_metric(self, state_history, done):
        info = {}
        if not done:
            return 0, info
        num_sales = 0
        num_waits = 0
        num_wastes = 0
        num_products = 0
        for key in self.product_list:
            num_sales += sum(state_history['serve_queue_'+key])
            num_waits += state_history['order_queue_'+key][-1]
            num_products += sum(state_history['ready_queue_'+key])
            num_wastes += state_history['inventory_'+key][-1]
        
        sale_wait_ratio = num_sales/(num_sales + num_waits) if num_sales + num_waits>0 else 0
        product_waste_ratio = num_products/(num_products+num_wastes) if num_products+num_wastes>0 else 0
        score = (sale_wait_ratio + product_waste_ratio)/2

        info["sales"] = num_sales
        info["waits"] = num_waits
        info["products"] = num_products
        info["wastes"] = num_wastes
        info["sale_wait_ratio"] = sale_wait_ratio
        info["product_waste_ratio"] = product_waste_ratio
        return score, info