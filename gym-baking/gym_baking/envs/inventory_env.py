import copy
from collections import Counter
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
        self.state = {'products': self._products}

    def reset(self):
        self._products.clear()
        return self.get_state()

    def add(self, product):
        for item in product:
            self._products.append(item)
        
    def take(self, taken_queue):
        for item in taken_queue:
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
            item = ProductItem(self.config[product_type]["type"], self.config[product_type]["production_time"], self.config[product_type]["expire_time"])
            self._production_queue.append(item)
        
        return True

    def step(self):
        # update
        for item in self._production_queue:
            item.step()


class ConsumerModel():
    def __init__(self, config):
        self.config = config
        self._order_queue = []
        self.state = {}
        self.state["order_queue"] = []

    def reset(self):
        self._order_queue.clear()
        self._debug_new_order_queue = []
        return self.get_state()

    def get_state(self):
        self.state["order_queue"] = self._order_queue.copy()
        self.state["debug_new_order_queue"] = self._debug_new_order_queue.copy()
        return self.state

    def make_orders(self, inventory_products, order_queue, timestep):
        num_new_order = np.random.randint(0,3)
        type_ids = np.random.choice(len(self.config), num_new_order, replace=True)

        self._debug_new_order_queue = []
        for i in range(num_new_order):
            order = Order(self.config[type_ids[i]]['type'])
            self._debug_new_order_queue.append(order)
        return num_new_order, type_ids

    def step(self):
        for order in self._order_queue:
            order.step()

    def _serve_orders(self, inventory_products, timestep):
        """
        split orders and available, remove orders from the order queue
        """
        n, type_ids = self.make_orders(inventory_products, self._order_queue, timestep)

        for i in range(n):
            order = Order(self.config[type_ids[i]]['type'])
            self._order_queue.append(order)

        order_counter = Counter([x._item_type for x in self._order_queue])
        product_counter = Counter([x._item_type for x in inventory_products])
        
        union_counter = order_counter & product_counter

        order_counter.subtract(union_counter)

        # update order queue
        order_dict = {}
        for order in self._order_queue:
            order_dict.setdefault(order._item_type, []).append(order)       
       
        new_order_queue = []
        for item_type, num in order_counter.items():
            new_order_queue += order_dict.get(item_type, [])[:num]

        self._order_queue = new_order_queue

        # update take queue
        inventory_dict = {}
        for item in inventory_products:
            inventory_dict.setdefault(item._item_type, []).append(item)

        take_queue = []
        for item_type, num in union_counter.items():
            take_queue += inventory_dict.get(item_type, [])[:num]      

        return take_queue, new_order_queue


class InventoryManagerEnv(gym.Env):
    def __init__(self):
        self.config = {
            0: {'type':'brot','production_time':5, 'expire_time':100},
            1: {'type':'pretzel','production_time':15, 'expire_time':20},
        }

        self._validate_config(self.config)
        self.product_list = [x['type'] for x in self.config.values()]

        num_types = len(self.config)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([num_types-1,15]), dtype=np.int64)

        self._producer_model = ProducerModel(self.config)
        self._inventory = Inventory()
        self._consumer_model = ConsumerModel(self.config)

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
        if plt.get_fignums():
            plt.ioff()
            plt.show()
            self.fig = None
            self.axes = None

        self.timestep = 0

        self.state["timestep"] = 0
        self.state["action"] = None
        self.state["ready_queue"] = []
        self.state["taken_queue"] = []
        self.state["consumer_state"] = self._consumer_model.reset()
        self.state["producer_state"] = self._producer_model.reset()
        self.state["inventory_state"] = self._inventory.reset()

        self.state_history = {}

        observation = {k:self.state[k] for k in ["producer_state", "inventory_state", "consumer_state"]}

        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        self.act = action
 
        ready_products = self._producer_model.get_ready_products()
        self._inventory.add(ready_products)
        curr_products = self._inventory.get_state()["products"]
        taken_products, orderqueue = self._consumer_model._serve_orders(curr_products, self.timestep)
        self._inventory.take(taken_products)
        self._producer_model.start_producing(action[0], action[1])
        self._producer_model.step()
        self._consumer_model.step()
        self._inventory.step()

        self.state["timestep"] = self.timestep
        self.state["action"] = action
        self.state["ready_queue"] = ready_products
        self.state["taken_queue"] = taken_products
        self.state["consumer_state"] = self._consumer_model.get_state()
        self.state["producer_state"] = self._producer_model.get_state()
        self.state["inventory_state"] = self._inventory.get_state()
        self.get_state_history(self.state)

        self.timestep += 1
        
        observation = {k:self.state[k] for k in ["producer_state", "inventory_state", "consumer_state"]}
        
        done = self.timestep>100

        reward = self._metric.get_metric(self.state_history, done)

        return observation, reward, done, {}

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
            self.axes[1].plot(self.state_history["debug_new_order_"+key], label = 'new_order_'+key)
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_title("step actions")

        self.axes[2].plot(self.state_history["in_production"], label="in production")
        for key in self.product_list:
            self.axes[2].plot(self.state_history['ready_queue_'+key], label = 'ready_product_'+key)
        self.axes[2].legend(loc="upper right")
        self.axes[2].set_title("producer model")

        for key in self.product_list:
            self.axes[3].plot(self.state_history["taken_queue_"+key], label = 'taken_queue_'+key)
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
        taken_count = Counter([x._item_type for x in state["taken_queue"]])
        order_count = Counter([x._item_type for x in state["consumer_state"]["order_queue"]])
        inventory_count = Counter([x._item_type for x in state["inventory_state"]["products"]])
        new_order = Counter([x._item_type for x in state["consumer_state"]["debug_new_order_queue"]])
        
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
            self.state_history.setdefault('taken_queue_'+key, []).append(taken_count[key])
            self.state_history.setdefault('debug_new_order_'+key, []).append(new_order[key])

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
        if not done:
            return 0
        sales = 0
        wastes = 0
        for key in self.product_list:
            sales += sum(state_history['taken_queue_'+key])
            wastes += sum(state_history['inventory_'+key])
        
        return sales - wastes