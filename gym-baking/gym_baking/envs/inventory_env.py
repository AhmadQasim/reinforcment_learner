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
        self.state["products"] = self._products

    def add(self, product):
        for item in product:
            self._products.append(item)
        
    def take(self, product_id):
        for item in product_id:
            print(item)
            self._products.remove(item)
        # tmp, self._products = self._products, []
        # for item in product_id:
        #     self._products.append(item)

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
        self.state["production_queue"] = []
        self.state["is_busy"] = self.is_busy()

    def get_state(self):
        self.state["production_queue"] = self._production_queue.copy()
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
        self.state = {"order_queue" : []}

    def reset(self):
        self._order_queue.clear()
        self._num_new_order = np.random.randint(0,2)
        self._debug_new_order_queue = []

    def get_state(self):
        self.state["order_queue"] = self._order_queue.copy()
        self.state["debug_new_order_queue"] = self._debug_new_order_queue.copy()
        return self.state

    def make_orders(self, inventory_products, order_queue, timestep):
        self._num_new_order = np.random.randint(0,3)
        print(self._num_new_order)
        type_ids = np.random.choice(len(self.config), self._num_new_order, replace=True)

        self._debug_new_order_queue = []
        for i in range(self._num_new_order):
            order = Order(self.config[type_ids[i]]['type'])
            self._debug_new_order_queue.append(order)
        return self._num_new_order, type_ids

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


class InventoryTrackingEnv(gym.Env):
    def __init__(self):
        self.config = {
            0: {'type':'type0','production_time':5, 'expire_time':100},
            1: {'type':'type1','production_time':15, 'expire_time':20},
        }

        self.product_list = [x['type'] for x in self.config.values()]

        num_types = len(self.config)
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([num_types-1,15]), dtype=np.int64)
        self.states_space = np.zeros((2,))

        self._producer_model = ProducerModel(self.config)
        self._inventory = Inventory()
        self._consumer_model = ConsumerModel(self.config)

        self.timestep = 0
        self.state = dict()
        self.state_history = None
        self.fig = None
        self.axes = None

        self.new_history = {}
    
    def process_history(self, consumer_state, producer_state, inventory_state, ready_product):

        ready_count = Counter([x._item_type for x in ready_product])
        order_count = Counter([x._item_type for x in consumer_state["order_queue"]])
        new_order = Counter([x._item_type for x in consumer_state["debug_new_order_queue"]])

        for key in self.product_list:
            self.new_history.setdefault('ready_product_'+key, []).append(ready_count[key])
            self.new_history.setdefault('order_queue_'+key, []).append(order_count[key])
            self.new_history.setdefault('debug_new_order_'+key, []).append(new_order[key])


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
        self._producer_model.reset()
        self._consumer_model.reset()
        self._inventory.reset()

        self.new_history = {}

        self.state = {
            "timestep": 0,
            "producer": self._producer_model.get_state(),
            "consumer": self._consumer_model.get_state(),
            "inventory": self._inventory.get_state(),
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
 
        ready_products = self._producer_model.get_ready_products()
        self._inventory.add(ready_products)
        curr_products = self._inventory.get_state()["products"]
        taken_products, orderqueue = self._consumer_model._serve_orders(curr_products, self.timestep)
        print(taken_products)
        self._inventory.take(taken_products)
        self._producer_model.start_producing(action[0], action[1])
        self._producer_model.step()
        self._consumer_model.step()
        self._inventory.step()

        self.state["timestep"] = self.timestep
        self.state["num_ready"] = len(ready_products)
        self.state["take"] = len(taken_products)
        self.state["act"] = action[1]
        self.state["add_and_take"] = len(ready_products)-len(taken_products)
        self.state["num_new_orders"] = len(self._consumer_model.get_state()["order_queue"]) - len(self.state["consumer"]["order_queue"]) + len(taken_products)
        self.state["consumer"] = self._consumer_model.get_state()
        self.state["producer"] = self._producer_model.get_state()
        self.state["inventory"] = self._inventory.get_state()

        self.process_history(self.state["consumer"], None, None, ready_products)


        self.state["consumer_orders"] = {}
        for key, value in Counter([x._item_type for x in self.state["consumer"]["order_queue"]]).items():
            self.state["consumer_orders"][key] = value
        
        self.state["producer_items"] = {}
        for key, value in Counter([x._item_type for x in self.state["producer"]["production_queue"]]).items():
            self.state["producer_items"][key] = value
        
        self.state["inventory_items"] = {}
        for key, value in Counter([x._item_type for x in self.state["inventory"]["products"]]).items():
            self.state["inventory_items"][key] = value

        self.timestep += 1

        self.accumulate_state(self.state)
 
        done = self.timestep>100
 
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
        for key in self.config.values():
            key = "num_products_" + key["type"]
            self.axes[0].plot(self.state_history[key], label = key)
        for key in self.product_list:
            self.axes[0].plot(self.new_history['order_queue_'+key], label = 'order_queue_'+key)
        self.axes[0].legend(loc="upper right")
        self.axes[0].set_title("inventory")

        self.axes[1].plot(self.state_history["num_new_production"], label="produce")
        self.axes[1].plot(self.state_history["num_new_orders"], label="order")
        for key in self.product_list:
            self.axes[1].plot(self.new_history["debug_new_order_"+key], label = 'new_order_'+key)
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_title("step actions")

        self.axes[2].plot(self.state_history["num_production"], label="in baking")
        self.axes[2].plot(self.state_history["num_ready"], label="done baking")
        for key in self.product_list:
            self.axes[2].plot(self.new_history['ready_product_'+key], label = 'ready_product'+key)
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
        self.state_history["num_products"] = [len(x["inventory"]["products"]) for x in self.acc_list]

        for key in self.config.values():
            inv_key = "num_products_" + key['type']
            self.state_history[inv_key] = [x["inventory_items"].get(key['type'], 0) for x in self.acc_list]            

        self.state_history["num_orders"] = [len(x["consumer"]["order_queue"]) for x in self.acc_list]
        self.state_history["num_new_orders"] = [x["num_new_orders"] for x in self.acc_list]
        self.state_history["num_new_production"] = [x["act"] for x in self.acc_list]
        self.state_history["num_ready"] = [x["num_ready"] for x in self.acc_list]
        self.state_history["num_production"] = [len(x["producer"]["production_queue"]) for x in self.acc_list]
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