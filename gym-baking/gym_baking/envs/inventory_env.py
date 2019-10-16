import uuid
import numpy as np
import gym

class ProductItem():
    def __init__(self, item_type, production_time, expire_time):
        self._id = uuid.uuid1()
        self._item_type = item_type
        # how long it takes to produce
        self._production_time = production_time

        # how long the product stays fresh
        self._expire_time = expire_time

        # age of product, if negative it is still being produced
        self._age = -production_time
    
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
        self.waiting = 0

    def step(self):
        if not self.is_done:
            self.waiting += 1

class Inventory():
    def __init__(self):
        self._products = {}

    def add(self, product):
        self._products[product._id] = product
        
    def take(self, product_id):
        return self._products.pop(product_id)
    
    def products(self):
        return self._products

    def step(self):
        # update items
        for item in self._products.items():
            item.step()
        

class ProducerModel():
    def __init__(self):
        # baking in the oven
        self._production_queue()

        # product is ready
        self.ready_queue()

    def is_busy(self):
        return len(self._production_queue)>0

    def production_queue(self):
        return self._production_queue()

    def ready_products(self):
        return self.ready_queue

    def is_all_ready(self):
        for item in self._production_queue:
            if not item.is_done():
                return False
        return True

    def start_producing(self, products):
        """
        products: [item_type, item_number]
        """
        if self.is_busy():
            return

        for item_type, item_number in products:
            
            if item_type not in self._products_lists:
                self._products_lists[item] = ProductItem()

            # TODO create new instances with "age" attributes set to 0, but use corresponding uuid
            product_item = ProductItem(item_type, production_time, expire_time)

            self._production_queue.push(product_item)
        
    def step(self):
        # update
        for item in self._production_queue:
            item.step()

        if self.is_all_ready(): # check status
            self.ready_queue, self._production_queue = self._production_queue, [] # clear if ready

            # self.inventory.add(self.ready_queue)

        
class ConsumerModel():
    def __init__(self, product_list):
        self._order_queues = []
        self._consumer_queue = []
        self._product_list = product_list

    def consumer_queue(self):
        return self._consumer_queue()

    def sample_from_product_list(self):
        # random samples from all products(currently available and not available)
        num_samples = np.random.random_integers(0, len(self._product_list))
        sample_indices = np.random.choice(len(self._product_list), num_samples, replace=True)
        return self._product_list[sample_indices]

    def sample_from_existing(self, inventory_products):
        # random samples from products that are currently available
        num_samples = np.random.random_integers(0, len(inventory_products))
        sample_indices = np.random.choice(len(inventory_products), num_samples, replace=False)
        return inventory_products[sample_indices]

    def sample_from_nonexisting(self, inventory_products):
        # random samples from products that are currently not available
        nonexisting = []
        for item in self._product_list:
            if not item in inventory_products:
                nonexisting.append(item)
        
        num_samples = np.random.random_integers(0, len(nonexisting))
        sample_indices = np.random.choice(len(nonexisting), num_samples, replace=True)
        return self._product_list[sample_indices]

    def add_random_orders(self, inventory_products):

        if np.random.rand()>0.8:
            self._order_queue.extend(self.sample_from_existing())
        else:
            self._order_queue.extend(self.sample_from_nonexisting())
    
        for order in self._order_queue:
            if order in inventory_products:
                self.order_queue.push(order)
                self.consumer_queue.add(order)

        return self.consumer_queue, self.order_queue

    def step(self):

        for order in self._order_queues:
            order.step()



class InventoryTrackingEnv(gym.Env):
    def __init__(self, producer_model, inventory, consumer_model):
        self.action_space
        self.states_space
        self.reward_range
        self.viewer = None

        self._producer_model = producer_model
        self._inventory = inventory
        self._consumer_model = consumer_model

    def step(self, action):
        items = self._producer_model.ready_products()
        
        self._inventory.add(items)
        
        items = self._inventory.products()

        items, orders = self._consumer_model.add_random_orders(items)
        
        self._inventory.take(items)

        self._producer_model.start_producing(action)
        
        self._producer_model.step()
        
        self._consumer_model.step()
        
        self._inventory.step()


        states = self.get_state_summary()

        reward = reward_function()

        return states, reward, done, info

    def reset(self):
        self._producer_model.reset()
        self._consumer_model.reset()
        self._inventory.reset()
        return self.get_state_summary()

    def render(self):
        if self.viewer is None:
            fig, self.viewer = plt.subplots(2,1)
            plt.ion()
            plt.show()
        
        self.viewer[0].plot()
        self.viewer[1].plot()
        return self.fig.tonumpy()

    def close(self):
        pass
    def seed(self, seed=None):
        pass

    def get_state_summary(self):
        # gather states from producer, inventory and consumer
        states = {}
        states["consumer_queue"] = self.consumer_model.consumer_queue


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