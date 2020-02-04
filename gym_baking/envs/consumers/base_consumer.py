from collections import Counter
from gym_baking.envs.order import Order
import numpy as np


class BaseConsumer():
    def __init__(self, config):
        self.config = config
        self.products = config['product_list']
        self.domain_randomization = config['domain_randomization']
        self._order_queue = []
        self.state = {}
        self.state["order_queue"] = []

        #self.dummy_data = np.load('../reinforcemnet_learner/consumer_demand.npy')

    def reset(self):
        self._order_queue.clear()
        return self.get_state()

    def get_state(self):
        self.state["order_queue"] = self._order_queue.copy()
        return self.state

    def make_orders(self, inventory_products, order_queue, timestep):

        num_new_order = -1
        type_ids = -1
        assert False, "make_orders function should be overriden by a subclass"

        return num_new_order, type_ids


    def _serve_orders(self, inventory_products, timestep):
        """
        split orders and available, remove orders from the order queue
        """
        n, type_ids = self.make_orders(timestep)

        for i in range(n):
            order = Order(self.products[type_ids[i]]['type'])
            self._order_queue.append(order)

        order_counter = Counter([x._item_type for x in self._order_queue])
        product_counter = Counter([x._item_type for x in inventory_products])
        union_counter = order_counter & product_counter
        order_counter.subtract(union_counter)

        # update order queue
        order_dict = {}
        for order in self._order_queue:
            order_dict.setdefault(order._item_type, []).append(order)

        tmp_order_queue = []
        for item_type, num in order_counter.items():
            tmp_order_queue += order_dict.get(item_type, [])[:num]

        self._order_queue = tmp_order_queue

        # update serve queue
        inventory_dict = {}
        for item in inventory_products:
            inventory_dict.setdefault(item._item_type, []).append(item)

        serve_queue = []
        for item_type, num in union_counter.items():
            serve_queue += inventory_dict.get(item_type, [])[:num]

        return serve_queue

    def step(self):
        for order in self._order_queue:
            order.step()