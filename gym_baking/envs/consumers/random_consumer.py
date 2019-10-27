import numpy as np
from reinforcemnet_learner.gym_baking.envs.consumers.base_consumer import BaseConsumer


class RandomConsumer(BaseConsumer):
    def __init__(self, config):
        super().__init__(config)

    def make_orders(self, inventory_products, order_queue, timestep):
        num_new_order = np.random.randint(0, 6)
        type_ids = np.random.choice(len(self.config), num_new_order, replace=True)

        return num_new_order, type_ids