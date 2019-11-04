from gym_baking.envs.consumers.base_consumer import BaseConsumer
from gym_baking.envs.consumers.utils import *


class AverageConsumer(BaseConsumer):
    def __init__(self, config):
        super().__init__(config)
        self.top_n_items = config['top_n_items']
        self.data_path = config['data_path']
        self.opening_time = get_min_of_day(config['opening_time'])
        self.model = None
        self.std = 0.3

        self.data = load_data(self.data_path)
        self.data, self.X, self.y = preprocess_data(self.data, self.top_n_items, self.opening_time)

    def make_orders(self, inventory_products, order_queue, timestep):
        time = self.get_nearest_time(timestep)

        # sample order
        order = self.sample_order(time)

        # add gaussian noise to order, making it stochastic
        mutated_order = self.mutate(order)[0]

        # conform to return format
        return format_order_result(mutated_order)

    def get_nearest_time(self, time):
        while self.X[self.X == time].shape[0] == 0:
            time += 1

        return time

    def sample_order(self, time):
        sample = self.y[(self.X == time).flatten(), :]
        days = np.sum(sample != 0, axis=0)
        order = np.sum(sample, axis=0) / (days + 1e-5)

        return order

    def mutate(self, order):
        order = order + (np.random.randn(1, order.shape[0]) * self.std)

        return np.round(order)
