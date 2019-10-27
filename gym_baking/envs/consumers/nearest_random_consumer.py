from gym_baking.envs.consumers.base_consumer import BaseConsumer
from gym_baking.envs.consumers.utils import *


class NearestRandomConsumer(BaseConsumer):
    def __init__(self, config):
        super().__init__(config)
        self.top_n_items = config['top_n_items']
        self.data_path = config['data_path']
        self.opening_time = get_min_of_day(config['opening_time'])
        self.model = None

        self.data = load_data(self.data_path)
        self.data, self.X, self.y = preprocess_data(self.data, self.top_n_items, self.opening_time)

    def make_orders(self, inventory_products, order_queue, timestep):
        timestep = np.random.randint(0, 300)
        time = self.get_nearest_time(timestep)

        # sample order
        order = self.sample_order(time)

        # conform to return format
        return format_order_result(order)

    def get_nearest_time(self, time):
        while self.X[self.X == time].shape[0] == 0:
            time += 1

        return time

    def sample_order(self, time):
        sample = self.y[(self.X == time).flatten(), :]
        idx = np.random.randint(0, sample.shape[0])
        return sample[idx, :]
