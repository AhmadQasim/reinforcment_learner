# %%
import numpy as np
from gym_baking.envs.consumers.base_consumer import BaseConsumer


def sample_from_multinomial(probs):
    return np.random.multinomial(1, probs)


def sample_beta_prior(weight_vector):
    return np.random.dirichlet(weight_vector)


def sample_from_poisson(mean):
    return np.random.poisson(mean)


class CategoricalConsumerModel(BaseConsumer):
    def __init__(self, config):
        """there should be a list called "weight_list" for the weights of the categories in yaml file
        the last weight is the weight of no-order category"""
        super().__init__(config)
        self.weight_vector = config["weight_list"]

    def make_orders(self, inventory_products, order_queue, timestep):
        assert len(self.weight_vector) == len(self.products) + 1, "Alpha vector size \
                       should be 1 bigger than the products size"
        beta_prior = sample_beta_prior(self.weight_vector)
        sampled_category = np.asscalar(sample_from_multinomial(beta_prior).nonzero()[0])

        if sampled_category == len(self.products):
            return 0, None

        return 1, [sampled_category]


class PoissonConsumerModel(BaseConsumer):
    def __init__(self, config):
        """some examples to construct counts parameter in products:
        - if we assume a product will be ordered 100 times in second half of the episode
            counts: [0,100]
        - if it is ordered 50 times for the first quarter, 20 for the second half
            counts: [50,0,10,10]
        - if it is assumed to be ordered 200 for the whole episode
            counts: [200]
        as we add new element to these lists, it will produce more partitions
        with considering the values as the estimated number of products for that
        partition
        """
        super().__init__(config)
        self.maximum_time_steps = config["episode_max_steps"]
        if self.domain_randomization:
            self.counts_list = []
            self.ranges = [product['ranges'] for key, product in self.products.items()]
            for product_ranges in self.ranges:
                self.counts_list.append([np.random.randint(range[0], range[1]) for range in product_ranges])
        else:
            self.counts_list = [product['counts'] for key, product in self.products.items()]
        self.numbers_of_dilation = [float(len(counts)) for counts in self.counts_list]
        self.lambdas_list = self._get_lambdas_list(self.counts_list)
        self.is_overriden = False
        self.prediction = config['PREDICTION']

    def fix_seed(self, seed):
        np.random.seed(seed)

    def _lambdas_for_timestep(self, timestep):
        lambda_vals = []
        for ind, lambdas in enumerate(self.lambdas_list):
            lambda_vals.append(next(val for index, val in enumerate(lambdas)
                                    if
                                    (index + 1) * (self.maximum_time_steps / self.numbers_of_dilation[ind]) > timestep))
        return lambda_vals

    def _get_lambdas_list(self, counts_list):
        lambdas_list = []
        for ind, counts in enumerate(counts_list):
            lambdas_list.append([count * self.numbers_of_dilation[ind] / self.maximum_time_steps for count in counts])
        return lambdas_list

    def make_orders(self, timestep):
        # override orders
        if self.is_overriden:
            item_list = self.prediction
            number_of_item_list = [sum(lst) for lst in self.prediction]
            a = [index for index, item in enumerate(item_list[timestep]) for _ in range(item)]
            return (number_of_item_list[timestep], a)

        poisson_outcomes = [sample_from_poisson(i) for i in self._lambdas_for_timestep(timestep)]
        number_of_items = np.sum(poisson_outcomes)
        items = [i for i, item_count in enumerate(poisson_outcomes) for _ in range(item_count)]

        return number_of_items, items

# %%
'''
config = {'product_list': {"brot": {'counts': [100, 50, 0, 0]}, "pretzel": {'counts': [25, 25, 25, 0]},
                           "vater": {'counts': [300]},
                           "pizza": {'counts': [0, 25, 50, 50]},
                           "beer": {'counts': [0, 25, 100, 100]}, "döner": {'counts': [0, 0, 25, 50]}},
          'episode_max_steps': 100}
consumer = PoissonConsumerModel(config)
orders = consumer.make_orders("", "", 80)
print(orders)
counts = Counter(orders[1])
print(counts)
if orders is not None:
    plt.bar(range(len(counts)), counts.values(), align='center', alpha=0.5)
    plt.xticks(range(len(counts)), [list(config['product_list'].keys())[element] for element in counts.keys()])
    plt.show()
'''