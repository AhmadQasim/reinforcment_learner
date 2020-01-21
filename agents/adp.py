import sys
import numpy as np
from collections import Counter
import tensorflow as tf
import yaml
from gym_baking.envs.consumers.parametric_consumer import PoissonConsumerModel as Oracle

MAXIMUM_INVENTORY = 20
MAXIMUM_DELIVERY = 10
SAMPLE_SIZE = 1024
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3

class ADPAgent():
    def __init__(self, config_path):

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        self.products = config['product_list']
        self.maximum_time_step = config['episode_max_steps']
        self.produce_times = list(map(lambda x: x[1]["production_time"], self.products.items()))
        self.maximum_produce_time = max(self.produce_times)
        self.number_of_products = len(self.products.items())
        self.oracle = Oracle(config)

        # our state is the actual inventory vector + the last delivery vectors of max_produce_time steps
        # which is (self.max_produce_time+1)*self.number_of_products size matrix in total
        inputs = tf.keras.Input(shape=((self.maximum_produce_time + 1) * self.number_of_products,), name='state')
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_3')(x)
        outputs = tf.keras.layers.Dense(1, activation='softmax', name='costs')(x)
        self.func_approximator_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.r = np.array([2., 3.])  # cost of storing products(helps in penalizing aging)
        self.o = np.array([1., 5.])  # cost of stock-out
        self.c = np.array([1., 1.])  # cost of delivering a product

    def train_adp(self):
        for step in reversed(range(self.maximum_time_step)):
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            state_feature_matrix, cost_values = self.create_state_optimal_cost_pairs(orders, step, sample_size=SAMPLE_SIZE)
            self.train(state_feature_matrix, cost_values)

    def train(self, feature_matrix, y_values):
        train_dataset = tf.data.Dataset.from_tensor_slices((feature_matrix, y_values)).batch(BATCH_SIZE)
        self.func_approximator_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                           loss=tf.keras.losses.MeanSquaredError())
        self.func_approximator_model.fit(train_dataset, epochs=EPOCHS)

    def create_state_optimal_cost_pairs(self, orders, step, sample_size):
        state_feature_matrix = np.zeros((sample_size, (self.maximum_produce_time + 1) * self.number_of_products))
        cost_values = np.zeros(sample_size)
        for i in range(sample_size):
            random_inventory_state = np.random.randint(MAXIMUM_INVENTORY, size=self.number_of_products)
            random_delivery_matrix, last_delivery_time_step = self.create_random_delivery_matrix()
            state = (random_inventory_state, random_delivery_matrix)
            optimal_cost, _ = self.get_optimal_cost_action_pair(state, orders, last_delivery_time_step, step)
            state_feature_matrix[i, :] = np.append(random_inventory_state, np.matrix.flatten(random_delivery_matrix))
            cost_values[i] = optimal_cost

        return state_feature_matrix, cost_values

    # generate all possible actions
    def get_action_variants(self):
        for prod_index in range(self.number_of_products):
            for count in range(MAXIMUM_DELIVERY):
                action_vector = np.zeros(self.number_of_products)
                action_vector[prod_index] = count + 1
                yield (prod_index, action_vector)

    def get_optimal_cost_action_pair(self, state, orders, last_delivery_time_step, step):
        inventory, last_deliveries = state
        min_cost = sys.maxsize
        optimal_action = np.zeros(self.number_of_products)
        for prod_index, action_vector in self.get_action_variants():
            next_step_apprx_cost = 0

            if self.maximum_produce_time - last_delivery_time_step < self.produce_times[prod_index]:
                # do not produce anything if we are not able to do in time...
                action_vector = np.zeros(self.number_of_products)

            new_inventory = np.maximum(np.zeros_like(inventory), (inventory + action_vector - orders))
            new_state = np.append(new_inventory, np.append(last_deliveries[1:, :], action_vector))

            if step < self.maximum_time_step - 1:
                next_step_apprx_cost = self.func_approximator_model(new_state)
            else:
                # we do not have a next step cost if we are at the end
                next_step_apprx_cost = 0

            cost = self.cost(inventory, action_vector, orders) + next_step_apprx_cost
            min_cost = cost if cost < min_cost else min_cost
            optimal_action = action_vector if cost < min_cost else optimal_action

        return min_cost, optimal_action

    # creates a random delivery matrix obeying the produce time constraints
    def create_random_delivery_matrix(self):
        import random as rd
        random_delivery_matrix = np.zeros((self.number_of_products, self.maximum_produce_time))
        checked_time_steps = []
        remaining_time_steps = list(range(self.maximum_produce_time))

        while len(remaining_time_steps) != 0:
            delivery_step = rd.choice(remaining_time_steps)
            delivered_product = np.random.randint(self.number_of_products)
            if len(checked_time_steps) > 0:
                delivery_steps_placed_before = list(filter(lambda x: x < delivery_step, checked_time_steps))
                if len(delivery_steps_placed_before) > 0:
                    order_step_of_last_order = max(delivery_steps_placed_before)
                    if delivery_step - order_step_of_last_order < self.produce_times[delivered_product]:
                        remaining_time_steps.remove(delivery_step)
                        continue

            # the probability to not produce anything (can be changed later)
            no_delivery_prob = 1.0 / (1 + self.number_of_products)
            should_deliver = np.random.choice([0, 1], p=[no_delivery_prob, 1.0 - no_delivery_prob])
            if should_deliver == np.array([0]):
                remaining_time_steps.remove(delivery_step)
                continue

            amount_of_delivery = np.random.randint(30) + 1
            # sets the related amount to the related delivery in the matrix
            random_delivery_matrix[delivered_product, delivery_step] = amount_of_delivery
            checked_time_steps.append(delivery_step)
            # cleans the actual time step and the time steps before where the product is being produced
            # from the list for the next loop
            produce_time = self.produce_times[delivered_product]
            for i in range(delivery_step, max(-1, delivery_step - produce_time), -1):
                if i in remaining_time_steps:
                    remaining_time_steps.remove(i)

        last_delivery_step = max(checked_time_steps) if len(checked_time_steps) > 0 else 0
        return (random_delivery_matrix, last_delivery_step)

    def vectorize_counter(self, counter: Counter):
        counts = np.zeros(self.number_of_products)
        for indx, count in counter.items():
            counts[indx] = count
        return counts

    def cost(self, inventory, delivery, orders):
        return np.dot(self.c, delivery) + np.dot(self.r, inventory) + np.dot(self.o, np.maximum([0., 0.],
                                                                                                orders - delivery - inventory))

    def act(self, observation):
        return


if __name__ == '__main__':
    agent = ADPAgent(config_path="../inventory.yaml")
    agent.train_adp()
