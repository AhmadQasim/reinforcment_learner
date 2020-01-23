import sys
import numpy as np
from collections import Counter
import tensorflow as tf
import yaml
from gym_baking.envs.consumers.parametric_consumer import PoissonConsumerModel as Oracle
import logging
from itertools import chain

# for simplicity we assume all products have the same inventory and delivery limits
MAXIMUM_INVENTORY = 2
MAXIMUM_DELIVERY = 2
ADP_THRESHOLD = 1e6 # size of state space to switch adp when exceeded
SHOULD_CHECK_FOR_ADP_THRESHOLD = False # bypasses the threshold check and does "exact dp" if True# , otherwise trains the "adp"

# variables needed in case it is "adp"
SAMPLE_SIZE = 1024
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
NO_DELIVERY_PROB_IN_STATE_SPACE_SEARCH = 1e-1 # if this value is not -1, it creates actions without any delivery with
# this probability while creating random states to approximate the values

class DPAgent():
    def __init__(self, config_path):

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        self.products = config['product_list']
        self.horizon = config['episode_max_steps']
        self.items = self.products.items()
        self.produce_times = list(map(lambda x: x[1]["production_time"], self.items))
        self.maximum_produce_time = max(self.produce_times)
        self.number_of_products = len(self.items)
        self.oracle = Oracle(config) #behave the poisson model as it is the oracle that knows what orders will come
        # which are exactly the expectance of the order amounts -> Dimitry Bertsekas calls this "certainty equivalent control"
        # which is swapping the random variable with the expected value of it
        self.store_cost_vector = np.array(list(map(lambda x: x[1]["storing_cost"], self.items)))
        self.delivery_cost_vector = np.array(list(map(lambda x: x[1]["delivery_cost"], self.items)))
        self.stock_out_cost_vector = np.array(list(map(lambda x: x[1]["stock_out_cost"], self.items)))

        # our state is "the actual inventory vector" + "the last delivery vectors of max_produce_time steps"
        # which can be thought as a (1 + self.max_produce_time)*self.number_of_products sized matrix
        # however there are some constraints in a delivery vector and between the delivery vectors
        # one can not deliver a product if the owen can not free during the time to produce that product
        # one can only produce a product at a time -> an example delivery vector:[10, 0, 0], not an example [10, 5, 0]
        # also inventory vector has a maximum limit
        # if the size of the state space exceeds ADP_THRESHOLD threshold we switch to adp
        # btw, this is a upper bound to the state space because some combinatons(maybe a large portion)
        # of delivery vectors can not be aligned due to the producing time constraints
        # if that amount is great, maybe we should change the calculation of our state_space_size later

        state_space_size = (MAXIMUM_INVENTORY**self.number_of_products+1) * \
                           (self.number_of_products*MAXIMUM_DELIVERY+1)**self.maximum_produce_time

        logging.log(f'state space has approximately {state_space_size} different states')
        if (SHOULD_CHECK_FOR_ADP_THRESHOLD and state_space_size > ADP_THRESHOLD):
            #adp part hasn't been checked yet
            logging.log('doing approximate dp analysis')
            self.adp = True
            inputs = tf.keras.Input(shape=((self.maximum_produce_time + 1) * self.number_of_products,), name='state')
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_3')(x)
            outputs = tf.keras.layers.Dense(1, activation='softmax', name='costs')(x)
            self.cost_approximator_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            logging.log('doing exact dp analysis')
            self.adp = False
            self.lookup_table = {}
            self.states = None

    def train(self):
        if (self.adp):
            self.train_adp()
        else:
            self.train_dp()

    def cost_of_state(self, state: np.ndarray):
        if (self.adp):
            return self.cost_approximator_model(state)
        else:
            return self.lookup_table.get(state.tobytes(), 0.)

    def train_dp(self):
        for step in reversed(range(self.horizon)):
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            self.train_exact_dp(orders, step)

    def train_exact_dp(self, orders, step):
        if self.states is None:
            self.states = self.get_state_variants()
        #TODO: check the value types
        return

    def get_state_variants(self):
        for inventory in self.get_inventory_variants():
            for last_step_delivered, delivery_matrix in self.recursively_create_all_delivery_matrixes():
                yield (last_step_delivered, np.append(inventory, delivery_matrix.flatten()))

    def get_inventory_variants(self) -> np.array:
        for number_of_products in range(self.number_of_products*MAXIMUM_INVENTORY+1):
            inventory_vector = np.zeros(self.number_of_products)
            remainder = number_of_products % MAXIMUM_INVENTORY
            quotient = int(number_of_products / MAXIMUM_INVENTORY)
            rest = self.number_of_products - quotient - 1
            chunks = [MAXIMUM_INVENTORY for _ in range(quotient)] + [remainder] + [0 for _ in range(rest)]
            for indx, chunk in enumerate(chunks):
                inventory_vector[indx] = chunk
            yield inventory_vector

    def recursively_create_all_delivery_matrixes(self):
        matrix_list = []
        def create_delivery_vector(vectors_before):
            for vector in self.get_delivery_variants():
                new_chain = vectors_before + [vector]
                if len(vectors_before) < self.maximum_produce_time-1:
                    create_delivery_vector(new_chain)
                else:
                    matrix_list.append(new_chain)

        create_delivery_vector([])
        refined_matrix_list = self.eliminate_faulty_matrices(matrix_list)

        return refined_matrix_list

    def eliminate_faulty_matrices(self, matrix_list):
        refined_list = []
        for matrix in matrix_list:
            if self.is_matrix_okay(matrix):
                refined_list.append(matrix)

        return refined_list

    def is_matrix_okay(self, matrix: np.ndarray):
        reversed_transposed_matrix = matrix[:,::-1].T
        for ind, column_in_original_matrix in enumerate(reversed_transposed_matrix):
            range_to_look_out = self.produce_times[np.argwhere(column_in_original_matrix > 0).item()]
            if(np.any(reversed_transposed_matrix[ind:ind+self.maximum_produce_time-range_to_look_out, :] > 0)):
                return False

        return True

    def get_optimal_cost_action_pair(self, state, orders, inventory_state, last_delivery_time_step, step):
        #TODO: inventory + last_deliveries are tuple, check all parts if they are always tuple (maybe in somepart concetanated np array)
        inventory, last_deliveries = state
        min_cost = sys.maxsize
        optimal_action = np.zeros(self.number_of_products)
        for prod_index, action_vector in self.get_delivery_variants():
            next_step_apprx_cost = 0

            if(prod_index == -1):
                action_vector = np.zeros(self.number_of_products)

            elif self.maximum_produce_time - last_delivery_time_step < self.produce_times[prod_index]:
                # do not produce anything if we are not able to do in time...
                action_vector = np.zeros(self.number_of_products)

            new_inventory = np.maximum(np.zeros_like(inventory), (inventory + action_vector - orders))
            new_state = np.append(new_inventory, np.append(last_deliveries[1:, :], action_vector))

            if step < self.horizon - 1:
                next_step_apprx_cost = self.cost_of_state(new_state)
            else:
                # we do not have a next step cost if we are at the end
                next_step_apprx_cost = 0

            cost = self.cost_func(inventory, action_vector, orders) + next_step_apprx_cost
            min_cost = cost if cost < min_cost else min_cost
            optimal_action = action_vector if cost < min_cost else optimal_action

        return min_cost, optimal_action

    # generate all possible actions
    def get_delivery_variants(self):
        yield (-1, 0)
        for prod_index in range(self.number_of_products):
            for count in range(MAXIMUM_DELIVERY):
                action_vector = np.zeros(self.number_of_products)
                action_vector[prod_index] = count + 1 # TODO:should I decrease this and include 0 ? or is it checked in get_optimal_cost_action_pair?
                yield (prod_index, action_vector) # TODO: prod index == 0 even if it is not produced ?

    def train_adp(self):
        for step in reversed(range(self.horizon)):
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            state_feature_matrix, cost_values = self.create_state_and_optimal_cost_pairs(orders, step, sample_size=SAMPLE_SIZE)
            self.train_neural_network(state_feature_matrix, cost_values)

    def train_neural_network(self, feature_matrix, y_values):
        #TODO: todo in this side
        train_dataset = tf.data.Dataset.from_tensor_slices((feature_matrix, y_values)).batch(BATCH_SIZE)
        self.func_approximator_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                           loss=tf.keras.losses.MeanSquaredError())
        self.func_approximator_model.fit(train_dataset, epochs=EPOCHS)

    def create_state_and_optimal_cost_pairs(self, orders, step, sample_size):
        state_feature_matrix = np.zeros((sample_size, (self.maximum_produce_time + 1) * self.number_of_products))
        cost_values = np.zeros(sample_size)
        for i in range(sample_size):
            random_inventory_state = np.random.randint(MAXIMUM_INVENTORY, size=self.number_of_products)
            random_delivery_matrix, last_delivery_time_step = self.create_random_delivery_matrix()
            state = (random_inventory_state, random_delivery_matrix)
            optimal_cost, _ = self.get_optimal_cost_action_pair(state, orders,random_inventory_state, last_delivery_time_step, step)
            state_feature_matrix[i, :] = np.append(random_inventory_state, random_delivery_matrix.flatten())
            cost_values[i] = optimal_cost

        return state_feature_matrix, cost_values

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


            no_delivery_prob = 1.0 / (1 + self.number_of_products) if NO_DELIVERY_PROB_IN_STATE_SPACE_SEARCH == -1 \
                else NO_DELIVERY_PROB_IN_STATE_SPACE_SEARCH

            should_deliver = np.random.choice([0, 1], p=[no_delivery_prob, 1.0 - no_delivery_prob])
            if should_deliver == np.array([0]): # should not deliver
                remaining_time_steps.remove(delivery_step)
                continue

            amount_of_delivery = np.random.randint(MAXIMUM_DELIVERY) + 1
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

    def cost_func(self, inventory, delivery, orders):
        return np.dot(self.delivery_cost_vector, delivery) + np.dot(self.store_cost_vector, inventory) + \
               np.dot(self.stock_out_cost_vector, np.maximum([0., 0.],orders - delivery - inventory))

    def act(self, observation):
        return


if __name__ == '__main__':
    agent = DPAgent(config_path="../inventory.yaml")
    agent.train()

#%%
import numpy as np


arr = np.array([1, 0 , 0 , 0])

ind = np.argwhere(arr>0).item()

print(ind)

arr111 = np.array([[1, 2] , [3 , 4]])

for ind, a in enumerate(arr111.T):
    print(ind)
    print(a)

acd = np.any(arr111>4)
print(acd)