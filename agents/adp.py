import sys
import numpy as np
from collections import Counter
#import tensorflow as tf
import yaml
from gym_baking.envs.consumers.parametric_consumer import PoissonConsumerModel as Oracle
import logging
from itertools import chain, product
from copy import deepcopy
import gym
from demand_models.ar_demand_predictor import AutoRegressiveDemandPredictor
import os
# for simplicity we assume all products have the same inventory and delivery limits
ADP_THRESHOLD = 1e6 # size of state space to switch adp when exceeded
SHOULD_CHECK_FOR_ADP_THRESHOLD = False # bypasses the threshold check and does "exact dp" if True# , otherwise trains the "adp"

# variables needed in case it is "adp"
SAMPLE_SIZE = 1024
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
NO_DELIVERY_PROB_IN_STATE_SPACE_SEARCH = 1e-1 # if this value is not -1, it creates actions without any delivery with
# this probability while creating random states to approximate the values

YAML = "inventory.yaml"
class DPAgent():
    def __init__(self, config_path, loglevel=logging.WARNING):
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        self.products = config['product_list']
        self.horizon = config['episode_max_steps']
        self.prediction = config['PREDICTION']
        self.items = self.products.items()
        self.produce_times = list(map(lambda x: x[1]["production_time"], self.items))
        self.maximum_produce_time = max(self.produce_times)
        self.number_of_products = len(self.items)
        self.maximum_inventory = config['maximum_inventory']
        self.maximum_delivery = config['maximum_delivery']
        self.oracle = Oracle(config) #behave the poisson model as it is the oracle that knows what orders will come
        # which are exactly the expectance of the order amounts -> Dimitry Bertsekas calls this "certainty equivalent control"
        # which is swapping the random variable with the expected value of it
        self.store_cost_vector = np.array(list(map(lambda x: x[1]["storing_cost"], self.items)))
        self.delivery_cost_vector = np.array(list(map(lambda x: x[1]["delivery_cost"], self.items)))
        self.stock_out_cost_vector = np.array(list(map(lambda x: x[1]["stock_out_cost"], self.items)))
        #logging.basicConfig(filename="example.log", level=loglevel)
        logging.basicConfig(level=loglevel)

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

        state_space_size = ((self.maximum_inventory+1)**self.number_of_products) * \
                           (self.number_of_products*self.maximum_delivery+1)**(self.maximum_produce_time-1)

        self.is_trained = False
        logging.info(f'state space has approximately {state_space_size} different states')
        if (SHOULD_CHECK_FOR_ADP_THRESHOLD and state_space_size > ADP_THRESHOLD):
            #adp part hasn't been checked yet
            logging.info('doing approximate dp analysis')
            self.adp = True
            #inputs = tf.keras.Input(shape=((self.maximum_produce_time) * self.number_of_products,), name='state')
            #x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
            #x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
            #x = tf.keras.layers.Dense(64, activation='relu', name='dense_3')(x)
            #outputs = tf.keras.layers.Dense(1, activation='softmax', name='costs')(x)
            #self.cost_approximator_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            logging.info('doing exact dp analysis')
            self.adp = False
            self.lookup_table = {}
            self.states = []
            self.orders = []
            self._injected_orders = [] # used for development but can be used also later as a convenient way to give orders
            self.optimal_actions = [{} for _ in range(self.horizon)]
            self.start_state = None

    def refresh(self, timestep):
        self.lookup_table = {}
        self.orders = []
        self._injected_orders = []
        self.start_state = None

    def train(self, start_step=0, start_inventory=None, last_delivered_step=None, env_not_used=True):
        if (self.adp):
            self.train_adp()
        else:
            self.train_dp(env_not_used, start_step=start_step, start_inventory=start_inventory, start_last_delivery_step=last_delivered_step)

    def get_next_action_and_inv(self, print_meanwhile=False):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products+1))

        first_inventory = min_state[:-1]
        if print_meanwhile:
            logging.critical(f'start inv \n {first_inventory}')

        if self.start_state is not None:
            min_state = self.start_state

        total_cost = 0
        returned_act = np.zeros(self.number_of_products, dtype='int64')
        for step in range(self.horizon):
            inventory, last_delivery_step = min_state[:-1], min_state[-1]
            action_table = self.optimal_actions[step]
            action = action_table[min_state.tobytes()]
            if np.any(action > 0):
                range_to_look_out = self.get_produce_time_of_delivery(action)
                if step-range_to_look_out is 0:
                    returned_act = action
                    if not print_meanwhile:
                        return returned_act, first_inventory

            order = None
            if len(self._injected_orders) == 0:
                order = self.make_orders(step, train=False)
            else:
                order = self.vectorize_order(self.make_orders(step, train=False))

            if print_meanwhile:
                logging.critical(f'order \n {order}')
                logging.critical(f'delivery \n {action}')

            cost, act, min_state = self.take_action(min_state, order, last_delivery_step, action, step)

        if print_meanwhile:
            logging.critical(f'returned_act \n {returned_act}')
        return returned_act, first_inventory

    def get_next_action_and_score(self, print_meanwhile=False):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products+1))

        first_inventory, last_delivery = min_state[:-1], min_state[-1]
        if self.start_state is not None:
            min_state = self.start_state

        total_cost = 0
        returned_act = np.zeros(self.number_of_products, dtype='int64')
        for step in range(self.horizon):
            inventory, last_delivery_step = min_state[:-1], min_state[-1]
            action_table = self.optimal_actions[step]
            action = action_table[min_state.tobytes()]
            if np.any(action > 0):
                range_to_look_out = self.get_produce_time_of_delivery(action)
                if step-range_to_look_out is 0:
                    returned_act = action
            if print_meanwhile:
                logging.critical(f'step {step}')
                logging.critical(f'inventory \n {inventory}')
            order = None
            if len(self._injected_orders) == 0:
                order = self.make_orders(step, train=False)
            else:
                order = self.vectorize_order(self.make_orders(step, train=False))

            if print_meanwhile:
                logging.critical(f'order \n {order}')
                logging.critical(f'delivery \n {action}')

            cost, act, min_state = self.take_action(min_state, order, 0, action, step)
            total_cost += cost
        if print_meanwhile:
            logging.critical(f'cost of this solution is {total_cost}')
            logging.critical(f'produce order of now is {returned_act}')
        return returned_act, total_cost, first_inventory


    def decrypt_dic_with_np_state_keys(self, lookup_table, log=False):
        temp_dic = []
        for item in lookup_table.items():
            state = np.reshape(np.frombuffer(item[0], dtype="int64"),
                                (self.number_of_products+1))
            value = item[1]
            if not log:
                element = (state, value)
                temp_dic.append(element)
            else:
                logging.info(f" {state} : {value}")

        if not log:
            return temp_dic

    def decrypt_actions_table(self, actions_list, log=False):
        actions = []
        for ind, table in enumerate(actions_list):
            if not log:
                actions.append(self.decrypt_dic_with_np_state_keys(table, log))
            if log:
                logging.info(f'table at {ind} : {self.decrypt_dic_with_np_state_keys(table, log)}')

        if not log:
            return actions

    def index_of_product_type(self, product):
        for key, val in self.products.items():
            if val["type"] == product:
                return key


    def get_inv_from_inventory_state(self, state):
        return self.vectorize_order(["dummyval", [self.index_of_product_type(product._item_type) for product in state["products"]]])

    def train_with_env(self, seed=None, test_seed=None):
        env = gym.make('gym_baking:Inventory-v0', config_path=YAML)
        env._consumer_model.fix_seed(seed)
        predictor = AutoRegressiveDemandPredictor(config_path=YAML, steps=self.horizon, days=10, bins_size=1, model_path="../saved_models")
        for episode in range(1):
            observation = env.reset()
            reward = 0
            done = False
            last_delivery = -1
            start_inventory = None
            # just for now
            #horizon = len(self.prediction)
            curr_data = [0 for _ in range(self.number_of_products)]
            for timestep in range(self.horizon):
                #env.render()
                #prediction = self.pretend_oracle(last_orders=last_delivery, time_step=timestep)

                if test_seed:
                    env._consumer_model.fix_seed(test_seed)
                    env._consumer_model.is_overriden = False
                    test_samples = [self.vectorize_order(tuple) for tuple in
                                    env._consumer_model.give_all_samples(test_seed)]
                    self.inject_prediction(test_samples[-(self.horizon - timestep):])
                else:
                    prediction_matrix = [[0 for _ in range(self.number_of_products)] for i in range(self.horizon)]
                    for i in range(self.number_of_products):
                        prediction = predictor.predict(curr_data=np.array(curr_data[i]).reshape(1, 1), pred_steps=self.horizon-timestep, item=i)
                        for ind_p, p in enumerate(prediction):
                            prediction_matrix[ind_p][i] = p

                    self.refresh(timestep)
                    self.inject_prediction(prediction_matrix)

                self.train(start_step=timestep, start_inventory=start_inventory, last_delivered_step=last_delivery,
                           env_not_used=False)
                act, inv = self.get_next_action_and_inv()

                if timestep is 0:
                    env.add_deliveries(inv)
                    #print(f' inv \n {inv}')

                action = np.zeros(2, dtype="int64")
                if np.any(act > 0):
                    action[0], action[1] = np.argwhere(act > 0).item(), act[act > 0]

                observation, reward, done, info, did_deliver, type_ids = env.step(action)
                curr_data = self.vectorize_order(("dummyval",type_ids))
                print(curr_data)
                inventory_state = observation["inventory_state"]
                start_inventory = self.get_inv_from_inventory_state(inventory_state)

                last_delivery = timestep if did_deliver else last_delivery

                #print(f' observation \n {observation}')
                #print(f' act \n {act}')
                #print(f' inventory after order \n {start_inventory}')
                #print(f' last_deliveries \n {last_deliveries}')
                s, info = env._metric.get_metric(state_history=env.state_history, done=True, step=timestep)
                print(f'timestep {timestep}')
                print(f'score: {s} and \n info {info}')
                #print(f'{timestep}')
                if done:
                    #print('Episode finished after {} timesteps'.format(timestep))
                    break

            env.close()
            return [s, info]

    def train_dp(self, env_not_used=True, start_step=0, start_inventory=None, start_last_delivery_step=None):
        if len(self.states) == 0:
            self.create_state_variants() #stores all state variants in self.states

        #commented out when train_with_env is used
        range_ = self.horizon
        if env_not_used:
            if self._injected_orders:
                self.inject_prediction(self.vectorize_orders(self._injected_orders)[-(self.horizon - start_step):])

            range_ = self.horizon-start_step if not self._injected_orders else self.horizon

        for step in reversed(range(range_)):
            temp_look_up_table = deepcopy(self.lookup_table)
            logging.info(f"************RIGHT NOW LOOK UP TABLE **************")
            self.decrypt_dic_with_np_state_keys(temp_look_up_table, log=True)

            orders = self.vectorize_order(self.make_orders(step))
            self.orders.append(orders) #we store the orders to get the same orders in the forward pass
            self.train_exact_dp(orders, step, temp_look_up_table, start_inventory, start_last_delivery_step, start_step)
        #after training the orders will be processed in the normal order
        self.orders = list(reversed(self.orders))

    def train_exact_dp(self, orders, step, look_up, start_inventory, start_last_delivery_step, start_step):
        optimal_action_table = {}
        for (last_step_delivered, inventory) in self.states:
            if step is 0 and start_inventory is not None:
                inventory = start_inventory
                diff = start_step-start_last_delivery_step if start_last_delivery_step is not -1 else None
                last_step_delivered = 0 if diff is None or diff >= self.maximum_produce_time else self.maximum_produce_time - diff

            cost, optimal_action, _ = self.get_optimal_cost_action_pair_and_new_state(inventory, orders,
                                                                     last_step_delivered, step, look_up, start_inventory, start_step)
            if self.maximum_produce_time > 1:
                key = np.array(np.append(inventory, last_step_delivered), dtype="int64")
            else:
                key = np.array(inventory, dtype="int64")

            if step is 0 and start_inventory is not None:
                self.start_state = key

            self.lookup_table[key.tobytes()] = cost
            self.optimal_actions[step][key.tobytes()] = optimal_action

        return

    def pretend_oracle(self, last_orders, time_step):
        return self.prediction[time_step:]

    def train_adp(self):
        for step in reversed(range(self.horizon)):
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            state_feature_matrix, cost_values = self.create_state_and_optimal_cost_pairs(orders, step, sample_size=SAMPLE_SIZE)
            self.train_neural_network(state_feature_matrix, cost_values)

    # def train_neural_network(self, feature_matrix, y_values):
       # TODO: todo in this side
       # train_dataset = tf.data.Dataset.from_tensor_slices((feature_matrix, y_values)).batch(BATCH_SIZE)
       # self.func_approximator_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
       #                    loss=tf.keras.losses.MeanSquaredError())
       # self.func_approximator_model.fit(train_dataset, epochs=EPOCHS)

    def make_orders(self, step, train=True):
        if len(self._injected_orders) == 0:
            if train:
                return self.oracle.make_orders(step)
            else:
                return self.orders[step]
        else:
            return self._injected_orders[step]

    def create_state_and_optimal_cost_pairs(self, orders, step, sample_size):
        state_feature_matrix = np.zeros((sample_size, (self.maximum_produce_time + 1) * self.number_of_products), dtype="int64")
        cost_values = np.zeros(sample_size)
        for i in range(sample_size):
            random_inventory_state = np.random.randint(self.maximum_inventory, size=self.number_of_products)
            random_delivery_matrix, last_delivery_time_step = self.create_random_delivery_matrix()
            state = (random_inventory_state, random_delivery_matrix)
            optimal_cost, _, _ = self.get_optimal_cost_action_pair_and_new_state(state, orders, last_delivery_time_step, step)
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

            amount_of_delivery = np.random.randint(self.maximum_delivery) + 1
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

    def create_state_variants(self):
        if self.maximum_produce_time > 1:
            for inventory in self.get_inventory_variants():
                for last_step_delivered in range(self.maximum_produce_time):
                    self.states.append((last_step_delivered, inventory))
        else:
            for inventory in self.get_inventory_variants():
                self.states.append((None, (inventory, None)))

    def get_inventory_variants(self) -> np.array:
        range_of_inv = list(range(self.maximum_inventory+1))
        combinations = product(*[range_of_inv for _ in range(self.number_of_products)])
        for combination in combinations:
            yield np.array(combination)

    def create_all_possible_delivery_matrixes(self):
        matrix_list = []
        def inject_delivery_vectors_recursively(matrix: np.ndarray, ind=0):
            for (prod_index, vector) in self.get_delivery_variants():
                matrix_copy = matrix.copy()
                matrix_copy[..., ind] = vector
                if ind < self.maximum_produce_time-2:
                    index = ind + 1
                    inject_delivery_vectors_recursively(matrix_copy, index)
                else:
                    matrix_list.append(matrix_copy)

        inject_delivery_vectors_recursively(np.zeros((self.number_of_products, self.maximum_produce_time-1), dtype="int64"))
        return self.eliminate_faulty_matrices(matrix_list)

    def eliminate_faulty_matrices(self, matrix_list):
        refined_list = []
        for matrix in matrix_list:
            last_step_delivered, matrix_is_okay =  self.check_matrix(matrix)
            if matrix_is_okay:
                refined_list.append((last_step_delivered,matrix))
        return refined_list

    def get_produce_time_of_delivery(self, delivery):
        return self.produce_times[np.argwhere(delivery > 0).item()]

    def check_matrix(self, matrix):
        #we check the deliverys in a reversed order, it makes it easier
        #if there is an delivery, there shouldn't be any delivery during the time it took to produce that delivery
        last_step_delivered_for_the_matrix = 0
        reversed_transposed_matrix = matrix[:,::-1]
        for ind, column_in_original_matrix in enumerate(reversed_transposed_matrix.T):
            if not np.any(column_in_original_matrix > 0):
                continue

            index = self.maximum_produce_time-1-ind
            last_step_delivered_for_the_matrix = index if index > last_step_delivered_for_the_matrix else last_step_delivered_for_the_matrix

            range_to_look_out = self.get_produce_time_of_delivery(column_in_original_matrix)-1
            if range_to_look_out == 0:
                continue
            look_up_limit = ind+1 + min(self.maximum_produce_time-1-ind-1, range_to_look_out)
            look_window = reversed_transposed_matrix[:,ind+1:look_up_limit]
            if(np.any(look_window > 0)):
                return (last_step_delivered_for_the_matrix, False)

        last_step_delivered_for_the_matrix = 0 if last_step_delivered_for_the_matrix is None else last_step_delivered_for_the_matrix
        return (last_step_delivered_for_the_matrix, True)

    def get_optimal_cost_action_pair_and_new_state(self, state, orders, last_delivery_time_step, step, look_up, forward=False, start_step=0):
        inventory = state
        min_cost = sys.maxsize
        optimal_action = np.zeros(self.number_of_products, dtype="int64")
        new_state = np.zeros(self.number_of_products+1)

        if self.maximum_produce_time > 1:
            for prod_index, action_vector in self.get_delivery_variants():
                act = action_vector
                #last_delivery_time_step = 0 if last_delivery_time_step < (self.maximum_produce_time-1) - step else last_delivery_time_step
                if start_step+step < self.produce_times[prod_index]:
                    act = np.zeros(self.number_of_products, dtype="int64")

                if(prod_index == -1):
                    act = np.zeros(self.number_of_products, dtype="int64")
                elif self.maximum_produce_time - last_delivery_time_step < self.produce_times[prod_index]:
                    # do not produce anything if we are not able to do in time...
                    act = np.zeros(self.number_of_products, dtype="int64")

                if np.any(act > 0):
                    last_delivery_time_step_temp = last_delivery_time_step if last_delivery_time_step > 0 else last_delivery_time_step+1
                else:
                    last_delivery_time_step_temp = last_delivery_time_step - 1 if last_delivery_time_step > 0 else 0

                new_inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
                new_inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64")*self.maximum_inventory, new_inventory_temp)

                new_state_temp = np.array(np.append(new_inventory_temp, last_delivery_time_step_temp), dtype="int64")

                logging.info(f"step : {step}")
                logging.info(f"investigated state inventory \n : {inventory}")
                logging.info(f"order : {orders}")
                logging.info(f"chosen action : {act}")

                cost = self.cost_func(inventory, act, orders)
                logging.info(f"current state cost : {cost}")

                if step < self.horizon-1 or forward is True:
                    next_step_cost = self.cost_of_state(new_state_temp, look_up)
                    logging.info(f"next state cost : {next_step_cost}")
                    cost +=  next_step_cost

                logging.info(f"now total cost : {cost}")
                logging.info(f"old cost this : {min_cost}")

                logging.info(f"next inventory : \n {new_inventory_temp}")
                logging.info(f"next deliveries : \n {new_state_temp}")

                if cost < min_cost:
                    logging.info(f"----------------------------SEÇİLDİ----------------------------")
                    new_state = new_state_temp
                    optimal_action = act
                    min_cost = cost

            return min_cost, optimal_action, new_state # TODO: new state is not tuple but the parameter state is tuple, they should be same
        else:
            for prod_index, action_vector in self.get_delivery_variants():
                inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + action_vector - orders))
                inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64") * self.maximum_inventory,
                                                inventory_temp)
                next_step_cost = self.cost_of_state(inventory_temp, look_up)
                cost_of_this_state = self.cost_func(inventory, action_vector, orders)
                cost = cost_of_this_state + next_step_cost

                if cost < min_cost:
                    new_state = inventory_temp
                    optimal_action = action_vector
                    min_cost = cost

            return min_cost, optimal_action, new_state

    def get_last_delivery_step_of_delivery_matrix(self, delivery_matrix):
        last_step_delivered_for_the_matrix = 0

        if np.shape(delivery_matrix)[-1] is 1:
            if np.any(delivery_matrix > 0):
                return 1
            return 0

        for ind, column_in_original_matrix in enumerate(delivery_matrix.T):
            if not np.any(column_in_original_matrix > 0):
                continue
            last_step_delivered_for_the_matrix = ind+1 if ind+1 > last_step_delivered_for_the_matrix else last_step_delivered_for_the_matrix

        return last_step_delivered_for_the_matrix

    # generate all possible actions
    def get_delivery_variants(self):
        yield (-1, np.zeros(self.number_of_products, dtype="int64"))
        for prod_index in range(self.number_of_products):
            for count in range(self.maximum_delivery):
                action_vector = np.zeros(self.number_of_products, dtype="int64")
                action_vector[prod_index] = count + 1
                yield (prod_index, action_vector)


    def cost_of_state(self, state: np.ndarray, look_up):
        if (self.adp):
            return self.cost_approximator_model(state)
        else:
            return look_up.get(state.tobytes(), 0.)

    def cost_func(self, inventory, delivery, orders):
        stock_out = np.maximum(np.zeros(self.number_of_products, dtype="float32"),orders - delivery - inventory)
        logging.info('loss calculation')
        logging.info(f'orders \n {orders}')
        logging.info(f'delivery \n {delivery}')
        logging.info(f'inventory \n {inventory}')
        logging.info(f' stock out np.maximum([0., 0.],orders - delivery - inventory) = {stock_out}')
        logging.info(f' {self.delivery_cost_vector} dotted by {delivery} == {np.dot(self.delivery_cost_vector, delivery)} ')
        logging.info(f' {self.store_cost_vector} dotted by {inventory} == {np.dot(self.delivery_cost_vector, inventory)} ')
        logging.info(f' {self.stock_out_cost_vector} dotted by {stock_out} == {np.dot(self.stock_out_cost_vector, stock_out)} ')
        return np.dot(self.delivery_cost_vector, delivery) + np.dot(self.store_cost_vector, inventory) + \
               np.dot(self.stock_out_cost_vector, stock_out)

    def vectorize_orders(self, orders):
        vectorized_orders = []
        for order in orders:
            vectorized_orders.append(np.array(self.vectorize_order(order), dtype="int64").tolist())
        return vectorized_orders

    def vectorize_order(self, order):
        counter_obj = Counter(order[1])
        counts = self.vectorize_counter_np(counter_obj)
        return counts

    def vectorize_counter_np(self, counter):
        counts = np.zeros(self.number_of_products, dtype="int64")
        for indx, count in  counter.items():
            counts[indx] = count
        return counts

    """def act(self, step):
        for step in range(self.horizon):
            min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"), (self.number_of_products, self.maximum_produce_time))
            inventory, delivery_matrix = min_state[:, 0], min_state[:, 1:]
            last_step_delivered = self.get_last_delivery_step_of_delivery_matrix(delivery_matrix)
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            cost, optimal_action, next_state = self.get_optimal_cost_action_pair_and_new_state((inventory, delivery_matrix),
                                                                                      orders,
                                                                                      last_step_delivered)
            print(optimal_action)"""

    def get_starting_point(self):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products+1))
        return min_state

    def non_vectorize_order(self, order):
        return (sum(order), [index for index, item in enumerate(order) for _ in range(item)])

    def nonvectorize_orders(self, orders):
        return [self.non_vectorize_order(order) for order in orders]

    def inject_prediction(self, orders):
        orders_non_vectorized = self.nonvectorize_orders(orders)
        self._injected_orders = orders_non_vectorized
        self.horizon = len(orders)
        self.optimal_actions = [{} for _ in range(self.horizon)]

    # this method is for manually checking any action's cost
    def take_action(self, state, orders, last_delivery_time_step, action, step):
        inventory, last_delivery_step = state[:-1], state[-1]

        act = np.array(action, dtype="int64")
        if (self.maximum_produce_time > 1):
            if np.any(act > 0):
                prod_time = self.produce_times[np.argwhere(act > 0).item()]
                if (self.maximum_produce_time - last_delivery_time_step < prod_time):
                    # do not produce anything if we are not able to do in time...
                    act = np.zeros(self.number_of_products, dtype="int64")

            new_inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
            new_inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64") * self.maximum_inventory,
                                            new_inventory_temp)

            last_delivery_time_step = last_delivery_time_step if np.any(act > 0) else last_delivery_time_step - 1 \
                if last_delivery_time_step > 0 else 0

            new_state = np.array(np.append(new_inventory_temp,last_delivery_time_step), dtype="int64")

            logging.error(f"step : {step}")
            logging.error(f"investigated state inventory \n : {inventory}")
            logging.error(f"order : {orders}")
            logging.error(f"chosen action : {act}")

            cost_of_this_state = self.cost_func(inventory, act, orders)
            cost = cost_of_this_state

            logging.error(f"current state cost : {cost_of_this_state}")

            logging.error(f"next inventory : \n {new_inventory_temp}")
            logging.error(f"next state : \n {new_state}")

            return cost, act, new_state

        else:
            new_state_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
            new_state = np.minimum(np.ones_like(new_state_temp, dtype="int64") * self.maximum_inventory,
                                   new_state_temp)

            cost_of_this_state = self.cost_func(inventory, act, orders)

            return cost_of_this_state, act, new_state

    # this function is for manually returning the total cost of actions given
    def cost_of_actions(self, actions):
        state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                           (self.number_of_products+1))
        logging.error(f'started from the state {state}')
        total_cost = 0
        for step, action in enumerate(actions):
            if (self.maximum_produce_time > 1):
                inventory, delivery_matrix = state[:, 0], state[:, 1:]
                logging.error(f'step {step}')
                logging.error(f'inventory \n {inventory}')

                order = self.vectorize_order(self.make_orders(step, train=False))
                logging.error(f'order \n {order}')

                last_step_delivered = self.get_last_delivery_step_of_delivery_matrix(delivery_matrix)

                cost, optimal_action, state = self.take_action(
                    (inventory, delivery_matrix),
                    order,
                    last_step_delivered, action, step, self.lookup_table)
                logging.error(f'next state {state}')
                total_cost += cost
            else:
                inventory = state[:, 0]
                order = self.vectorize_order(self.make_orders(step, train=False))

                cost, optimal_action, state = self.take_action(
                    (inventory, _),
                    order,
                    None, action, step, self.lookup_table)
                total_cost += cost

        return total_cost



    def print(self):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products+1))

        for step in range(self.horizon):

            if self.maximum_produce_time > 1:
                inventory, delivery_matrix = min_state[:-1], min_state[-1]
                logging.error(f'step {step}')

                logging.error(f'inventory \n {inventory}')
                logging.error(f'delivery \n {delivery_matrix}')

                #print(f'min_state delivery_matrix \n {delivery_matrix}')

                order = None
                if len(self._injected_orders) == 0:
                    order = self.make_orders(step, train=False)
                else:
                    order = self.vectorize_order(self.make_orders(step, train=False))

                logging.error(f'order \n {order}')

                last_step_delivered = self.get_last_delivery_step_of_delivery_matrix(delivery_matrix)
                cost, optimal_action, min_state = self.get_optimal_cost_action_pair_and_new_state((inventory, delivery_matrix),
                                                                                          order,
                                                                                          last_step_delivered, step, self.lookup_table, forward=True)
                logging.error(f'optimal action \n {optimal_action}')

            else:
                inventory = min_state
                logging.error(f'step {step}')

                logging.error(f'inventory \n {inventory}')
                if len(self._injected_orders) == 0:
                    order = self.make_orders(step, train=False)
                else:
                    order = self.vectorize_order(self.make_orders(step, train=False))

                logging.error(f'order \n {order}')

                cost, optimal_action, min_state = self.get_optimal_cost_action_pair_and_new_state(
                    (inventory, None),
                    order,
                    None, step, self.lookup_table)

if __name__ == '__main__':
    agent = DPAgent(config_path=YAML, loglevel=logging.CRITICAL)
    #agent.train_with_env(0)
    #injection = [[1,0],[2,0]] # order 1 of first product in first time step, 2 of first product in second time step
    #injection = [[2,0],[0,0], [1,0], [2,0]] # 1 order of first product in first time step, 1 of first product in third time step and 2 of first product in last step
    #injection_long = [[1, 0], [0, 0], [1, 0], [2,5], [2,5], [0,0], [7,1], [2,5], [2,5], [0,0], [7,1], [1, 0], [0, 0], [1, 0], [2,5], [2,5], [0,0]]
    #agent.inject_prediction(injection)
    # first we train from beginning
    #agent.train()
    # and then when we give the state of the first step in the optimal solution as below it gives the correct produce commend
    #agent.train(start_step=1, start_inventory=np.array([1, 2], dtype="int64"), last_deliveries= np.array([0,0], dtype="int64"))
    #cost = agent.cost_of_actions([[2, 0], [0, 0], [1, 0], [0, 0]])

#%%
#agent.get_next_action_and_inv(print_meanwhile=True)
agent.train_with_env(test_seed=11)
#print("------------print finishe------------")
#cost = agent.cost_of_actions([])
#print(f'total cost: {cost}')
