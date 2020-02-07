import sys
import numpy as np
from collections import Counter
import tensorflow as tf
import yaml
from gym_baking.envs.consumers.parametric_consumer import PoissonConsumerModel as Oracle
import logging
from itertools import chain, product
from copy import deepcopy

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
    def __init__(self, config_path, loglevel=logging.WARNING):

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        self.products = config['product_list']
        self.horizon = config['episode_max_steps']
        self.seeds = config['seeds']
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

        state_space_size = ((MAXIMUM_INVENTORY+1)**self.number_of_products) * \
                           (self.number_of_products*MAXIMUM_DELIVERY+1)**(self.maximum_produce_time-1)

        self.is_trained = False
        logging.info(f'state space has approximately {state_space_size} different states')
        if (SHOULD_CHECK_FOR_ADP_THRESHOLD and state_space_size > ADP_THRESHOLD):
            #adp part hasn't been checked yet
            logging.info('doing approximate dp analysis')
            self.adp = True
            inputs = tf.keras.Input(shape=((self.maximum_produce_time) * self.number_of_products,), name='state')
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
            x = tf.keras.layers.Dense(64, activation='relu', name='dense_3')(x)
            outputs = tf.keras.layers.Dense(1, activation='softmax', name='costs')(x)
            self.cost_approximator_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            logging.info('doing exact dp analysis')
            self.adp = False
            self.lookup_table = {}
            self.states = []
            self.orders = []
            self._injected_orders = []
            self.optimal_actions = [{} for _ in range(self.horizon)]

    def train(self):
        if (self.adp):
            self.train_adp()
        else:
            self.train_dp()

    def decrypt_dic_with_np_state_keys(self, lookup_table, log=False):
        temp_dic = []
        for item in lookup_table.items():
            status = np.reshape(np.frombuffer(item[0], dtype="int64"),
                                (self.number_of_products, self.maximum_produce_time))
            result = item[1]
            if not log:
                element = (status, result)
                temp_dic.append(element)
            else:
                logging.info(f" {status} : {result}")

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

    def train_dp(self):
        if len(self.states) == 0:
            self.create_state_variants() #stores states in self.states
        for step in reversed(range(self.horizon)):
            temp_look_up_table = deepcopy(self.lookup_table)
            logging.info(f"************RIGHT NOW LOOK UP TABLE **************")
            self.decrypt_dic_with_np_state_keys(temp_look_up_table, log=True)

            orders = self.vectorize_order(self.make_orders(step))
            self.orders.append(orders) #we store the orders to get the same orders in the forward pass
            self.train_exact_dp(orders, step, temp_look_up_table)
        #after training the orders will be processed in the normal order
        self.orders = list(reversed(self.orders))

    def train_exact_dp(self, orders, step, look_up):
        optimal_action_table = {}
        for (last_step_delivered, (inventory, delivery_matrix)) in self.states:
            cost, optimal_action, _ = self.get_optimal_cost_action_pair_and_new_state((inventory,delivery_matrix), orders,
                                                                     last_step_delivered, step, look_up)
            if self.maximum_produce_time > 1:
                key = np.array(np.append(inventory[:, np.newaxis], delivery_matrix, axis=1), dtype="int64")
            else:
                key = np.array(inventory, dtype="int64")

            self.lookup_table[key.tobytes()] = cost
            self.optimal_actions[step][key.tobytes()] = optimal_action

        return


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
            random_inventory_state = np.random.randint(MAXIMUM_INVENTORY, size=self.number_of_products)
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

    def create_state_variants(self):
        if self.maximum_produce_time > 1:
            for inventory in self.get_inventory_variants():
                for (last_step_delivered, delivery_matrix) in iter(self.create_all_possible_delivery_matrixes()):
                    self.states.append((last_step_delivered, (inventory, delivery_matrix)))
        else:
            for inventory in self.get_inventory_variants():
                self.states.append((None, (inventory, None)))

    def get_inventory_variants(self) -> np.array:
        range_of_inv = list(range(MAXIMUM_INVENTORY+1))
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

            range_to_look_out = self.produce_times[np.argwhere(column_in_original_matrix > 0).item()]-1
            if range_to_look_out == 0:
                continue
            look_up_limit = ind+1 + min(self.maximum_produce_time-1-ind-1, range_to_look_out)
            look_window = reversed_transposed_matrix[:,ind+1:look_up_limit]
            if(np.any(look_window > 0)):
                return (last_step_delivered_for_the_matrix, False)

        last_step_delivered_for_the_matrix = 0 if last_step_delivered_for_the_matrix is None else last_step_delivered_for_the_matrix
        return (last_step_delivered_for_the_matrix, True)

    def get_optimal_cost_action_pair_and_new_state(self, state, orders, last_delivery_time_step, step, look_up, forward=False):
        inventory, last_deliveries = state
        min_cost = sys.maxsize
        optimal_action = np.zeros(self.number_of_products, dtype="int64")
        new_state = np.zeros((self.number_of_products, self.maximum_produce_time))

        if self.maximum_produce_time > 1:
            for prod_index, action_vector in self.get_delivery_variants():
                act = action_vector
                #last_delivery_time_step = 0 if last_delivery_time_step < (self.maximum_produce_time-1) - step else last_delivery_time_step
                if(prod_index == -1):
                    act = np.zeros(self.number_of_products, dtype="int64")
                elif self.maximum_produce_time - last_delivery_time_step < self.produce_times[prod_index]:
                    # do not produce anything if we are not able to do in time...
                    act = np.zeros(self.number_of_products, dtype="int64")
                elif step < self.produce_times[prod_index]:
                    act = np.zeros(self.number_of_products, dtype="int64")


                new_inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
                new_inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64")*MAXIMUM_INVENTORY, new_inventory_temp)
                new_state_temp = np.array(np.append(new_inventory_temp[:, np.newaxis], np.append(last_deliveries[:, 1:], act[:, np.newaxis], axis=1), axis=1), dtype="int64")

                logging.info(f"step : {step}")
                logging.info(f"investigated state inventory \n : {inventory}")
                logging.info(f"order : {orders}")
                logging.info(f"investigated state deliveries \n : {last_deliveries}")
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

            return min_cost, optimal_action, new_state
        else:
            for prod_index, action_vector in self.get_delivery_variants():
                inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + action_vector - orders))
                inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64") * MAXIMUM_INVENTORY,
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
        for ind, column_in_original_matrix in enumerate(delivery_matrix.T):
            if not np.any(column_in_original_matrix > 0):
                continue
            last_step_delivered_for_the_matrix = ind+1 if ind+1 > last_step_delivered_for_the_matrix else last_step_delivered_for_the_matrix

        return last_step_delivered_for_the_matrix

    # generate all possible actions
    def get_delivery_variants(self):
        yield (-1, np.zeros(self.number_of_products, dtype="int64"))
        for prod_index in range(self.number_of_products):
            for count in range(MAXIMUM_DELIVERY):
                action_vector = np.zeros(self.number_of_products, dtype="int64")
                action_vector[prod_index] = count + 1
                yield (prod_index, action_vector)


    def cost_of_state(self, state: np.ndarray, look_up):
        if (self.adp):
            return self.cost_approximator_model(state)
        else:
            return look_up.get(state.tobytes(), 0.)

    def cost_func(self, inventory, delivery, orders):
        stock_out = np.maximum([0., 0.],orders - delivery - inventory)
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

    def vectorize_order(self, order):
        counter_obj = Counter(order[1])
        counts = np.zeros(self.number_of_products)
        for indx, count in counter_obj.items():
            counts[indx] = count
        return counts

    def act(self, observation):
        return

    def act(self, step):
        for step in range(self.horizon):
            min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"), (self.number_of_products, self.maximum_produce_time))
            inventory, delivery_matrix = min_state[:, 0], min_state[:, 1:]
            last_step_delivered = self.get_last_delivery_step_of_delivery_matrix(delivery_matrix)
            orders = self.vectorize_counter(Counter(self.oracle.make_orders(step)[1]))
            cost, optimal_action, next_state = self.get_optimal_cost_action_pair_and_new_state((inventory, delivery_matrix),
                                                                                      orders,
                                                                                      last_step_delivered)
            print(optimal_action)

    def inject_prophecy(self, orders):
        orders_non_vectorized = [(sum(order), [index for index, item in enumerate(order) for _ in range(item)]) for order in orders]
        self._injected_orders = orders_non_vectorized
        self.horizon = len(orders)
        self.optimal_actions = [{} for _ in range(self.horizon)]

    # this method is for manually checking any action's cost
    def take_action(self, state, orders, last_delivery_time_step, action, step):
        inventory, last_deliveries = state[:, 0], state[:, 1:]

        act = np.array(action, dtype="int64")
        if (self.maximum_produce_time > 1):
            if np.any(act > 0):
                prod_time = self.produce_times[np.argwhere(act > 0).item()]
                if (self.maximum_produce_time - last_delivery_time_step < prod_time):
                    # do not produce anything if we are not able to do in time...
                    act = np.zeros(self.number_of_products, dtype="int64")

            new_inventory_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
            new_inventory_temp = np.minimum(np.ones_like(inventory, dtype="int64") * MAXIMUM_INVENTORY,
                                            new_inventory_temp)

            if np.shape(last_deliveries)[1] > 1:
                new_state = np.array(np.append(new_inventory_temp[:, np.newaxis],
                                      np.append(last_deliveries[:, np.newaxis][:, 1:], act[:, np.newaxis], axis=1), axis=1), dtype="int64")
            else:
                new_state = np.array(np.append(new_inventory_temp[:, np.newaxis],act[:, np.newaxis],axis=1), dtype="int64")

            logging.error(f"step : {step}")
            logging.error(f"investigated state inventory \n : {inventory}")
            logging.error(f"order : {orders}")
            logging.error(f"investigated state deliveries \n : {last_deliveries}")
            logging.error(f"chosen action : {act}")

            cost_of_this_state = self.cost_func(inventory, act, orders)
            cost = cost_of_this_state

            logging.error(f"current state cost : {cost_of_this_state}")

            logging.error(f"next inventory : \n {new_inventory_temp}")
            logging.error(f"next state : \n {new_state}")

            return cost, act, new_state

        else:
            new_state_temp = np.maximum(np.zeros_like(inventory, dtype="int64"), (inventory + act - orders))
            new_state = np.minimum(np.ones_like(new_state_temp, dtype="int64") * MAXIMUM_INVENTORY,
                                   new_state_temp)

            cost_of_this_state = self.cost_func(inventory, act, orders)

            return cost_of_this_state, act, new_state

    # this function is for manually returning the total cost of actions given
    def cost_of_actions(self, actions):
        state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                           (self.number_of_products, self.maximum_produce_time))
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

    def print_optimal(self):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products, self.maximum_produce_time))

        total_waitings = 0
        servings = 0
        missings = 0
        orders = 0


        for step in range(self.horizon):
            inventory, delivery_matrix = min_state[:, 0], min_state[:, 1:]
            action_table = self.optimal_actions[step]
            action = action_table[min_state.tobytes()]
            logging.error(f'step {step}')
            logging.error(f'inventory \n {inventory}')
            logging.error(f'latest delivery \n {delivery_matrix}')
            order = None
            if len(self._injected_orders) == 0:
                order = self.make_orders(step, train=False)
            else:
                order = self.vectorize_order(self.make_orders(step, train=False))


            inventory_net = np.maximum([0., 0.],inventory-order+action)
            total_waitings += sum((inventory_net).tolist())
            servings += sum(action.tolist())
            missing = np.maximum([0., 0.],order-action-inventory)
            missings += sum(missing.tolist())


            logging.error(f'order \n {order}')
            logging.error(f'delivery \n {action}')

            cost, act, min_state = self.take_action(min_state, order, 0, action, step)

        mean_waiting = total_waitings/self.horizon
        miss_ratio = servings/(missings + servings)
        logging.error(f'product_wait_ratio \n {mean_waiting}')
        logging.error(f'sale_miss_ratio \n {miss_ratio}')
        return mean_waiting, miss_ratio


    def print(self):
        min_state = np.reshape(np.frombuffer(min(self.lookup_table, key=self.lookup_table.get), dtype="int64"),
                               (self.number_of_products, self.maximum_produce_time))

        for step in range(self.horizon):

            if self.maximum_produce_time > 1:
                inventory, delivery_matrix = min_state[:, 0], min_state[:, 1:]
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
    #injection = [[1,0],[2,0]] # order 1 of first product in first time step, 2 of first product in second time step
    result_dic = {}
    with open("../inventory.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    seeds = config['seeds']
    total_mean_waiting = 0
    total_miss_ratio = 0
    for ind, inj in enumerate(seeds):
        agent = DPAgent(config_path="../inventory.yaml", loglevel=logging.CRITICAL)
        agent.inject_prophecy(inj)
        agent.train()
        mean_waiting, miss_ratio = agent.print_optimal()
        total_mean_waiting += mean_waiting
        total_miss_ratio += miss_ratio
        result_dic[f"seed:{ind}"] = (f"miss_ratio: {miss_ratio}", f"mean_waiting: {mean_waiting}")

    result_dic[f"average_miss_ratio_inv_{MAXIMUM_INVENTORY}_del_{MAXIMUM_DELIVERY}"] = total_miss_ratio/len(seeds)
    result_dic[f"average_mean_waiting_inv_{MAXIMUM_INVENTORY}_del_{MAXIMUM_DELIVERY}"] = total_mean_waiting/len(seeds)

    print(result_dic)

#%%
#print("------------print finishe------------")
#cost = agent.cost_of_actions([])
#print(f'total cost: {cost}')