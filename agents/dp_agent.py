import gym
import numpy as np
from collections import Counter
#import gym_baking.envs.utils as utils

class DynamicProgramming():
    def __init__(self):
        self.env = gym.make("gym_baking:Inventory-v0", config_path="inventory.yaml")

        self.production_time = self.env._producer_model.config[0]['production_time']

        self.number_actions = self.env.action_space.high[1] + 1
        self.number_products = len(self.env.product_list)
        #self.consumer_demand = np.zeros([self.env.episode_max_steps, self.number_products])
        # init random or load from somewhere
        #self.consumer_demand[0:50] = np.random.randint(31, size=(50,1))

        #consumer demand oracle
        self.consumer_demand = np.load('../reinforcement_learner/consumer_demand.npy')

        self.value_function = np.zeros(self.env.episode_max_steps)

        # Add the amount to be produced at time_step for which product, rest of row is zero
        self.policy = np.zeros([self.env.episode_max_steps, self.number_products])
        self.theta = 0.0001
        self.discount_factor = 0.9

    def one_step_lookahead(self, state, product_counter, order_counter):
        expected_action_value_vector = np.zeros(self.number_actions)

        for a in range(self.number_actions):
            # How for more than one product? Currently not taking different cost for products into account nor cost
            # for producing: c(delivered).
            # reward = inventory = sum over all products: r(inv + delivered - demand) // r = abs
            # Idea consumer_demand[state + delivery time?]
            if state < self.env.episode_max_steps-1:
                expected_action_value_vector[a] = (abs(sum(product_counter.values()) + a - sum(self.consumer_demand[state+1]) - sum(order_counter.values())) + self.discount_factor * self.value_function[state+1])
            else:
                expected_action_value_vector[a] = abs(sum(product_counter.values()) + a - sum(self.consumer_demand[state]) - sum(order_counter.values()))
        return expected_action_value_vector

    def act(self, observation, reward, done):
        # make an observation of inventory x and orders o
        inventory_product_list = observation['inventory_state']['products']
        product_counter = Counter(getattr(product, '_item_type') for product in inventory_product_list)

        order_queue_list = observation['consumer_state']['order_queue']
        order_counter = Counter(getattr(order, '_item_type') for order in order_queue_list)

        while True:
            delta = 0.0
            # ToDo: current step and forward only
            for state in range(self.env.episode_max_steps):
                if state < self.env.timestep:
                    continue
                expected_action_values = self.one_step_lookahead(state, product_counter, order_counter)
                best_action_value = np.min(expected_action_values)

                delta = max(delta, np.abs(best_action_value - self.value_function[state]))

                self.value_function[state] = best_action_value

            if delta < self.theta:
                #print(self. value_function)
                expected_action_values = self.one_step_lookahead(self.env.timestep, product_counter, order_counter)
                best_action = np.argmin(expected_action_values)
                #print('Best action ' + str(best_action) + ' after ' + str(i) + ' value iterations.')
                #Todo: hack for fixed product ID
                return [0, best_action]
