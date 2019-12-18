import gym
import numpy as np
from collections import Counter
#import gym_baking.envs.utils as utils

class DynamicProgramming():
    def __init__(self):
        self.env = gym.make("gym_baking:Inventory-v0", config_path="inventory.yaml")
        #self.value_function = np.zeros(self.env.episode_max_steps)
        # self.production_time = self.env._producer_model.config[0]['production_time']
        self.number_actions = self.env.action_space.high[1] + 1
        self.number_products = len(self.env.product_list)
        self.consumer_demand = np.zeros([self.env.episode_max_steps, self.number_products])
        # init random or load from somewhere
        self.consumer_demand[0:50] = random_init_demand = np.random.randint(31, size=(50,1))
        self.value_function = np.zeros(self.env.episode_max_steps)
        #Todo: maybe track policies over time to show changing prediction.
        self.policy = np.zeros([self.env.episode_max_steps, self.number_products])
        self.theta = 0.0001
        self.discount_factor = 0.9

    def act(self, observation, reward, done):
        inventory_product_list = observation['inventory_state']['products']
        product_counter = Counter(getattr(product, '_item_type') for product in inventory_product_list)

        order_queue_list = observation['consumer_state']['order_queue']
        order_counter = Counter(getattr(order, '_item_type') for order in order_queue_list)

        delta = 0.0
        i = 0
        while True:
            i += 1
            # ToDo: current step and forward only
            for step in range(self.env.episode_max_steps):
                if step < self.env.timestep:
                    continue
                A = self.one_step_lookahead(step, product_counter, order_counter)
                best_action_value = np.min(A)

                delta = max(delta, np.abs(best_action_value - self.value_function[step]))

                self.value_function[step] = best_action_value

            if delta < self.theta or i == 20:
                #print(self. value_function)
                A = self.one_step_lookahead(self.env.timestep, product_counter, order_counter)
                best_action = np.argmin(A)
                #print('Best action ' + str(best_action) + ' after ' + str(i) + ' value iterations.')
                #Todo: hack for fixed product ID
                return [0, best_action]

        # Create a deterministic policy using the optimal value function
        #for step in range(self.env.episode_max_steps):
        #    # One step lookahead to find the best action for this state
        #    A = self.one_step_lookahead(step, product_counter)
        #    best_action = np.argmin(A)
        #    # Always take the best action
        #    self.policy[step, best_action] = 1.0



    def one_step_lookahead(self, state, product_counter, order_counter):
        A = np.zeros(self.number_actions)
        for item in range(self.number_products):
            for a in range(self.number_actions):
                # How for more than one product? Currently not taking different cost for products into account nor cost
                # for producing: c(delivered).
                # reward = inventory = sum over all products: r(inv + delivered - demand) // r = abs
                # Idea consumer_demand[state + delivery time?]
                if state < self.env.episode_max_steps-1:
                    A[a] = (abs(sum(product_counter.values()) + a - sum(self.consumer_demand[state+1]) - sum(order_counter.values())) + self.discount_factor * self.value_function[state+1])
                else:
                    A[a] = abs(sum(product_counter.values()) + a - sum(self.consumer_demand[state]) - sum(order_counter.values()))
            return A