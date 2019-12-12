import gym
import numpy as np
#import gym_baking.envs.utils as utils

class DynamicProgramming():
    def __init__(self):
        self.env = gym.make("gym_baking:Inventory-v0", config_path="inventory.yaml")
        #self.value_function = np.zeros(self.env.episode_max_steps)
        self.number_actions = self.env.action_space.high[1]
        self.number_products = len(self.env.product_list)
        self.consumer_demand = np.zeros((self.env.episode_max_steps, self.number_products))
        self.policy = np.zeros_like(self.env.episode_max_steps)
    def act(self, observation, reward, done):
        prezels_inventory = 0
        brot_inventory = 0
        for item in observation['inventory_state']['products']:
            if item._item_type == 'brot':
                brot_inventory += 1
            else:
                prezels_inventory += 1

        return self.action_space.sample()