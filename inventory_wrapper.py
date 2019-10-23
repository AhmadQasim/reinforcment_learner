import gym
import numpy as np
from collections import Counter

class InventoryQueueToVector(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.product_list = self.observed_product
        self.n = len(self.product_list)
        self.observation_space = np.zeros( self.n*3 * (self.n+1) + self.n+1 ) # 3 for criterion n+1 for action[class, num] + bias

    def observation(self, observation):
        inv = Counter([x._item_type for x in observation["inventory_state"]["products"]])
        prd = Counter([x._item_type for x in observation["producer_state"]["production_queue"]])
        cns = Counter([x._item_type for x in observation["consumer_state"]["order_queue"]])

        new_obs = np.zeros(len(self.product_list)*3)
        i = 0

        for counter in [inv, prd, cns]:
            for key in self.product_list:
                new_obs[i] = counter[key]
                i += 1
        
        return new_obs

if __name__=='__main__':

    env = InventoryQueueToVector(gym.make('gym_baking:Inventory-v0'))

    env.reset()
    for i in range(5):
        for t in range(100):
            env.render()
            a_t = env.action_space.sample()
            s_t, r_t, done, info = env.step(a_t)
            print(s_t)
        if done:
            s_t = env.reset()