import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

from .demand_function import DemandDecision
import matplotlib.pyplot as plt

class BakingEnv(gym.Env):
    """
    Description
    Baker agent makes baking decision (true, false) at each timestamp, the environment makes purchasing decision according to some implicit distributions

    Observation:
        Type: Scalar
        Num Observation     Min     Max
        0   demand          0       50
        1   inventory       0       50

    Actions:
        Type: Discrete(2)
        Num Action
        0   do not bake
        1   bake

    Reward:
        Reward is calculated based on current inventory, demand, new_demand and action

    Starting State:
        Both observations are assigned a uniform random value in [0, 10]

    Episode Termination:
        Num of demand or inventory is greater than 50
        Episode length is greater than 200
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.timestamp = 0
        self.inventory = 0
        self.demand = 0
        self.buyer = DemandDecision()

        self.demand_threshold = 50
        self.inventory_threshold = 50

        self.action_space = spaces.Discrete(2)
        self.observation_space = np.zeros((2,))
        self.seed()

        self.state = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        timestamp, prev_inventory, prev_demand = state #possible issue about time ordering
        # prev_inventory = curr_inventory; prev_demand = curr_demand
        new_demand = self.buyer.binary_buy(timestamp)
        # print(int(new_demand))
        curr_demand = prev_demand + new_demand
        curr_inventory = prev_inventory + int(action)
        inventory = max(curr_inventory - curr_demand, 0)
        demand = max(curr_demand - curr_inventory, 0)
        timestamp += 1
        self.state = (timestamp, inventory, demand)
        self.acts.append([action, new_demand])
        self.obs.append([self.state[1], self.state[2]])

        done = inventory >= self.inventory_threshold or demand > self.demand_threshold
        if not done:
            if prev_demand>0:
                if action==1:
                    reward = 2 if new_demand else 1
                else:
                    reward = -2 if new_demand else -1
            elif prev_inventory>0:
                if action == 0:
                    reward = 2 if new_demand else 1
                else:
                    reward = -1 if new_demand else -2
            else:
                reward = 2 if action==new_demand else -2

            # reward = 1 - (inventory + demand)
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.insert(self.np_random.randint(0, 10, 2), 0, 0)
        self.obs = list()
        self.acts = list()
        if self.viewer is not None: self.viewer.clear();self.viewer1.clear()
        return np.array(self.state)

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400


        if self.viewer is None:
            fig = plt.figure()
            self.viewer = fig.add_subplot(211)
            self.viewer1 = fig.add_subplot(212)
            plt.ion()
            plt.show()
        # print(self.obs[-1])
        self.viewer.plot([x[0] for x in self.obs], color='green')
        self.viewer.plot([x[1] for x in self.obs], color='red')
        self.viewer1.plot([x[0] for x in self.acts], color='green')
        self.viewer1.plot([x[1] for x in self.acts], color='red')
        plt.pause(0.1)


    def close(self):
        plt.savefig('result.png')
        plt.ioff()
        plt.show()
        plt.close()


