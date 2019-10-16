import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding

from ._demand_function import DemandDecision
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
        'video.frames_per_second': 50,
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

        self.timestamp = 0
        self.state = None
        self.fig = None
        self.axes = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        prev_inventory, prev_demand = state

        # generate random orders
        new_demand = self.buyer.binary_buy(self.timestamp)
        curr_demand = prev_demand + new_demand

        # produce new items
        curr_inventory = prev_inventory + int(action)
        
        # update states
        inventory = max(curr_inventory - curr_demand, 0)
        demand = max(curr_demand - curr_inventory, 0)
        self.state = (inventory, demand)
        self.acts.append([action, new_demand])
        self.obs.append([self.state[0], self.state[1]])
        self.timestamp += 1

        done = inventory >= self.inventory_threshold or demand > self.demand_threshold
        
        # reward function
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
        else:
            reward = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.timestamp = 0
        self.state = self.np_random.randint(0, 10, 2)
        self.obs = list()
        self.acts = list()
        return np.array(self.state)

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(2,1)
            plt.ion()

        self.axes[0].clear()
        self.axes[1].clear()

        self.axes[0].plot([x[0] for x in self.obs], color='green', label='demand')[0]
        self.axes[0].plot([x[1] for x in self.obs], color='red', label='inventory')[0]
        self.axes[0].legend()

        self.axes[1].plot([x[0] for x in self.acts], color='green', label='do not bake')
        self.axes[1].plot([x[1] for x in self.acts], color='red', label='bake')
        self.axes[1].legend()

        plt.draw()
        plt.pause(0.001)

        # return image (Numpy Array) representation of the rendered figure
        return np.array(self.fig.canvas.renderer.buffer_rgba())

    def close(self):
        if self.fig:
            plt.close()
            self.fig = None
            self.axes = None