#import fire
import gym
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gym_baking.envs.utils as utils

#from agents.base_agent import BaseAgent


class BaselineAgent():
    def __init__(self):
        super().__init__()
        self.config_path = "../reinforcemnet_learner/inventory.yaml"
        f = open(self.config_path, 'r')
        self.config = yaml.load(f, Loader=yaml.Loader)

        self.env = gym.make("gym_baking:Inventory-v0", config_path="../reinforcemnet_learner/inventory.yaml")
        self.observation_space = {'producer_state': {'production_queue': [], 'is_busy': False},
                                  'inventory_state': {'products': []},
                                  'consumer_state': {'order_queue': []}}

        self.items_to_id = utils.map_items_to_id(self.config)
        self.items_count = len(self.items_to_id.keys())

        self.feature_count = 5
        self.state_shape = (self.items_count * self.feature_count,)
        self.action_shape = self.env.action_space.shape

        self.max_steps_per_episode = 100
        self.rewards = []
        self.test_eps = 300
        self.test_rewards = []

        self.min_quantity = 10
        self.max_quantity = 30

    def take_action(self, state):

        action = np.zeros(self.action_shape, dtype=np.int32)

        # produce item which is below threshold
        for item in range(self.items_count):
            if item not in state[1].keys() or state[1][item][0] < self.min_quantity:
                action[0] = item
                action[1] = self.max_quantity

        # take action
        new_observation, reward, done, info = self.env.step(action)

        return new_observation, reward, done

    def train(self):
        """ No training needed. Baseline agent
         produces whatever falls below threshold."""
        pass

    def test(self):
        total_mean_reward = []
        total_reward = 0

        for ep in range(self.test_eps):
            episode_reward = []
            self.env.reset()
            self.env.step(self.env.action_space.sample())
            obs, reward, done, _ = self.env.step(self.env.action_space.sample())
            obs = utils.observation_state_vector(obs, return_count=True, items_to_id=self.items_to_id)

            for j in range(self.max_steps_per_episode - 2):
                new_observation, reward, done = self.take_action(obs)
                obs = utils.observation_state_vector(new_observation, return_count=True, items_to_id=self.items_to_id)

                reward = reward * 1000
                episode_reward.append(reward)

            # episode summary
            total_reward += sum(episode_reward)
            total_mean_reward.append(total_reward / (ep + 1))
            print("Episode : ", ep)
            print("Episode Reward : ", sum(episode_reward))
            print("Total Mean Reward: ", total_reward / (ep + 1))
            print("==========================================")

        plt.plot(list(range(self.test_eps)), total_mean_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()


if __name__ == "__main__":
#    fire.Fire(BaselineAgent())
