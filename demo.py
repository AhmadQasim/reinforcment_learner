import argparse
import os
import gym
from gym import wrappers
from agents.random_agent import RandomAgent


'''
demo script for gym environment and random agent
'''

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--env_id', default='CartPole-v0', type=str, help="define a gym environment [CartPole-v0, gym_baking:Baking-v0]")
parser.add_argument('-n', '--num_episodes', default=10, type=int)
parser.add_argument('-t', '--num_timesteps', default=100, type=int)
parser.add_argument('-o', '--output_dir', default='agent-results')
parser.add_argument('-l', '--log', action='store_true')
ap = parser.parse_args()

if not os.path.exists(ap.output_dir):
    os.makedirs(ap.output_dir)

env = gym.make(ap.env_id)
agent = RandomAgent(env.action_space)
env.seed(0)

if not ap.log:
    for episode in range(ap.num_episodes):
        observation = env.reset()
        reward = 0
        done = False
        for timestep in range(ap.num_timesteps):
            env.render()
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            if done:
                print('Episode finished after {} timesteps'.format(timestep))
                break
    env.close()


else:
    """
    demo of gym.wrappers.Monitor
    """
    print('start logging into folder: {}'.format(ap.output_dir))
    env = wrappers.Monitor(env, ap.output_dir, force=True)

    for episode in range(5):
        env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

    env.close()