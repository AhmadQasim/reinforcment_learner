import os

import gym
from gym import wrappers
from agents.random_agent import RandomAgent
import gym_baking
# from agents.cem import CEMAgent


env_id = 'CartPole-v0'
num_episodes = 10
num_timesteps = 50

outdir = './agent-results/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

env = gym.make(env_id)
# env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = RandomAgent(env.action_space)
# agent = CEMAgent(env.action_space)
reward = 0
done = False

for episode in range(num_episodes):
    observation = env.reset()
    for timestep in range(num_timesteps):
        env.render()
        action = agent.act(observation, reward, done)
        observation, reward, done, info = env.step(action)
        print(observation, reward, action)
        if done:
            print('Episode finished after {} timesteps'.format(timestep))
            break
env.close()
