import gym


env = gym.make('CartPole-v0')
env = gym.make('gym_baking:Baking-v0')
env.reset()

for episode in range(10):
    for timestep in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} timesteps'.format(timestep))
            break
    env.reset()
env.close()
