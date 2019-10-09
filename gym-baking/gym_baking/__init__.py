from gym.envs.registration import register

register(id='Baking-v0',
         entry_point = 'gym_baking.envs:BakingEnv',)
