import gym
from gym.envs.registration import register

envs = gym.envs.registry.env_specs

if 'Baking-v0' not in envs.keys():
    register(
        id='Baking-v0',
        entry_point='gym_baking.envs:BakingEnv',
    )
if 'Inventory-v0' not in envs.keys():
    register(
        id='Inventory-v0',
        entry_point='gym_baking.envs:InventoryManagerEnv',
    )
