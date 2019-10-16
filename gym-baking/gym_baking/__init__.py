from gym.envs.registration import register

register(
    id='Baking-v0',
    entry_point = 'gym_baking.envs:BakingEnv',
)

register(
    id='Inventory-v0',
    entry_point = 'gym_baking.envs:InventoryTrackingEnv',
)