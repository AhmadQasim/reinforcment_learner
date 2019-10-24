## Example Learning with CEM-Agent
extract information from observation space
```
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
```

## Example LinearPolicy
```
class LinearActionLinearPolicy(object):
    def __init__(self, theta):
        # theta: [(n * 3) * (n + 1) + (n+1)]
        assert len(theta) == 21 # only supports two products classes
        self.w = theta[:-3].reshape(6,3)
        self.b = theta[-3:]
    def act(self,ob):
        action = np.zeros(2, dtype=np.int64)
        y = ob.dot(self.w) + self.b
        action[0] = np.argmax(y[:2]) # classification: which product to produce
        action[1] = min(max(0, int(y[2])), 30) # amount of product to produce
        return action
```
## Example Training Logs
```
INFO: Making new env: gym_baking:Inventory-v0 ({'config_path': 'inventory.yaml'})
Iteration  0. Episode mean reward:   0.422
{'sales': 184, 'waits': 77, 'products': 184, 'wastes': 0, 'sale_wait_ratio': 0.7049808429118773, 'product_waste_ratio': 1.0}
Iteration  1. Episode mean reward:   0.719
Iteration  2. Episode mean reward:   0.852
Iteration  3. Episode mean reward:   0.846
Iteration  4. Episode mean reward:   0.902
Iteration  5. Episode mean reward:   0.903
{'sales': 233, 'waits': 14, 'products': 240, 'wastes': 7, 'sale_wait_ratio': 0.9433198380566802, 'product_waste_ratio': 0.97165991902834}
```
